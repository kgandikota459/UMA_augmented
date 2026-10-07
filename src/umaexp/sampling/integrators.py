"""Samplers in PyTorch: Langevin BAOAB (NVT), MC barostat (NPT), velocity Verlet (NVE).

Units: eV, A, u; time in the internal unit tau (units.FS converts fs -> tau);
velocities in A/tau. Sampling never needs parameter gradients: forces are
computed with create_graph=False and detached.

BAOAB (Leimkuhler & Matthews, AMRX 2013) has O(dt^2) configurational bias and
excellent configurational accuracy for its cost. The Monte Carlo barostat
proposes isotropic volume changes in ln V, scaling molecular centres of mass
(molecules move rigidly), and accepts with

    min(1, exp(-beta [dU + p dV] + (N_mol + 1) ln(V'/V)))

where the +1 is the Jacobian of the ln V proposal. This samples the NPT
ensemble of the molecular-COM representation (Frenkel & Smit, ch. 5.4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .. import units
from ..potential.base import Potential
from ..systems.system import System


@dataclass
class MDState:
    positions: torch.Tensor  # (N, 3)
    velocities: torch.Tensor  # (N, 3) A/tau
    cell: torch.Tensor | None  # (3, 3)
    forces: torch.Tensor | None = None
    energy: float | None = None


def maxwell_boltzmann(masses: torch.Tensor, temperature_k: float, generator: torch.Generator,
                      remove_com: bool = True) -> torch.Tensor:
    kt = units.KB_EV_K * temperature_k
    v = torch.randn(masses.shape[0], 3, dtype=torch.float64, generator=generator) * torch.sqrt(kt / masses)[:, None]
    if remove_com:
        p = (masses[:, None] * v).sum(0)
        v = v - p / masses.sum()
    return v


def kinetic_energy(masses, v):
    return 0.5 * float((masses[:, None] * v * v).sum())


def instantaneous_temperature(masses, v, n_constraints: int = 3):
    dof = 3 * masses.shape[0] - n_constraints
    return 2.0 * kinetic_energy(masses, v) / (dof * units.KB_EV_K)


class ForceEvaluator:
    """Energy + forces on the sampling device, returned as float64 CPU."""

    def __init__(self, potential: Potential, system: System, device: str = "cpu", dtype=torch.float64):
        self.potential = potential
        self.system = system
        self.device = device
        self.dtype = dtype
        self.n_calls = 0

    def __call__(self, pos: torch.Tensor, cell: torch.Tensor | None):
        self.n_calls += 1
        # cast on the source device, then move (MPS cannot hold float64)
        x = pos.detach().to(self.dtype).to(self.device)
        c = None if cell is None else cell.detach().to(self.dtype).to(self.device)
        e, f = self.potential.energy_and_forces(self.system, x, c)
        return float(e), f.detach().to("cpu").to(torch.float64)


class LangevinBAOAB:
    def __init__(self, evaluator: ForceEvaluator, masses: torch.Tensor, temperature_k: float,
                 dt_fs: float, friction_per_ps: float, seed: int):
        self.f = evaluator
        self.m = masses.to(torch.float64)
        self.kt = units.KB_EV_K * temperature_k
        self.temperature = temperature_k
        self.dt = dt_fs * units.FS
        gamma = friction_per_ps / (1000.0 * units.FS)  # 1/tau
        self.c1 = math.exp(-gamma * self.dt)
        self.c2 = math.sqrt(1.0 - self.c1**2)
        self.gen = torch.Generator().manual_seed(int(seed))

    def init_state(self, pos, cell) -> MDState:
        v = maxwell_boltzmann(self.m, self.temperature, self.gen)
        e, f = self.f(pos, cell)
        return MDState(pos.clone().to(torch.float64), v, None if cell is None else cell.clone().to(torch.float64), f, e)

    def step(self, s: MDState) -> MDState:
        inv_m = (1.0 / self.m)[:, None]
        h = 0.5 * self.dt
        v = s.velocities + h * s.forces * inv_m  # B
        x = s.positions + h * v  # A
        noise = torch.randn(v.shape, dtype=torch.float64, generator=self.gen)
        v = self.c1 * v + self.c2 * torch.sqrt(self.kt * inv_m) * noise  # O
        x = x + h * v  # A
        e, f = self.f(x, s.cell)
        v = v + h * f * inv_m  # B
        return MDState(x, v, s.cell, f, e)


class VelocityVerlet:
    """NVE integrator for energy-drift checks."""

    def __init__(self, evaluator: ForceEvaluator, masses: torch.Tensor, dt_fs: float):
        self.f = evaluator
        self.m = masses.to(torch.float64)
        self.dt = dt_fs * units.FS

    def step(self, s: MDState) -> MDState:
        inv_m = (1.0 / self.m)[:, None]
        v = s.velocities + 0.5 * self.dt * s.forces * inv_m
        x = s.positions + self.dt * v
        e, f = self.f(x, s.cell)
        v = v + 0.5 * self.dt * f * inv_m
        return MDState(x, v, s.cell, f, e)


class MonteCarloBarostat:
    def __init__(self, evaluator: ForceEvaluator, system: System, temperature_k: float, pressure_bar: float,
                 seed: int, max_dlnv: float = 0.01, scale_molecules: bool = True):
        self.f = evaluator
        self.beta = units.beta(temperature_k)
        self.p = pressure_bar * units.BAR_TO_EV_A3
        self.max_dlnv = max_dlnv
        self.gen = torch.Generator().manual_seed(int(seed) + 7919)
        self.n_accepted = 0
        self.n_trials = 0
        if scale_molecules and system.molecule_index is not None:
            idx = torch.as_tensor(system.molecule_index, dtype=torch.long)
            self.mol = idx
            self.n_units = int(idx.max()) + 1
        else:
            self.mol = torch.arange(system.n_atoms)
            self.n_units = system.n_atoms
        self.masses = system.mass_tensor()

    def _com(self, pos):
        m = self.masses[:, None]
        tot = torch.zeros(self.n_units, 3, dtype=torch.float64).index_add(0, self.mol, m * pos)
        mm = torch.zeros(self.n_units, dtype=torch.float64).index_add(0, self.mol, self.masses)
        return tot / mm[:, None]

    def attempt(self, s: MDState) -> MDState:
        self.n_trials += 1
        vol = float(torch.linalg.det(s.cell).abs())
        dlnv = (2.0 * torch.rand(1, generator=self.gen, dtype=torch.float64).item() - 1.0) * self.max_dlnv
        new_vol = vol * math.exp(dlnv)
        scale = (new_vol / vol) ** (1.0 / 3.0)
        com = self._com(s.positions)
        shift = (scale - 1.0) * com[self.mol]
        new_pos = s.positions + shift
        new_cell = s.cell * scale
        e_new, f_new = self.f(new_pos, new_cell)
        arg = -self.beta * (e_new - s.energy + self.p * (new_vol - vol)) + (self.n_units + 1) * math.log(new_vol / vol)
        if arg >= 0 or torch.rand(1, generator=self.gen, dtype=torch.float64).item() < math.exp(arg):
            self.n_accepted += 1
            return MDState(new_pos, s.velocities, new_cell, f_new, e_new)
        return s

    @property
    def acceptance(self) -> float:
        return self.n_accepted / max(self.n_trials, 1)

    def adapt(self, target: float = 0.4):
        """Step-size adaptation; call only during equilibration (breaks detailed balance otherwise)."""
        if self.n_trials >= 10:
            acc = self.acceptance
            self.max_dlnv *= 1.1 if acc > target else 0.9
            self.max_dlnv = min(max(self.max_dlnv, 1e-5), 0.2)
            self.n_accepted = self.n_trials = 0
