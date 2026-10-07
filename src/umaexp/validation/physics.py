"""Physical-consistency checks for a (fine-tuned) potential.

These are acceptance gates, run before and after fitting:
  symmetry        E invariant, F equivariant under rotation/translation,
                  E invariant under permutation of identical atoms
  smoothness      E and F continuous along a separation scan through the cutoff
  energy drift    NVE total-energy drift per atom per ps (small dt)
  stability       short Langevin run: finite energies, no atom overlaps
"""

from __future__ import annotations

import numpy as np
import torch

from ..potential.base import Potential
from ..sampling.integrators import ForceEvaluator, LangevinBAOAB, MDState, VelocityVerlet, maxwell_boltzmann
from ..systems.system import System


def _rotation(seed):
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(seed)))
    return q * torch.sign(torch.det(q))


def symmetry_check(pot: Potential, system: System, pos: torch.Tensor, n_trials: int = 3) -> dict:
    pos = pos.detach().to(torch.float64)
    e0, f0 = pot.energy_and_forces(system, pos)
    de, df = 0.0, 0.0
    for k in range(n_trials):
        R = _rotation(k)
        t = torch.randn(3, dtype=torch.float64, generator=torch.Generator().manual_seed(100 + k))
        e1, f1 = pot.energy_and_forces(system, pos @ R.T + t)
        de = max(de, abs(float(e1 - e0)))
        df = max(df, float((f1 - f0 @ R.T).abs().max()))
    dp = 0.0
    z = np.asarray(system.atomic_numbers)
    for el in np.unique(z):
        idx = np.where(z == el)[0]
        if len(idx) >= 2:
            perm = np.arange(len(z))
            perm[idx[0]], perm[idx[1]] = idx[1], idx[0]
            dp = max(dp, abs(float(pot.energy(system, pos[perm]) - e0)))
    return {"max_rot_trans_energy_dev": de, "max_force_equivariance_dev": df, "max_permutation_energy_dev": dp}


def cutoff_smoothness(pot: Potential, system: System, pos: torch.Tensor, i: int, j: int,
                      r_min: float, r_max: float, n: int = 200) -> dict:
    """Scan |r_j - r_i| and report the largest finite-difference jumps in E and F."""
    pos = pos.detach().to(torch.float64)
    u = pos[j] - pos[i]
    u = u / u.norm()
    rs = torch.linspace(r_min, r_max, n, dtype=torch.float64)
    E, F = [], []
    for r in rs:
        x = pos.clone()
        x[j] = x[i] + r * u
        e, f = pot.energy_and_forces(system, x)
        E.append(float(e))
        F.append(float((f[j] * u).sum()))
    E, F = np.array(E), np.array(F)
    dr = float(rs[1] - rs[0])
    dE = np.abs(np.diff(E))
    dF = np.abs(np.diff(F))
    # consistency: -dE/dr vs projected force (conservativeness along the path)
    fd = -np.diff(E) / dr
    fm = 0.5 * (F[1:] + F[:-1])
    return {"max_energy_step": float(dE.max()), "max_force_step": float(dF.max()),
            "max_force_path_inconsistency": float(np.abs(fd - fm).max()), "r": rs.numpy(), "E": E, "F": F}


def nve_drift(pot: Potential, system: System, pos: torch.Tensor, cell=None, temperature_k: float = 300.0,
              dt_fs: float = 0.25, n_steps: int = 400, seed: int = 0) -> dict:
    ev = ForceEvaluator(pot, system)
    m = system.mass_tensor()
    v = maxwell_boltzmann(m, temperature_k, torch.Generator().manual_seed(seed))
    e, f = ev(pos, cell)
    s = MDState(pos.detach().to(torch.float64), v, cell, f, e)
    integ = VelocityVerlet(ev, m, dt_fs)
    tot = []
    for _ in range(n_steps):
        s = integ.step(s)
        tot.append(s.energy + 0.5 * float((m[:, None] * s.velocities**2).sum()))
    tot = np.array(tot)
    t_ps = np.arange(n_steps) * dt_fs / 1000.0
    slope = np.polyfit(t_ps, tot, 1)[0]
    return {"drift_eV_per_atom_per_ps": float(slope / system.n_atoms),
            "fluctuation_eV_per_atom": float(tot.std() / system.n_atoms)}


def stability_run(pot: Potential, system: System, pos: torch.Tensor, cell=None, temperature_k: float = 300.0,
                  n_steps: int = 500, dt_fs: float = 0.5, min_distance: float = 0.5, seed: int = 0) -> dict:
    ev = ForceEvaluator(pot, system)
    integ = LangevinBAOAB(ev, system.mass_tensor(), temperature_k, dt_fs, 1.0, seed)
    s = integ.init_state(pos.detach().to(torch.float64), cell)
    dmin = float("inf")
    for k in range(n_steps):
        s = integ.step(s)
        if not np.isfinite(s.energy):
            return {"stable": False, "step": k, "reason": "non-finite energy"}
        d = torch.cdist(s.positions, s.positions) + torch.eye(system.n_atoms, dtype=torch.float64) * 1e9
        dmin = min(dmin, float(d.min()))
        if dmin < min_distance:
            return {"stable": False, "step": k, "reason": f"atoms closer than {min_distance} A"}
    return {"stable": True, "min_distance_A": dmin, "final_energy": s.energy}


def extensivity_check(pot: Potential, system: System, pos: torch.Tensor, separation: float = 20.0) -> dict:
    """E(two copies beyond the cutoff) - 2 E(one copy), aperiodic systems.

    For UMA the MoLE routing depends on the *reduced* composition, so two
    distant copies use the same expert mixture and the energy must be exactly
    additive. (FAIRChem notes a UMA-S-1.2 composition-embedding bug that breaks
    this; it is keyed on model_id.)
    """
    from dataclasses import replace

    pos = pos.detach().to(torch.float64)
    shift = torch.tensor([separation, 0.0, 0.0], dtype=torch.float64)
    two = replace(system, system_id=f"{system.system_id}x2",
                  atomic_numbers=system.atomic_numbers * 2, masses=system.masses * 2,
                  charge=system.charge * 2 if system.task == "omol" else 0,
                  spin=system.spin, molecule_index=None,
                  positions=None)
    e1 = float(pot.energy(system, pos))
    e2 = float(pot.energy(two, torch.cat([pos, pos + shift])))
    return {"E_single": e1, "E_pair": e2, "deviation_eV": e2 - 2 * e1}
