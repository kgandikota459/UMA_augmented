r"""Reference anchors and physical penalties.

Quantum anchors (L_quantum): compatible quantum labels (same level of theory
as the checkpoint's task; for `omol`, OMol25's wB97M-V/def2-TZVPD) constrain
forces and *energy differences* within a structure set. Absolute energies
are gauge-dependent (element references, per-atom offsets), so only
differences within a set sharing a composition are used unless an explicit
reference model is fitted.

Baseline drift (L_baseline): distance between U_w and the frozen U_{w0} on
replay configurations (forces and centred energies). It limits drift; it is
not experimental truth.

Physical penalties (L_physical): target identified weaknesses only, e.g.
short-range collapse along dimer compressions (energy must rise as r->0).
Arbitrary regularizers do not guarantee physical correctness.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..potential.base import Potential
from ..systems.system import System


@dataclass
class AnchorSet:
    """Configurations of one system (same composition) with reference labels."""

    system: System
    positions: torch.Tensor  # (M, N, 3)
    energies: torch.Tensor | None  # (M,) eV, same theory as the UMA task
    forces: torch.Tensor | None  # (M, N, 3) eV/A
    theory: str
    source: str


def quantum_anchor_loss(potential: Potential, anchors: list[AnchorSet], energy_sd: float = 0.02,
                        force_sd: float = 0.05) -> torch.Tensor:
    total = torch.zeros((), dtype=torch.float64)
    n = 0
    for a in anchors:
        pos = a.positions.detach().requires_grad_(True)
        with torch.enable_grad():
            e = potential.energy(a.system, pos)
            (g,) = torch.autograd.grad(e.sum(), pos, create_graph=True)
        if a.energies is not None and len(e) > 1:
            de = (e - e.mean()) - (a.energies - a.energies.mean())  # differences only
            total = total + 0.5 * ((de / energy_sd) ** 2).sum()
            n += len(e) - 1
        if a.forces is not None:
            total = total + 0.5 * (((-g - a.forces) / force_sd) ** 2).sum() / a.system.n_atoms
            n += len(e)
    return total / max(n, 1)


def baseline_drift_loss(potential: Potential, baseline: Potential, system: System, positions: torch.Tensor,
                        energy_scale: float = 0.01, force_scale: float = 0.05) -> torch.Tensor:
    pos = positions.detach()
    with torch.no_grad():
        e0, f0 = baseline.energy_and_forces(system, pos)
    x = pos.clone().requires_grad_(True)
    with torch.enable_grad():
        e = potential.energy(system, x)
        (g,) = torch.autograd.grad(e.sum(), x, create_graph=True)
    de = (e - e.mean()) - (e0 - e0.mean())
    df = (-g) - f0
    return 0.5 * ((de / energy_scale) ** 2).mean() + 0.5 * ((df / force_scale) ** 2).mean()


def short_range_repulsion_penalty(potential: Potential, system: System, i: int, j: int,
                                  reference_positions: torch.Tensor, r_values: torch.Tensor,
                                  margin: float = 0.0) -> torch.Tensor:
    """Penalize energy *decreasing* as the (i, j) distance is compressed.

    Atom j is moved along the i->j axis to each distance in `r_values`.
    With energies sorted by increasing r, each consecutive pair must satisfy
    E(r_k) - E(r_{k+1}) >= margin; violations are penalized linearly.
    """
    x0 = reference_positions.detach()
    u = x0[j] - x0[i]
    u = u / u.norm()
    frames = []
    for r in r_values:
        x = x0.clone()
        x[j] = x0[i] + r * u
        frames.append(x)
    X = torch.stack(frames)
    e = potential.energy(system, X)
    order = torch.argsort(r_values)
    e_sorted = e[order]
    # energy must increase as r decreases: E(r_small) - E(r_large) >= margin
    return torch.relu(margin - (e_sorted[:-1] - e_sorted[1:])).sum()


@dataclass
class GaussianPrior:
    """Prior on a calibrated nuisance parameter phi (e.g. a frequency scale)."""

    mean: float
    sd: float

    def nll(self, phi: torch.Tensor) -> torch.Tensor:
        return 0.5 * ((phi - self.mean) / self.sd) ** 2
