"""Reference potentials with known answers, used to validate the machinery.

These are *not* models of chemistry. They exist so that estimators,
integrators and gradient paths can be checked against closed-form results
before the same code is pointed at UMA:

  HarmonicWell   <|r - r0|^2> = 3 k_B T / k per atom (exact canonical result)
  MorseDiatomic  harmonic wavenumber sqrt(2 D a^2 / mu) / (2 pi c)
  LennardJones   smooth shifted-force LJ fluid for NVT/NPT sampling tests
  ValenceToy     bonds + angles for polyatomic vibrational/implicit-geometry tests
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..systems.system import System
from .base import Potential, as_batch, minimum_image


class HarmonicWell(Potential):
    """U = 1/2 k sum_i |r_i - r0_i|^2 with trainable log k (eV/A^2)."""

    def __init__(self, k: float = 1.0, center: torch.Tensor | None = None):
        super().__init__()
        self.log_k = nn.Parameter(torch.tensor(math.log(k), dtype=torch.float64))
        self.register_buffer("center", center if center is not None else torch.zeros(1, 3, dtype=torch.float64))

    @property
    def k(self):
        return self.log_k.exp()

    def energy(self, system, positions, cell=None):
        pos, _, single = as_batch(positions.to(torch.float64), None)
        e = 0.5 * self.k * ((pos - self.center) ** 2).sum(dim=(-1, -2))
        return e[0] if single else e


class MorseDiatomic(Potential):
    """U = D (1 - exp(-a (r - re)))^2 for a two-atom system."""

    def __init__(self, D: float = 4.0, a: float = 2.0, re: float = 1.0):
        super().__init__()
        self.D = nn.Parameter(torch.tensor(D, dtype=torch.float64))
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float64))
        self.re = nn.Parameter(torch.tensor(re, dtype=torch.float64))

    def energy(self, system, positions, cell=None):
        pos, _, single = as_batch(positions.to(torch.float64), None)
        r = (pos[:, 0] - pos[:, 1]).norm(dim=-1)
        e = self.D * (1 - torch.exp(-self.a * (r - self.re))) ** 2
        return e[0] if single else e


class LennardJones(Potential):
    """Shifted-force Lennard-Jones, single species, with trainable eps/sigma.

    U = sum_{i<j, r<rc} [phi(r) - phi(rc) - (r - rc) phi'(rc)], which makes
    energy *and* force continuous at the cutoff (smoothness matters for
    stable NVE dynamics and finite-difference checks). Periodic systems use
    the minimum-image convention and require rc < L/2.
    """

    def __init__(self, epsilon: float = 0.0104, sigma: float = 3.40, cutoff: float = 8.5):
        super().__init__()
        self.log_eps = nn.Parameter(torch.tensor(math.log(epsilon), dtype=torch.float64))
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma), dtype=torch.float64))
        self.cutoff = float(cutoff)

    def _phi(self, r, eps, sig):
        sr6 = (sig / r) ** 6
        return 4 * eps * (sr6 * sr6 - sr6)

    def _dphi(self, r, eps, sig):
        sr6 = (sig / r) ** 6
        return 4 * eps * (-12 * sr6 * sr6 + 6 * sr6) / r

    def energy(self, system, positions, cell=None):
        pos, c, single = as_batch(positions.to(torch.float64), None if cell is None else cell.to(torch.float64))
        eps, sig = self.log_eps.exp(), self.log_sigma.exp()
        n = pos.shape[1]
        iu = torch.triu_indices(n, n, offset=1, device=pos.device)
        dr = pos[:, iu[0]] - pos[:, iu[1]]
        if system.is_periodic:
            if c is None:
                raise ValueError("periodic LJ needs a cell")
            dr = torch.stack([minimum_image(dr[b], c[b]) for b in range(pos.shape[0])])
            lmin = torch.stack([torch.linalg.det(c[b]).abs() ** (1 / 3) for b in range(pos.shape[0])]).min()
            if 2 * self.cutoff > float(lmin) + 1e-9:
                raise ValueError(f"cutoff {self.cutoff} exceeds half the box ({float(lmin):.2f} A)")
        r = dr.norm(dim=-1)
        rc = torch.tensor(self.cutoff, dtype=torch.float64, device=pos.device)
        inside = r < rc
        r_safe = torch.where(inside, r, rc)
        u = self._phi(r_safe, eps, sig) - self._phi(rc, eps, sig) - (r_safe - rc) * self._dphi(rc, eps, sig)
        e = torch.where(inside, u, torch.zeros_like(u)).sum(-1)
        return e[0] if single else e


class ValenceToy(Potential):
    """Harmonic bonds and angles over an explicit topology (trainable constants).

    bonds: list of (i, j); angles: list of (i, j, k) with j the vertex.
    Parameters are shared per bond/angle *type* index to mimic transferable
    force-field parameters.
    """

    def __init__(self, bonds, angles, kb, r0, ka, theta0, bond_types=None, angle_types=None):
        super().__init__()
        self.register_buffer("bonds", torch.as_tensor(bonds, dtype=torch.long).reshape(-1, 2))
        self.register_buffer("angles", torch.as_tensor(angles, dtype=torch.long).reshape(-1, 3))
        nb, na = len(self.bonds), len(self.angles)
        self.register_buffer("bond_types", torch.as_tensor(bond_types if bond_types is not None else [0] * nb))
        self.register_buffer("angle_types", torch.as_tensor(angle_types if angle_types is not None else [0] * na))
        f64 = dict(dtype=torch.float64)
        self.kb = nn.Parameter(torch.as_tensor(kb, **f64).reshape(-1))
        self.r0 = nn.Parameter(torch.as_tensor(r0, **f64).reshape(-1))
        self.ka = nn.Parameter(torch.as_tensor(ka, **f64).reshape(-1))
        self.theta0 = nn.Parameter(torch.as_tensor(theta0, **f64).reshape(-1))

    def energy(self, system, positions, cell=None):
        pos, _, single = as_batch(positions.to(torch.float64), None)
        e = torch.zeros(pos.shape[0], dtype=torch.float64, device=pos.device)
        if len(self.bonds):
            i, j = self.bonds[:, 0], self.bonds[:, 1]
            r = (pos[:, i] - pos[:, j]).norm(dim=-1)
            t = self.bond_types
            e = e + 0.5 * (self.kb[t] * (r - self.r0[t]) ** 2).sum(-1)
        if len(self.angles):
            i, j, k = self.angles.T
            u = pos[:, i] - pos[:, j]
            v = pos[:, k] - pos[:, j]
            cos = (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1))
            th = torch.acos(cos.clamp(-1 + 1e-12, 1 - 1e-12))
            t = self.angle_types
            e = e + 0.5 * (self.ka[t] * (th - self.theta0[t]) ** 2).sum(-1)
        return e[0] if single else e


def lj_argon_system(n_side: int = 4, density_g_cm3: float = 1.40) -> System:
    """Simple-cubic argon lattice start (to be melted/equilibrated by MD)."""
    import numpy as np

    from ..units import U_PER_A3_TO_G_CM3
    from ..systems.isotopes import standard_atomic_weight

    n = n_side**3
    m = standard_atomic_weight(18)
    vol = n * m / (density_g_cm3 / U_PER_A3_TO_G_CM3)
    L = vol ** (1 / 3)
    g = (np.arange(n_side) + 0.5) * L / n_side
    pos = np.array([[x, y, z] for x in g for y in g for z in g])
    return System(
        system_id=f"LJ_Ar_{n}",
        atomic_numbers=(18,) * n,
        masses=(m,) * n,
        charge=0,
        spin=0,
        task="toy",
        pbc=(True, True, True),
        molecule_index=tuple(range(n)),
        positions=pos,
        cell=np.eye(3) * L,
        mass_convention="standard_atomic_weight",
        provenance={"builder": "lj_argon_system"},
    )
