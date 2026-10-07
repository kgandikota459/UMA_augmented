"""Additive energy correction U = U_{w0} + dU_psi (comparator only).

With UMA frozen this does NOT satisfy the project's central requirement of
updating UMA's own weights (`updates_uma_weights = False`). It is kept as an
informative ablation: if a tiny pairwise correction explains an experimental
residual as well as updating UMA does, the data do not constrain much beyond
pair structure.

dU is a smooth, conservative, permutation/rotation/translation-invariant sum
over element-pair radial Gaussians with a cosine cutoff, initialised to zero.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .base import Potential, as_batch, minimum_image


class PairCorrection(nn.Module):
    def __init__(self, elements: list[int], n_basis: int = 8, r_min: float = 0.8,
                 cutoff: float = 6.0, width: float | None = None):
        super().__init__()
        self.elements = sorted(set(int(z) for z in elements))
        idx = {z: i for i, z in enumerate(self.elements)}
        self.register_buffer("z_index", torch.full((119,), -1, dtype=torch.long))
        for z, i in idx.items():
            self.z_index[z] = i
        ne = len(self.elements)
        self.n_pairs = ne * (ne + 1) // 2
        self.coef = nn.Parameter(torch.zeros(self.n_pairs, n_basis, dtype=torch.float64))
        self.register_buffer("centers", torch.linspace(r_min, cutoff, n_basis, dtype=torch.float64))
        self.width = width or (cutoff - r_min) / n_basis
        self.cutoff = cutoff
        self.ne = ne

    def pair_index(self, zi, zj):
        a, b = self.z_index[zi], self.z_index[zj]
        lo, hi = torch.minimum(a, b), torch.maximum(a, b)
        return lo * self.ne - lo * (lo - 1) // 2 + (hi - lo)

    def forward(self, z, pos, cell=None, periodic=False):
        n = pos.shape[-2]
        iu = torch.triu_indices(n, n, offset=1, device=pos.device)
        dr = pos[..., iu[0], :] - pos[..., iu[1], :]
        if periodic:
            dr = torch.stack([minimum_image(dr[b], cell[b]) for b in range(pos.shape[0])])
        r = dr.norm(dim=-1)
        fc = torch.where(r < self.cutoff, 0.5 * (torch.cos(math.pi * r / self.cutoff) + 1), torch.zeros_like(r))
        basis = torch.exp(-((r[..., None] - self.centers) / self.width) ** 2)
        pidx = self.pair_index(z[iu[0]], z[iu[1]])
        c = self.coef[pidx]  # (P, K)
        return (basis * c * fc[..., None]).sum(dim=(-1, -2))


class CorrectedPotential(Potential):
    updates_uma_weights = False

    def __init__(self, base: Potential, correction: PairCorrection):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.correction = correction

    def energy(self, system, positions, cell=None):
        e0 = self.base.energy(system, positions, cell)
        pos, c, single = as_batch(positions.to(torch.float64), None if cell is None else cell.to(torch.float64))
        z = system.z_tensor(pos.device)
        de = self.correction(z, pos, c, periodic=system.is_periodic)
        de = de[0] if single else de
        return e0 + de.to(e0.device)
