r"""Structural ensemble observables: RDFs, distances, angles/dihedrals, populations.

With reweighting, cached frames are fixed, so per-frame quantities may be
hard histograms/indicators: the gradient flows through the weights a_i(w),
not through the frame values. (Smoothing is only needed when positions
themselves are differentiated, e.g. unrolled dynamics.)

Populations require adequate basin coverage. An experiment-derived
structural model (crystal/NMR/cryo-EM coordinates) is not a population
label; only measured population-sensitive quantities should enter a loss.
"""

from __future__ import annotations

import math

import torch

from ..potential.base import Potential
from ..reweighting.difftre import ReweightedEnsemble
from ..systems.system import System
from .base import Prediction
from .ensemble import ensemble_prediction


def _mic(dr, cells):
    inv = torch.linalg.inv(cells)
    frac = torch.einsum("bpi,bij->bpj", dr, inv)
    return dr - torch.einsum("bpi,bij->bpj", torch.round(frac), cells)


def pair_vectors(pos, cells, i_idx, j_idx, periodic: bool):
    dr = pos[:, i_idx] - pos[:, j_idx]
    return _mic(dr, cells) if periodic else dr


def dihedral(pos, i, j, k, l):
    """Dihedral angle (radians) for (b, N, 3) positions; IUPAC sign convention."""
    b0 = pos[:, i] - pos[:, j]
    b1 = pos[:, k] - pos[:, j]
    b2 = pos[:, l] - pos[:, k]
    b1n = b1 / b1.norm(dim=-1, keepdim=True)
    v = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n
    x = (v * w).sum(-1)
    y = (torch.linalg.cross(b1n, v, dim=-1) * w).sum(-1)
    return torch.atan2(y, x)


class RadialDistribution:
    """g_ab(r) with per-frame volume normalization (NPT-safe)."""

    def __init__(self, system: System, species_a: int, species_b: int, r_max: float, n_bins: int = 100,
                 intermolecular_only: bool = False):
        if not system.is_periodic:
            raise ValueError("RDF normalization here assumes a periodic box")
        z = torch.as_tensor(system.atomic_numbers)
        a = torch.nonzero(z == species_a).flatten()
        b = torch.nonzero(z == species_b).flatten()
        ii, jj = torch.meshgrid(a, b, indexing="ij")
        ii, jj = ii.flatten(), jj.flatten()
        keep = ii != jj
        if intermolecular_only and system.molecule_index is not None:
            mol = torch.as_tensor(system.molecule_index)
            keep &= mol[ii] != mol[jj]
        self.i, self.j = ii[keep], jj[keep]
        self.na, self.nb = len(a), len(b)
        self.same = species_a == species_b
        self.edges = torch.linspace(0.0, r_max, n_bins + 1, dtype=torch.float64)
        self.r = 0.5 * (self.edges[1:] + self.edges[:-1])
        self.shell = 4.0 / 3.0 * math.pi * (self.edges[1:] ** 3 - self.edges[:-1] ** 3)

    def frames(self, pos, cells):
        d = pair_vectors(pos, cells, self.i, self.j, True).norm(dim=-1)
        counts = torch.stack([torch.histc(d[b], bins=len(self.r), min=0.0, max=float(self.edges[-1]))
                              for b in range(d.shape[0])]).to(torch.float64)
        vol = torch.linalg.det(cells).abs()[:, None]
        n_pairs = self.i.numel()
        return counts * vol / (n_pairs * self.shell[None, :])

    def predict(self, potential: Potential, ens: ReweightedEnsemble, mode="graph") -> Prediction:
        return ensemble_prediction(ens, potential, self.frames, None, mode, meta={"r": self.r})


class DistanceAverage:
    """<r^n>^(1/n) for an atom pair (n=-6: isotropic NOE effective distance)."""

    def __init__(self, i: int, j: int, exponent: float = 1.0, periodic: bool = False):
        self.i, self.j, self.n, self.periodic = i, j, exponent, periodic

    def predict(self, potential, ens, mode="graph") -> Prediction:
        i, j, n, per = self.i, self.j, self.n, self.periodic

        def frames(p, c):
            d = pair_vectors(p, c, torch.tensor([i]), torch.tensor([j]), per).norm(dim=-1)
            return d ** n

        return ensemble_prediction(ens, potential, frames, lambda m: m ** (1.0 / n), mode)


class StatePopulation:
    """Population of states defined by dihedral windows (degrees, half-open)."""

    def __init__(self, dihedral_atoms: tuple[int, int, int, int], windows: dict[str, list[tuple[float, float]]]):
        self.atoms = dihedral_atoms
        self.names = list(windows)
        self.windows = windows

    def frames(self, p, c):
        phi = torch.rad2deg(dihedral(p, *self.atoms))
        cols = []
        for name in self.names:
            ind = torch.zeros_like(phi)
            for lo, hi in self.windows[name]:
                ind = ind + ((phi >= lo) & (phi < hi)).to(phi.dtype)
            cols.append(ind.clamp(max=1.0))
        return torch.stack(cols, -1)

    def predict(self, potential, ens, mode="graph") -> Prediction:
        return ensemble_prediction(ens, potential, self.frames, None, mode, meta={"states": self.names})
