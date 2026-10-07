"""Toy potentials and cache builders with closed-form answers, shared by tests."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from umaexp.potential.base import Potential, as_batch
from umaexp.sampling import SampleCache, SamplingProtocol, ThermoState
from umaexp.systems import System


class Zero(Potential):
    """U = 0 (kept on the graph so forces/Hessians are well defined)."""

    def energy(self, system, positions, cell=None):
        p = positions if positions.dim() == 3 else positions[None]
        e = 0.0 * p.to(torch.float64).sum(dim=(1, 2))
        return e if positions.dim() == 3 else e[0]


class SizeDependent(Potential):
    """U = 1/2 k sum |r|^2 + c N^2, N = number of atoms in the system."""

    def __init__(self, k: float = 1.0, c: float = 0.0):
        super().__init__()
        self.log_k = nn.Parameter(torch.tensor(math.log(k), dtype=torch.float64))
        self.c = nn.Parameter(torch.tensor(c, dtype=torch.float64))

    def energy(self, system, positions, cell=None):
        pos, _, single = as_batch(positions.to(torch.float64), None)
        n = pos.shape[1]
        e = 0.5 * self.log_k.exp() * (pos**2).sum(dim=(1, 2)) + self.c * n * n
        return e[0] if single else e


class PairMorse(Potential):
    """sum_{i<j} [D (1 - exp(-a (r - re)))^2 - D]; separated atoms have U = 0."""

    def __init__(self, D: float, a: float, re: float):
        super().__init__()
        self.D = nn.Parameter(torch.tensor(D, dtype=torch.float64))
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float64))
        self.re = nn.Parameter(torch.tensor(re, dtype=torch.float64))

    def energy(self, system, positions, cell=None):
        pos, _, single = as_batch(positions.to(torch.float64), None)
        n = pos.shape[1]
        iu = torch.triu_indices(n, n, offset=1)
        r = (pos[:, iu[0]] - pos[:, iu[1]]).norm(dim=-1)
        u = self.D * (1 - torch.exp(-self.a * (r - self.re))) ** 2 - self.D
        e = u.sum(-1) + 0.0 * pos.sum(dim=(1, 2))
        return e[0] if single else e


class InverseR(Potential):
    """sum_{i<j} -A / r: attractive all the way in (fails a repulsion check)."""

    def __init__(self, A: float = 1.0):
        super().__init__()
        self.A = nn.Parameter(torch.tensor(A, dtype=torch.float64))

    def energy(self, system, positions, cell=None):
        pos, _, single = as_batch(positions.to(torch.float64), None)
        n = pos.shape[1]
        iu = torch.triu_indices(n, n, offset=1)
        r = (pos[:, iu[0]] - pos[:, iu[1]]).norm(dim=-1)
        e = (-self.A / r).sum(-1)
        return e[0] if single else e


def toy_system(zs, positions, system_id="s", **kw) -> System:
    from umaexp.systems import isotopologue_masses

    return System(system_id, tuple(zs), isotopologue_masses(tuple(zs)), task="toy", spin=0,
                  positions=np.asarray(positions, dtype=float), **kw)


def cache_from_frames(system: System, positions, cells=None, u=None, state: ThermoState | None = None,
                      fingerprint: str = "test") -> SampleCache:
    """Cache of explicit frames; u defaults to zeros (use with Zero-like samplers)."""
    positions = np.asarray(positions, dtype=float)
    u = np.zeros(positions.shape[0]) if u is None else np.asarray(u, dtype=float)
    return SampleCache(system.system_id, system.atom_order_hash(), state or ThermoState(300.0),
                       SamplingProtocol(), fingerprint, positions,
                       None if cells is None else np.asarray(cells, dtype=float), u)
