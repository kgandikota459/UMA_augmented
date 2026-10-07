"""Model-based (label-free) mode characterization used by the Vib-v1 scoring rules.

torsion_fraction: for each normal mode, the largest squared overlap with a
rigid internal rotation of a molecular fragment about an acyclic bond
(mass-weighted, rigid-body motions projected out). A mode with fraction >= 0.5
is treated as a torsion / large-amplitude internal rotation (PROTOCOL 3.2),
independent of how the source's "approximate type" column words it.
"""

from __future__ import annotations

import numpy as np
from ase.data import covalent_radii


def bonds(positions: np.ndarray, numbers: np.ndarray, scale: float = 1.2) -> list[tuple[int, int]]:
    out = []
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if np.linalg.norm(positions[i] - positions[j]) < scale * (covalent_radii[numbers[i]] +
                                                                       covalent_radii[numbers[j]]):
                out.append((i, j))
    return out


def _side(n: int, edges: list[tuple[int, int]], start: int, cut: tuple[int, int]) -> set[int]:
    adj = {i: set() for i in range(n)}
    for a, b in edges:
        if {a, b} != set(cut):
            adj[a].add(b)
            adj[b].add(a)
    seen, stack = {start}, [start]
    while stack:
        for nb in adj[stack.pop()]:
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return seen


def _external(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    x = positions - (masses[:, None] * positions).sum(0) / masses.sum()
    w = np.sqrt(masses)
    cols = []
    for a in range(3):
        t = np.zeros_like(x)
        t[:, a] = 1.0
        cols.append((w[:, None] * t).ravel())
        r = np.cross(np.eye(3)[a], x)
        cols.append((w[:, None] * r).ravel())
    q, _ = np.linalg.qr(np.stack(cols, 1))
    return q


def torsion_fraction(positions, masses, numbers, modes: np.ndarray) -> np.ndarray:
    """(n_modes,) max over acyclic bonds of the squared overlap with a rigid fragment rotation."""
    x = np.asarray(positions, float)
    m = np.asarray(masses, float)
    z = np.asarray(numbers)
    n = len(z)
    edges = bonds(x, z)
    ext = _external(x, m)
    best = np.zeros(modes.shape[1])
    for a, b in edges:
        side_b = _side(n, edges, b, (a, b))
        if a in side_b:  # ring bond: no rigid internal rotation
            continue
        side_a = set(range(n)) - side_b
        frag, pivot = (side_b, b) if len(side_b) <= len(side_a) else (side_a, a)
        if len(side_a) < 2 or len(side_b) < 2:  # a terminal atom on the axis cannot rotate
            continue
        u = (x[b] - x[a]) / np.linalg.norm(x[b] - x[a])
        d = np.zeros_like(x)
        for i in frag:
            d[i] = np.cross(u, x[i] - x[pivot])
        t = (np.sqrt(m)[:, None] * d).ravel()
        t = t - ext @ (ext.T @ t)
        nt = np.linalg.norm(t)
        if nt < 1e-8:  # e.g. linear fragments
            continue
        t /= nt
        best = np.maximum(best, (t @ modes) ** 2)
    return best
