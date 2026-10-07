"""Molecular point groups: symmetry operations, closure, conjugacy classes and Schoenflies names.

Frozen with the data pipeline at Vib-v1 stage S2 (the held-out builder checks each page's point group
against the G2 geometry with ``point_group_name``). Mode assignment lives in umaexp/observables/symmetry.py; this module sits in the data package so that
importing it does not import the observables package.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

# expected group orders for NIST/Shimanouchi point-group strings
GROUP_ORDER = {"C1": 1, "Cs": 2, "Ci": 2, "C2": 2, "C3": 3, "S4": 4, "C2v": 4, "C2h": 4, "D2": 4, "C3v": 6,
               "C3h": 6, "D3": 6, "C4v": 8, "D2h": 8, "D2d": 8, "D3d": 12, "D3h": 12, "C6v": 12, "D4h": 16,
               "D5h": 20, "D6h": 24, "Td": 24, "Oh": 48}


@dataclass
class SymOp:
    R: np.ndarray  # (3, 3) orthogonal, acting on centred coordinates
    perm: np.ndarray  # atom i is carried onto site perm[i]
    kind: str = "?"  # E, C, S, sigma, i
    axis: np.ndarray | None = None
    angle: float = 0.0


@dataclass
class PointGroup:
    ops: list[SymOp]
    linear: bool
    principal_axis: np.ndarray | None
    principal_order: int
    unique_principal: bool
    d2_type: bool
    idx_inversion: int | None
    idx_sigma_h: int | None
    idx_principal: int | None
    idx_c2_perp: list[int] = field(default_factory=list)
    idx_sigma_v: list[int] = field(default_factory=list)
    idx_c2_all: list[int] = field(default_factory=list)
    idx_sigma_all: list[int] = field(default_factory=list)  # cubic groups: subscripts of A/T from sigma_d
    cubic: bool = False
    name: str = "?"

    @property
    def order(self) -> int:
        return len(self.ops)


def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def _refl(normal: np.ndarray) -> np.ndarray:
    n = normal / np.linalg.norm(normal)
    return np.eye(3) - 2.0 * np.outer(n, n)


def _image(R: np.ndarray, x: np.ndarray, types: np.ndarray, tol: float) -> np.ndarray | None:
    y = x @ R.T
    perm = np.empty(len(x), dtype=int)
    for i in range(len(x)):
        d = np.linalg.norm(x - y[i], axis=1)
        d[types != types[i]] = np.inf
        j = int(np.argmin(d))
        if d[j] > tol:
            return None
        perm[i] = j
    return perm if len(set(perm.tolist())) == len(x) else None


def _same(a: SymOp, R: np.ndarray, perm: np.ndarray, atol: float = 1e-3) -> bool:
    # distinct operations with the same atom mapping (E vs the molecular plane of a planar
    # molecule) differ by O(1) in R; the noise of a fitted R is far below 1e-3
    return np.array_equal(a.perm, perm) and np.allclose(a.R, R, atol=atol)


def _fit(x: np.ndarray, perm: np.ndarray, det: float) -> np.ndarray:
    """Best orthogonal R (fixed determinant sign) carrying x[i] onto x[perm[i]] (Kabsch)."""
    U, _, Vt = np.linalg.svd(x.T @ x[perm])
    d = np.sign(det) * np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def _classify(op: SymOp) -> None:
    R = op.R
    if np.allclose(R, np.eye(3), atol=1e-6):
        op.kind = "E"
        return
    if np.allclose(R, -np.eye(3), atol=1e-6):
        op.kind = "i"
        return
    det = np.linalg.det(R)
    w, v = np.linalg.eig(R)
    if det > 0:
        k = int(np.argmin(np.abs(w - 1.0)))
        op.kind, op.axis = "C", np.real(v[:, k])
        op.angle = float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    else:
        k = int(np.argmin(np.abs(w + 1.0)))
        op.axis = np.real(v[:, k])
        if np.allclose(R, R.T, atol=1e-6) and abs(np.trace(R) - 1.0) < 1e-6:
            op.kind = "sigma"
        else:
            op.kind = "S"
            op.angle = float(np.arccos(np.clip((np.trace(R) + 1.0) / 2.0, -1.0, 1.0)))
    op.axis = op.axis / np.linalg.norm(op.axis)


def find_point_group(positions: np.ndarray, masses: np.ndarray, numbers: np.ndarray,
                     tol: float = 0.01) -> PointGroup:
    """Symmetry operations (closed into a group) of a geometry, atoms told apart by Z and mass."""
    x = np.asarray(positions, float)
    m = np.asarray(masses, float)
    x = x - (m[:, None] * x).sum(0) / m.sum()
    types = np.unique(np.stack([np.asarray(numbers), np.round(m, 3)], 1), axis=0, return_inverse=True)[1].ravel()
    inertia = np.eye(3) * (m[:, None] * x * x).sum() - (m[:, None] * x).T @ x
    evals, evecs = np.linalg.eigh(inertia)
    linear = len(x) == 2 or evals[0] < 1e-5 * evals[-1]
    axes = [evecs[:, k] for k in range(3)]
    for i in range(len(x)):
        if np.linalg.norm(x[i]) > tol:
            axes.append(x[i])
        for j in range(i + 1, len(x)):
            if types[i] != types[j]:
                continue
            for v in (x[i] + x[j], x[i] - x[j], np.cross(x[i], x[j])):
                if np.linalg.norm(v) > tol:
                    axes.append(v)
    uniq: list[np.ndarray] = []
    for a in axes:
        a = a / np.linalg.norm(a)
        if all(abs(float(a @ b)) < 1 - 1e-6 for b in uniq):
            uniq.append(a)
    ops = [SymOp(np.eye(3), np.arange(len(x)))]

    def add(R):
        perm = _image(R, x, types, tol)
        if perm is not None:
            if not linear:  # rank-1 geometry: the fit cannot tell i from sigma_h, keep the analytic R
                R = _fit(x, perm, np.linalg.det(R))
            if not any(_same(o, R, perm) for o in ops):
                ops.append(SymOp(R, perm))

    add(-np.eye(3))
    if not linear:  # linear molecules: only E and i are needed (sigma/pi from dimension, g/u)
        for a in uniq:
            for n in (2, 3, 4, 5, 6):
                add(_rot(a, 2 * np.pi / n))
            add(_refl(a))
            for n in (3, 4, 6):
                add(_rot(a, 2 * np.pi / n) @ _refl(a))
        changed = True
        while changed:  # closure under composition
            changed = False
            for a, b in itertools.product(list(ops), list(ops)):
                perm = a.perm[b.perm]
                R = _fit(x, perm, np.linalg.det(a.R @ b.R))
                if not any(_same(o, R, perm) for o in ops):
                    ops.append(SymOp(R, perm))
                    changed = True
            if len(ops) > 120:
                raise RuntimeError("symmetry closure did not terminate (tolerance too loose?)")
    for o in ops:
        _classify(o)
    return _structure(ops, linear)


def _angle_count(ops, kind, angle):
    return sum(1 for o in ops if o.kind == kind and abs(o.angle - angle) < 1e-3)


def point_group_name(ops: list[SymOp], linear: bool) -> str:
    """Schoenflies symbol from the operation content (orders and axis counts), e.g. 'C2v', 'D2d', 'Td'."""
    if linear:
        return "D∞h" if any(o.kind == "i" for o in ops) else "C∞v"
    n = len(ops)
    c2 = _angle_count(ops, "C", np.pi)
    c3 = _angle_count(ops, "C", 2 * np.pi / 3) // 2
    c4 = _angle_count(ops, "C", np.pi / 2) // 2
    c5 = _angle_count(ops, "C", 2 * np.pi / 5) // 2
    c6 = _angle_count(ops, "C", np.pi / 3) // 2
    s4 = _angle_count(ops, "S", np.pi / 2) // 2
    sig = sum(o.kind == "sigma" for o in ops)
    inv = any(o.kind == "i" for o in ops)
    table = [
        (1, lambda: True, "C1"), (2, lambda: sig == 1, "Cs"), (2, lambda: inv, "Ci"), (2, lambda: c2 == 1, "C2"),
        (3, lambda: c3 == 1, "C3"),
        (4, lambda: c2 == 1 and sig == 2, "C2v"), (4, lambda: c2 == 1 and inv, "C2h"),
        (4, lambda: c2 == 3, "D2"), (4, lambda: s4 == 1, "S4"), (4, lambda: c4 == 1, "C4"),
        (6, lambda: c3 == 1 and sig == 3, "C3v"), (6, lambda: c3 == 1 and c2 == 3, "D3"),
        (6, lambda: c3 == 1 and sig == 1, "C3h"), (6, lambda: c3 == 1 and inv, "S6"), (6, lambda: c6 == 1, "C6"),
        (8, lambda: c2 == 3 and inv, "D2h"), (8, lambda: s4 == 1 and c2 == 3 and sig == 2, "D2d"),
        (8, lambda: c4 == 1 and sig == 4, "C4v"), (8, lambda: c4 == 1 and inv, "C4h"), (8, lambda: c4 == 1, "D4"),
        (10, lambda: c5 == 1 and sig == 5, "C5v"), (10, lambda: c5 == 1, "D5"),
        (12, lambda: c3 == 1 and inv and c2 == 3, "D3d"), (12, lambda: c3 == 1 and sig == 4, "D3h"),
        (12, lambda: c6 == 1 and sig == 6, "C6v"), (12, lambda: c6 == 1 and inv, "C6h"), (12, lambda: c3 == 4, "T"),
        (12, lambda: c6 == 1, "D6"),
        (16, lambda: c4 == 1 and inv, "D4h"), (16, lambda: c4 == 1, "D4d"),
        (20, lambda: c5 == 1 and sig == 6, "D5h"), (20, lambda: c5 == 1, "D5d"),
        (24, lambda: c3 == 4 and sig == 6 and not inv, "Td"), (24, lambda: c3 == 4 and inv, "Th"),
        (24, lambda: c6 == 1 and inv, "D6h"), (24, lambda: c4 == 1 or c6 == 0, "D6d"),
        (48, lambda: inv, "Oh"),
    ]
    for order, cond, name in table:
        if order == n and cond():
            return name
    return f"unknown({n})"


def _structure(ops: list[SymOp], linear: bool) -> PointGroup:
    rots = [(k, o) for k, o in enumerate(ops) if o.kind == "C"]
    order_of_axis: list[tuple[np.ndarray, int]] = []
    for _, o in rots:
        n = int(round(2 * np.pi / o.angle))
        for t, (ax, nn) in enumerate(order_of_axis):
            if abs(float(ax @ o.axis)) > 1 - 1e-6:
                order_of_axis[t] = (ax, max(nn, n))
                break
        else:
            order_of_axis.append((o.axis, n))
    n_max = max([n for _, n in order_of_axis], default=1)
    principal = [ax for ax, n in order_of_axis if n == n_max]
    s4 = [(k, o) for k, o in enumerate(ops) if o.kind == "S" and abs(o.angle - np.pi / 2) < 1e-3]
    s4_axes = []
    for _, o in s4:
        if all(abs(float(o.axis @ a)) < 1 - 1e-6 for a in s4_axes):
            s4_axes.append(o.axis)
    name = point_group_name(ops, linear)
    sig = [k for k, o in enumerate(ops) if o.kind == "sigma"]
    # D2d / S4: the improper S4 axis is the principal axis (critic report #3 P6)
    use_s4 = name in ("D2d", "S4") and len(s4_axes) == 1
    unique = use_s4 or (n_max >= 2 and len(principal) == 1)
    d2 = (not use_s4) and n_max == 2 and len(principal) == 3
    z = s4_axes[0] if use_s4 else (principal[0] if unique else None)
    pg = PointGroup(ops, linear, z, 4 if use_s4 else n_max, unique, d2,
                    idx_inversion=next((k for k, o in enumerate(ops) if o.kind == "i"), None),
                    idx_sigma_h=None, idx_principal=None, name=name, cubic=name in ("Td", "T", "Th", "Oh"))
    if unique:
        if use_s4:
            pg.idx_principal = next(k for k, o in s4 if abs(float(o.axis @ z)) > 1 - 1e-6)
        else:
            pg.idx_principal = next(k for k, o in rots if abs(float(o.axis @ z)) > 1 - 1e-6
                                    and abs(o.angle - 2 * np.pi / n_max) < 1e-6)
        pg.idx_sigma_h = next((k for k in sig if abs(float(ops[k].axis @ z)) > 1 - 1e-6), None)
        pg.idx_sigma_v = [k for k in sig if abs(float(ops[k].axis @ z)) < 1e-6]
        pg.idx_c2_perp = [k for k, o in rots if abs(o.angle - np.pi) < 1e-6 and abs(float(o.axis @ z)) < 1e-6]
    elif n_max == 1 and len(sig) == 1:  # Cs: the single plane carries the prime convention
        pg.idx_sigma_h = sig[0]
    if d2:
        pg.idx_c2_all = [k for k, o in rots if abs(o.angle - np.pi) < 1e-6]
    if pg.cubic:
        pg.idx_sigma_all = sig
    return pg


def operation_matrix(op: SymOp) -> np.ndarray:
    n = len(op.perm)
    M = np.zeros((3 * n, 3 * n))
    for i, j in enumerate(op.perm):
        M[3 * j:3 * j + 3, 3 * i:3 * i + 3] = op.R
    return M


def _classes(pg: PointGroup) -> list[list[int]]:
    ops = pg.ops
    inv = [(o.R.T, np.argsort(o.perm)) for o in ops]
    seen, classes = set(), []
    for g, og in enumerate(ops):
        if g in seen:
            continue
        cls = set()
        for h, oh in enumerate(ops):
            R = oh.R @ og.R @ inv[h][0]
            perm = oh.perm[og.perm[inv[h][1]]]
            k = next(k for k, o in enumerate(ops) if _same(o, R, perm, atol=1e-2))
            cls.add(k)
        seen |= cls
        classes.append(sorted(cls))
    return classes


