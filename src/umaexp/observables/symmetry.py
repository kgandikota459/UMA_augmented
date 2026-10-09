"""Point-group operations of a molecule and symmetry-constrained mode assignment.

Purpose (experiments/vib_v1/PROTOCOL.md 3.3): assign computed normal modes to
tabulated fundamentals WITHOUT using the measured values. Matching is done per
symmetry species; inside a species, computed modes in decreasing frequency are
paired with the source's mode numbers in increasing order (the spectroscopic
numbering convention). An ordering error of the model across species is then
scored as an error instead of being hidden by nearest-value matching.

Only convention-independent features of Mulliken labels are used:
  * dimension: A/B/Sigma -> 1, E/Pi -> 2, T/F -> 3;
  * g/u: sign of the character under inversion;
  * prime/double prime: sign under sigma_h (for Cs, its only plane);
  * A vs B: sign under the unique principal rotation (D2-type groups: A is
    symmetric under all three C2);
  * subscript of A: sign under C2' perpendicular to the principal axis, or under
    sigma_v when there is no C2' (Mulliken's rule);
  * subscript of E: character 2cos(2 pi k / n) under the principal C_n (n >= 5).
B1/B2/B3 and T1/T2 subscripts depend on axis conventions and are never used.
Species that remain indistinguishable after counts are merged into one class,
matched by frequency order of the measured values (the one remaining value-based
step; reported per molecule as `merged_classes`).
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

import numpy as np

from ..data.pointgroup import (  # noqa: F401 - re-exported
    GROUP_ORDER,
    PointGroup,
    SymOp,
    _classes,
    find_point_group,
    operation_matrix,
    point_group_name,
)


@dataclass
class ModeGroup:
    indices: list[int]  # indices into the input mode list (after separation, see vectors)
    vectors: np.ndarray  # (3N, d) orthonormal, irrep-pure
    eigenvalue: float  # mean eigenvalue (input units)
    characters: np.ndarray  # (|G|,)
    irreducible: bool


def symmetry_mode_groups(pg: PointGroup, eigenvalues: np.ndarray, modes: np.ndarray, wavenumbers: np.ndarray,
                         cluster_cm: float = 1.0, seed: int = 0) -> list[ModeGroup]:
    """Split modes into irrep-pure degenerate groups.

    Clusters of modes within `cluster_cm` are re-diagonalized with a random
    central (class-sum) operator, which is a scalar on each irreducible
    representation, so accidental near-degeneracies between different irreps
    are separated and true degeneracies are kept together.
    """
    Ms = [operation_matrix(o) for o in pg.ops]
    rng = np.random.default_rng(seed)
    K = np.zeros_like(Ms[0])
    for cls in _classes(pg):
        r = rng.uniform(1.0, 2.0)
        for g in cls:
            K += r * 0.5 * (Ms[g] + Ms[g].T)
    order = np.argsort(wavenumbers)
    clusters: list[list[int]] = []
    for i in order:
        if clusters and abs(wavenumbers[i] - wavenumbers[clusters[-1][-1]]) <= cluster_cm:
            clusters[-1].append(int(i))
        else:
            clusters.append([int(i)])
    groups: list[ModeGroup] = []
    for cl in clusters:
        V = modes[:, cl]
        lam = eigenvalues[cl]
        A = V.T @ K @ V
        kap, U = np.linalg.eigh(0.5 * (A + A.T))
        W = V @ U
        lam_w = (U**2).T @ lam  # Rayleigh quotients in the rotated basis
        scale = max(1.0, float(np.abs(kap).max()))
        sub: list[list[int]] = []
        for t in np.argsort(kap):
            if sub and abs(kap[t] - kap[sub[-1][-1]]) < 1e-6 * scale:
                sub[-1].append(int(t))
            else:
                sub.append([int(t)])
        for s in sub:
            Ws = W[:, s]
            chars = np.array([np.trace(Ws.T @ M @ Ws) for M in Ms])
            norm = float((chars**2).sum() / len(Ms))
            if pg.linear:  # only E and i are kept for linear molecules: sigma/pi come from the clustering
                groups.append(ModeGroup([cl[k] for k in s], Ws, float(np.mean(lam_w[s])), chars, True))
                continue
            if norm > 1.5 and not np.isclose(norm, 2.0, atol=0.2):
                # several copies of one irrep inside a cluster: split by the eigenvalues
                d = int(round(len(s) / round(np.sqrt(norm))))
                srt = np.argsort(lam_w[s])
                for c in range(0, len(s), d):
                    ss = [s[k] for k in srt[c:c + d]]
                    Wc = W[:, ss]
                    ch = np.array([np.trace(Wc.T @ M @ Wc) for M in Ms])
                    groups.append(ModeGroup([cl[k] for k in ss], Wc, float(np.mean(lam_w[ss])), ch,
                                            bool(abs((ch**2).sum() / len(Ms) - 1.0) < 0.2
                                                 or abs((ch**2).sum() / len(Ms) - 2.0) < 0.2)))
                continue
            groups.append(ModeGroup([cl[k] for k in s], Ws, float(np.mean(lam_w[s])), chars,
                                    bool(abs(norm - 1.0) < 0.2 or abs(norm - 2.0) < 0.2)))
    return groups


# ---------------------------------------------------------------------------
# Mulliken labels
# ---------------------------------------------------------------------------

_LETTER = {"a": 1, "b": 1, "sigma": 1, "e": 2, "pi": 2, "delta": 2, "t": 3, "f": 3}


def parse_species(label: str) -> dict:
    s = label.strip().lower().replace("σ", "sigma").replace("π", "pi").replace("δ", "delta")
    s = s.replace("″", "''").replace("′", "'").replace('"', "''")
    primes = s.count("'")
    s = s.replace("'", "")
    m = re.match(r"^(sigma|pi|delta|a|b|e|t|f)(\d*)([gu]?)([+-]?)$", s)
    if not m:
        raise ValueError(f"unrecognized symmetry species {label!r}")
    letter, sub, par, pm = m.groups()
    letter = "t" if letter == "f" else letter
    return {"letter": letter, "dim": _LETTER[letter], "sub": int(sub) if sub else None,
            "parity": par or None, "primes": primes, "sign": pm or None}


def compatible(sp: dict, g: ModeGroup, pg: PointGroup, tol: float = 0.25) -> bool:
    d = len(g.indices)
    if sp["dim"] != d:
        return False
    ch = g.characters / d
    if sp["parity"]:
        if pg.idx_inversion is None or abs(ch[pg.idx_inversion] - (1 if sp["parity"] == "g" else -1)) > tol:
            return False
    if sp["primes"]:
        if pg.idx_sigma_h is None or abs(ch[pg.idx_sigma_h] - (1 if sp["primes"] == 1 else -1)) > tol:
            return False
    if pg.linear:
        return True
    if d == 1 and sp["letter"] in ("a", "b"):
        if pg.unique_principal and pg.principal_order >= 2:
            want = 1 if sp["letter"] == "a" else -1
            if abs(ch[pg.idx_principal] - want) > tol:
                return False
        elif pg.d2_type:
            all_sym = all(abs(ch[k] - 1) < tol for k in pg.idx_c2_all)
            if all_sym != (sp["letter"] == "a"):
                return False
        if sp["letter"] == "a" and sp["sub"] in (1, 2) and pg.unique_principal:
            ref = pg.idx_c2_perp or pg.idx_sigma_v
            if ref:
                want = 1 if sp["sub"] == 1 else -1
                if any(abs(ch[k] - want) > tol for k in ref):
                    return False
    if pg.cubic and pg.idx_sigma_all and sp["sub"] in (1, 2) and sp["letter"] in ("a", "t"):
        # Td: A1/T2 symmetric under sigma_d (chi = +1), A2/T1 antisymmetric (chi = -1)
        want = 1 if (sp["letter"], sp["sub"]) in (("a", 1), ("t", 2)) else -1
        if any(np.sign(ch[k] * d) != want for k in pg.idx_sigma_all):
            return False
    if d == 2 and sp["letter"] == "e" and sp["sub"] and pg.unique_principal and pg.principal_order >= 5:
        if abs(ch[pg.idx_principal] - np.cos(2 * np.pi * sp["sub"] / pg.principal_order)) > tol:
            return False
    return True


# ---------------------------------------------------------------------------
# assignment
# ---------------------------------------------------------------------------


@dataclass
class SymmetryAssignment:
    matches: list[tuple[int, list[int]]]  # (exp index, input mode indices)
    excluded_exp: list[int]  # experimental modes in classes whose counts disagree
    merged_classes: list[list[str]]  # species that could only be matched as a merged class
    point_group_order: int
    point_group_ok: bool
    notes: list[str] = field(default_factory=list)


def _conventions(pg: PointGroup) -> list:
    """Axis conventions defining B subscripts (Mulliken), each as a predicate b_ok(species, chars/d)."""
    def b1b2(primary, secondary):
        def ok(sp, ch, tol=0.25):
            want = 1 if sp["sub"] == 1 else -1
            return (all(abs(ch[k] - want) < tol for k in primary) and
                    all(abs(ch[k] + want) < tol for k in secondary))
        return ok

    if pg.d2_type and len(pg.idx_c2_all) == 3:
        out = []
        for z, y, x in itertools.permutations(pg.idx_c2_all):
            def ok(sp, ch, z=z, y=y, x=x, tol=0.25):
                axis = {1: z, 2: y, 3: x}.get(sp["sub"])
                if axis is None:
                    return True
                return all(abs(ch[k] - (1 if k == axis else -1)) < tol for k in (z, y, x))
            out.append(ok)
        return out
    if pg.unique_principal:
        if pg.name == "D2d" and pg.idx_c2_perp and pg.idx_sigma_v:
            # convention-free in D2d: B1 is symmetric under C2', B2 under sigma_d (red-team report #4)
            return [b1b2(pg.idx_c2_perp, pg.idx_sigma_v)]
        perp = pg.idx_c2_perp or pg.idx_sigma_v
        classes = [c for c in _classes(pg) if set(c) <= set(perp)]
        if len(classes) == 2:
            return [b1b2(classes[0], classes[1]), b1b2(classes[1], classes[0])]
    return [lambda sp, ch: True]


def _solve(species_all, irr_keys, compat, count, size):
    """Per component of mutually compatible species: every count-consistent species->irrep bijection."""
    parent = {s: s for s in species_all}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for r in irr_keys:
        ss = [s for s in species_all if compat[(s, r)]]
        for a, b in zip(ss, ss[1:]):
            parent[find(a)] = find(b)
    comps: dict[str, list[str]] = {}
    for s_ in species_all:
        comps.setdefault(find(s_), []).append(s_)
    out = []
    for species in comps.values():
        rs = [r for r in irr_keys if any(compat[(s_, r)] for s_ in species)]
        valid = []
        if len(rs) == len(species):
            for perm in itertools.permutations(rs):
                if all(compat[(s_, r)] and count[s_] == size[r] for s_, r in zip(species, perm)):
                    valid.append(dict(zip(species, perm)))
        out.append((species, rs, valid))
    return out


def assign_by_symmetry(pg: PointGroup, groups: list[ModeGroup], exp_species: list[str],
                       exp_numbers: list[int], exp_values: list[float],
                       expected_point_group: str | None = None) -> SymmetryAssignment:
    """Assign computed mode groups to experimental modes using symmetry species only.

    exp_* describe the FULL experimental table (usable and flagged modes), so
    class counts can be compared; scoring decides later which modes are used.
    B subscripts are resolved by enumerating the source's possible axis
    conventions and keeping those consistent with every species count at once
    (critic report #2 M1); species stay merged only when that is ambiguous.
    """
    notes = []
    pg_ok = True
    if expected_point_group and not pg.linear and pg.name != expected_point_group:
        pg_ok = False
        notes.append(f"point group {pg.name} at the minimum != {expected_point_group} (source)")
    if any(not g.irreducible for g in groups):
        notes.append("non-irreducible mode group (broken symmetry)")
        pg_ok = False
    if not pg_ok:
        return SymmetryAssignment([], list(range(len(exp_species))), [], pg.order, False, notes)
    parsed = {s_: parse_species(s_) for s_ in set(exp_species)}
    irreps: dict[tuple, list[int]] = {}
    for gi, g in enumerate(groups):  # computed irreps = groups with identical (rounded) character vectors
        irreps.setdefault(tuple(np.round(g.characters, 1)), []).append(gi)
    irr_keys = list(irreps)
    size = {r: len(irreps[r]) for r in irr_keys}
    count = {s_: exp_species.count(s_) for s_ in parsed}
    base = {(s_, r): compatible(parsed[s_], groups[irreps[r][0]], pg) for s_ in parsed for r in irr_keys}

    def with_convention(ok):
        def chars(r):
            g = groups[irreps[r][0]]
            return g.characters / len(g.indices)
        return {(s_, r): base[(s_, r)] and (parsed[s_]["letter"] != "b" or parsed[s_]["sub"] is None
                                            or len(groups[irreps[r][0]].indices) != 1 or ok(parsed[s_], chars(r)))
                for s_ in parsed for r in irr_keys}

    solutions = []  # per consistent convention: list of (species, rs, valid)
    for ok in _conventions(pg):
        sol = _solve(list(parsed), irr_keys, with_convention(ok), count, size)
        if all(valid for _, _, valid in sol):
            solutions.append(sol)
    n_conv = len(solutions)
    if not solutions:  # no convention fits every count: fall back to convention-free constraints
        notes.append("no axis convention consistent with all species counts; B subscripts ignored")
        solutions = [_solve(list(parsed), irr_keys, base, count, size)]
    elif n_conv > 1:
        notes.append(f"{n_conv} axis conventions consistent with the species counts")
    possible: dict[str, set] = {s_: set() for s_ in parsed}
    excluded_species = set()
    for sol in solutions:
        for species, rs, valid in sol:
            if not valid:
                excluded_species |= set(species)
                n_comp = sum(size[r] for r in rs)
                notes.append(f"class {sorted(species)}: {sum(count[x] for x in species)} tabulated vs {n_comp} "
                             f"computed; excluded")
            for v in valid:
                for s_, r in v.items():
                    possible[s_].add(r)
    matches, merged = [], []
    excluded = [i for i, s_ in enumerate(exp_species) if s_ in excluded_species]
    fixed = {s_: next(iter(rs)) for s_, rs in possible.items() if len(rs) == 1 and s_ not in excluded_species}
    for s_, r in fixed.items():  # one species <-> one irrep: number order vs frequency order (value-free)
        ei = sorted([i for i, x in enumerate(exp_species) if x == s_], key=lambda i: exp_numbers[i])
        vals = [exp_values[i] for i in ei]
        if any(not np.isfinite(v) for v in vals) or any(b > a for a, b in zip(vals, vals[1:])):
            # DEVIATIONS.md DV-01: the source does not number this species in descending frequency
            # (or a value is missing), so number order cannot be matched to frequency order: fail closed
            excluded += ei
            notes.append(f"species {s_}: tabulated values not in descending order by mode number; excluded")
            continue
        gs = sorted(irreps[r], key=lambda gi: -groups[gi].eigenvalue)
        matches += [(e, groups[gi].indices) for e, gi in zip(ei, gs)]
    rest = [s_ for s_ in possible if s_ not in fixed and s_ not in excluded_species]
    seen: set = set()
    for s_ in rest:  # ambiguous species: merge those sharing possible irreps; order by measured value
        if s_ in seen:
            continue
        cls, irr, frontier = {s_}, set(possible[s_]), [s_]
        while frontier:
            frontier = [x for x in rest if x not in cls and possible[x] & irr]
            for x in frontier:
                cls.add(x)
                irr |= possible[x]
        seen |= cls
        merged.append(sorted(cls))
        members = [i for i, x in enumerate(exp_species) if x in cls]
        if any(not np.isfinite(exp_values[i]) for i in members):  # unrankable value: fail closed
            excluded += members
            notes.append(f"merged class {sorted(cls)} has a non-numeric tabulated value; excluded")
            continue
        ei = sorted(members, key=lambda i: -exp_values[i])
        gs = sorted([gi for r in irr for gi in irreps[r]], key=lambda gi: -groups[gi].eigenvalue)
        if len(ei) != len(gs):
            excluded += ei
            notes.append(f"merged class {sorted(cls)} does not add up; excluded")
            continue
        matches += [(e, groups[gi].indices) for e, gi in zip(ei, gs)]
    used = {r for rs in possible.values() for r in rs}
    orphan = [r for r in irr_keys if r not in used]
    if orphan:
        notes.append(f"{sum(size[r] for r in orphan)} computed modes in irreps absent from the table")
    return SymmetryAssignment(sorted(matches), sorted(set(excluded)), merged, pg.order, True, notes)
