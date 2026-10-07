"""Leakage-aware splits.

Observations that share ANY group key (molecule, scaffold, provenance
compilation, publication, system cluster) are placed in the same split:
connected components of the "shares a key" graph (union-find) are the units
that get assigned. This prevents, e.g., an isotopologue or a second
measurement of the same molecule from the same paper leaking into test.

Assignment is deterministic given the seed: components are ordered by a
hash of (seed, component key) and filled greedily toward target fractions
counted in observations.

Compilation keys: a publication/system key shared by more than
`max_group_fraction` of the records (e.g. a compilation DOI cited by
hundreds of FreeSolv entries) would merge everything into one component. It
identifies a compilation, not one study's correlated measurements, so it is
excluded from linking and reported; use it for a separate
leave-provenance-out evaluation instead.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict

from .schema import Observation

GROUP_FIELDS = ("molecule_group", "scaffold_group", "publication_group", "system_group")


class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def bemis_murcko_scaffold(smiles: str | None) -> str | None:
    if not smiles:
        return None
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return scaf or "acyclic"


def oversized_keys(obs: list[Observation], fields=GROUP_FIELDS, max_group_fraction: float = 0.2) -> set:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for o in obs:
        for f in fields:
            key = getattr(o.groups, f)
            if key is not None and f in ("publication_group", "system_group"):
                counts[(f, key)] += 1
    limit = max(2, int(max_group_fraction * len(obs)))
    return {k for k, c in counts.items() if c > limit}


def group_components(obs: list[Observation], fields=GROUP_FIELDS, skip: set | None = None) -> list[list[int]]:
    skip = skip or set()
    uf = _UnionFind(len(obs))
    first: dict[tuple[str, str], int] = {}
    for i, o in enumerate(obs):
        for f in fields:
            key = getattr(o.groups, f)
            if key is None or (f == "scaffold_group" and key == "acyclic") or (f, key) in skip:
                continue
            k = (f, key)
            if k in first:
                uf.union(first[k], i)
            else:
                first[k] = i
    comps = defaultdict(list)
    for i in range(len(obs)):
        comps[uf.find(i)].append(i)
    return list(comps.values())


def group_split(obs: list[Observation], fractions=(0.7, 0.15, 0.15), seed: int = 0,
                fields=GROUP_FIELDS, max_group_fraction: float = 0.2) -> dict[str, list[str]]:
    names = ("train", "val", "test")
    skip = oversized_keys(obs, fields, max_group_fraction)
    comps = group_components(obs, fields, skip)

    def key(c):
        ids = sorted(obs[i].observation_id for i in c)
        return hashlib.sha256(f"{seed}:{ids[0]}".encode()).hexdigest()

    comps.sort(key=key)
    total = len(obs)
    target = [f * total for f in fractions]
    out = {n: [] for n in names}
    counts = [0, 0, 0]
    for c in comps:
        # put the component where the deficit relative to target is largest
        k = max(range(3), key=lambda s: target[s] - counts[s])
        out[names[k]].extend(obs[i].observation_id for i in c)
        counts[k] += len(c)
    out["_excluded_compilation_keys"] = sorted(f"{f}={k}" for f, k in skip)
    return out


def assert_no_leakage(split: dict[str, list[str]], obs: list[Observation], fields=GROUP_FIELDS) -> None:
    skipped = set(split.get("_excluded_compilation_keys", []))
    where = {}
    for name, ids in split.items():
        if name.startswith("_"):
            continue
        for i in ids:
            where[i] = name
    seen: dict[tuple[str, str], str] = {}
    for o in obs:
        for f in fields:
            k = getattr(o.groups, f)
            if k is None or (f == "scaffold_group" and k == "acyclic") or f"{f}={k}" in skipped:
                continue
            s = where[o.observation_id]
            if seen.setdefault((f, k), s) != s:
                raise AssertionError(f"group {f}={k} appears in {seen[(f, k)]} and {s}")
