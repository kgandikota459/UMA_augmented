"""Experiment / reference registries stored as Parquet with versioned manifests.

Scalar fields become columns (queryable with pandas/pyarrow); nested models
are stored as JSON strings and re-validated on load, so a registry read back
is exactly the typed records that were written.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd

from .schema import Observation, ValueKind

_NESTED = ["system_ids", "stoichiometry", "value", "grid", "labels", "uncertainty", "censoring", "conditions",
           "forward_model", "nuisance", "compatible_uma_tasks", "provenance", "groups", "quality_flags"]


class ExperimentRegistry:
    def __init__(self, observations: Iterable[Observation] = ()):
        self._obs: dict[str, Observation] = {}
        for o in observations:
            self.add(o)

    def add(self, obs: Observation) -> None:
        if obs.observation_id in self._obs:
            raise KeyError(f"duplicate observation_id {obs.observation_id}")
        self._obs[obs.observation_id] = obs

    def __len__(self):
        return len(self._obs)

    def __iter__(self):
        return iter(self._obs.values())

    def get(self, oid: str) -> Observation:
        return self._obs[oid]

    def select(self, family=None, experimental_only: bool = True, system_id: str | None = None,
               exclude_flags: tuple[str, ...] = ()) -> list[Observation]:
        out = []
        for o in self._obs.values():
            if family is not None and o.family != family:
                continue
            if experimental_only and not o.is_experimental:
                continue
            if system_id is not None and system_id not in o.system_ids:
                continue
            if any(f in o.quality_flags for f in exclude_flags):
                continue
            out.append(o)
        return out

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for o in self._obs.values():
            d = o.model_dump(mode="json")
            row = {k: v for k, v in d.items() if k not in _NESTED}
            for k in _NESTED:
                row[k] = json.dumps(d.get(k), sort_keys=True)
            row["temperature_k"] = o.conditions.temperature_k
            row["phase"] = o.conditions.phase.value
            row["molecule_group"] = o.groups.molecule_group
            rows.append(row)
        return pd.DataFrame(rows)

    def to_parquet(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.to_frame().to_parquet(path, index=False)
        return sha256_file(path)

    @classmethod
    def from_parquet(cls, path: str) -> ExperimentRegistry:
        df = pd.read_parquet(path)
        obs = []
        for row in df.to_dict(orient="records"):
            d = {k: v for k, v in row.items() if k not in ("temperature_k", "phase", "molecule_group")}
            for k in _NESTED:
                d[k] = json.loads(d[k]) if d.get(k) is not None else None
            d = {k: v for k, v in d.items() if v is not None or k in ("value",)}
            obs.append(Observation.model_validate(d))
        return cls(obs)

    def summary(self) -> pd.DataFrame:
        df = self.to_frame()
        if df.empty:
            return df
        return df.groupby(["family", "value_kind"]).size().rename("n").reset_index()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ManifestEntry:
    path: str
    sha256: str
    n_records: int
    source: str
    license: str | None
    redistribution: str
    value_kinds: list[str]


@dataclass
class DatasetManifest:
    name: str
    version: str
    created: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    entries: list[ManifestEntry] = field(default_factory=list)
    extraction: dict = field(default_factory=dict)  # script, git hash/version, parameters
    notes: list[str] = field(default_factory=list)

    def add_registry(self, reg: ExperimentRegistry, path: str, source: str, license: str | None,
                     redistribution: str) -> None:
        digest = reg.to_parquet(path)
        kinds = sorted({o.value_kind.value if isinstance(o.value_kind, ValueKind) else o.value_kind for o in reg})
        self.entries.append(ManifestEntry(path, digest, len(reg), source, license, redistribution, kinds))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.__dict__ | {"entries": [e.__dict__ for e in self.entries]}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> DatasetManifest:
        with open(path) as f:
            d = json.load(f)
        d["entries"] = [ManifestEntry(**e) for e in d["entries"]]
        return cls(**d)

    def verify(self) -> list[str]:
        """Return entries whose file hash no longer matches (tampering / regeneration drift)."""
        bad = []
        for e in self.entries:
            if not os.path.exists(e.path) or sha256_file(e.path) != e.sha256:
                bad.append(e.path)
        return bad
