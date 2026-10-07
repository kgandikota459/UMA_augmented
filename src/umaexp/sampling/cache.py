"""Sample caches: detached configurations plus the identity of what produced them.

A cache is valid for reweighting only with the same system (atom ordering,
masses, charge/spin, task, pbc), the same thermodynamic state and the
sampling potential recorded in it. `u_sampling` holds the energies of the
potential that actually drove the dynamics (possibly an MPS float32 copy);
reweighting to U_w uses dU = U_w(R_i) - U_sampling(R_i), which is exact
importance sampling regardless of the sampler's precision.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch


@dataclass(frozen=True)
class ThermoState:
    temperature_k: float
    pressure_bar: float | None = None  # None -> NVT
    ensemble: str = "NVT"

    def __post_init__(self):
        if self.ensemble not in ("NVT", "NPT", "NVE"):
            raise ValueError(self.ensemble)
        if self.ensemble == "NPT" and self.pressure_bar is None:
            raise ValueError("NPT needs a pressure")


@dataclass
class SamplingProtocol:
    integrator: str = "langevin_baoab"
    dt_fs: float = 0.5
    friction_per_ps: float = 1.0
    n_equilibration: int = 2000
    n_production: int = 10000
    stride: int = 10
    barostat_every: int = 25
    barostat_max_dlnv: float = 0.01
    seed: int = 0
    device: str = "cpu"
    dtype: str = "float64"
    enhanced_sampling: str | None = None  # e.g. PLUMED bias; bias energies then stored in `bias`


@dataclass
class SampleCache:
    system_id: str
    system_hash: str
    state: ThermoState
    protocol: SamplingProtocol
    sampling_fingerprint: str  # parameter fingerprint of the potential that sampled
    positions: np.ndarray  # (F, N, 3) A
    cells: np.ndarray | None  # (F, 3, 3) A
    u_sampling: np.ndarray  # (F,) eV, energies of the sampling potential
    bias: np.ndarray | None = None  # (F,) eV, bias potential if enhanced sampling was used
    replica: int = 0
    created_unix: float = field(default_factory=time.time)
    diagnostics: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.positions.shape[0])

    @property
    def volumes(self) -> np.ndarray | None:
        if self.cells is None:
            return None
        return np.abs(np.linalg.det(self.cells))

    def positions_tensor(self, dtype=torch.float64):
        return torch.as_tensor(self.positions, dtype=dtype)

    def cells_tensor(self, dtype=torch.float64):
        return None if self.cells is None else torch.as_tensor(self.cells, dtype=dtype)

    def check_compatible(self, system_hash: str, state: ThermoState):
        if system_hash != self.system_hash:
            raise ValueError(f"cache {self.system_id}: system hash {self.system_hash} != {system_hash}")
        if state != self.state:
            raise ValueError(f"cache {self.system_id}: state {self.state} != {state}")

    # -- persistence ---------------------------------------------------
    def save(self, directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        stem = os.path.join(directory, f"{self.system_id}_r{self.replica}_{self.sampling_fingerprint[:8]}")
        arrays = {"positions": self.positions, "u_sampling": self.u_sampling}
        if self.cells is not None:
            arrays["cells"] = self.cells
        if self.bias is not None:
            arrays["bias"] = self.bias
        np.savez_compressed(stem + ".npz", **arrays)
        meta = {
            "system_id": self.system_id,
            "system_hash": self.system_hash,
            "state": asdict(self.state),
            "protocol": asdict(self.protocol),
            "sampling_fingerprint": self.sampling_fingerprint,
            "replica": self.replica,
            "created_unix": self.created_unix,
            "diagnostics": self.diagnostics,
            "meta": self.meta,
        }
        with open(stem + ".json", "w") as f:
            json.dump(meta, f, indent=2, default=float)
        return stem

    @classmethod
    def load(cls, stem: str) -> SampleCache:
        with open(stem + ".json") as f:
            meta = json.load(f)
        arr = np.load(stem + ".npz")
        return cls(
            system_id=meta["system_id"],
            system_hash=meta["system_hash"],
            state=ThermoState(**meta["state"]),
            protocol=SamplingProtocol(**meta["protocol"]),
            sampling_fingerprint=meta["sampling_fingerprint"],
            positions=arr["positions"],
            cells=arr["cells"] if "cells" in arr else None,
            u_sampling=arr["u_sampling"],
            bias=arr["bias"] if "bias" in arr else None,
            replica=meta["replica"],
            created_unix=meta["created_unix"],
            diagnostics=meta["diagnostics"],
            meta=meta["meta"],
        )
