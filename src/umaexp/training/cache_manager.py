"""Sample-cache lifecycle: build, reuse, extend, refresh, with an audit trail.

The reference potential of every cache is unambiguous: caches record the
fingerprint of the exact weights that sampled them (`sampling_fingerprint`).
Sampling may run on a different device/precision copy (e.g. MPS float32);
reweighting uses that copy's energies as the reference, so the importance
weights stay exact.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field, replace

import numpy as np

from ..potential.base import Potential
from ..reweighting.difftre import ReweightedEnsemble
from ..sampling.cache import SampleCache
from ..sampling.md import run_sampling
from .tasks import EnsembleTask

log = logging.getLogger(__name__)


@dataclass
class CacheEvent:
    task: str
    action: str
    reason: str
    fingerprint: str
    n_frames: int
    wall_s: float
    time_unix: float = field(default_factory=time.time)


class CacheManager:
    def __init__(self, directory: str | None = None, sampling_device: str = "cpu",
                 sampling_dtype: str = "float64"):
        self.directory = directory
        self.sampling_device = sampling_device
        self.sampling_dtype = sampling_dtype
        self.caches: dict[str, SampleCache] = {}
        self.events: list[CacheEvent] = []

    def _sampler_copy(self, potential: Potential) -> Potential:
        if self.sampling_device == "cpu" and self.sampling_dtype == "float64":
            return potential
        import torch

        if hasattr(potential, "to_device"):
            return potential.to_device(self.sampling_device, getattr(torch, self.sampling_dtype))
        return potential

    def _sample(self, task: EnsembleTask, potential: Potential, reason: str, positions=None, cell=None) -> SampleCache:
        t0 = time.time()
        sampler = self._sampler_copy(potential)
        proto = replace(task.protocol, device=self.sampling_device, dtype=self.sampling_dtype)
        cache = run_sampling(sampler, task.system, task.state, proto,
                             positions if positions is not None else task.start_positions,
                             cell if cell is not None else task.start_cell)
        # fingerprint of the *training-precision* weights that the sampler copies
        cache.meta["training_fingerprint"] = potential.parameter_fingerprint()
        if self.directory:
            cache.save(os.path.join(self.directory, task.name))
        self.events.append(CacheEvent(task.name, "build" if reason == "missing" else "refresh", reason,
                                      cache.sampling_fingerprint, cache.n_frames, time.time() - t0))
        return cache

    def get(self, task: EnsembleTask, potential: Potential) -> SampleCache:
        if task.name not in self.caches:
            self.caches[task.name] = self._sample(task, potential, "missing")
        return self.caches[task.name]

    def ensemble(self, task: EnsembleTask, potential: Potential) -> ReweightedEnsemble:
        return ReweightedEnsemble(self.get(task, potential), task.system, task.state, task.chunk_size)

    def refresh(self, task: EnsembleTask, potential: Potential, reason: str) -> SampleCache:
        old = self.caches.get(task.name)
        pos = cell = None
        if old is not None:  # continue from the last frame to shorten equilibration
            import torch

            pos = torch.as_tensor(old.positions[-1])
            cell = None if old.cells is None else torch.as_tensor(old.cells[-1])
        self.caches[task.name] = self._sample(task, potential, reason, pos, cell)
        return self.caches[task.name]

    def extend(self, task: EnsembleTask, potential: Potential, reason: str) -> SampleCache:
        """Append new frames sampled with the SAME weights (fingerprint must match)."""
        old = self.caches[task.name]
        sampler = self._sampler_copy(potential)
        if sampler.parameter_fingerprint() != old.sampling_fingerprint:
            return self.refresh(task, potential, f"extend requested but weights changed ({reason})")
        import torch

        t0 = time.time()
        proto = replace(task.protocol, device=self.sampling_device, dtype=self.sampling_dtype,
                        n_equilibration=0, seed=task.protocol.seed + len(self.events) + 1)
        new = run_sampling(sampler, task.system, task.state, proto, torch.as_tensor(old.positions[-1]),
                           None if old.cells is None else torch.as_tensor(old.cells[-1]))
        merged = replace(old,
                         positions=np.concatenate([old.positions, new.positions]),
                         cells=None if old.cells is None else np.concatenate([old.cells, new.cells]),
                         u_sampling=np.concatenate([old.u_sampling, new.u_sampling]))
        self.caches[task.name] = merged
        self.events.append(CacheEvent(task.name, "extend", reason, old.sampling_fingerprint, merged.n_frames,
                                      time.time() - t0))
        return merged
