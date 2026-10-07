"""Evaluate checkpoints with FRESH simulations and full forward models.

Training-cache (reweighted) predictions are not accepted as evidence of
improvement: every reported number here comes from new sampling with the
evaluated weights (uniform weights, no reweighting), with correlation-aware
errors, compared with held-out experimental targets.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import torch

from ..reweighting.difftre import ReweightedEnsemble
from ..sampling.md import run_sampling
from ..training.tasks import DirectTask, EnsembleTask


def _sampler_for(potential, protocol):
    """Device/precision copy for dynamics (e.g. MPS fp32); predictions stay in fp64."""
    dev = getattr(potential, "device_name", "cpu")
    if hasattr(potential, "to_device") and (protocol.device != dev or protocol.dtype != "float64"):
        return potential.to_device(protocol.device, getattr(torch, protocol.dtype))
    return potential


def evaluate_tasks(potential, tasks: list, fresh: bool = True, protocol_seed_offset: int = 10_000,
                   label: str = "") -> pd.DataFrame:
    rows = []
    for t in tasks:
        if isinstance(t, DirectTask):
            with torch.enable_grad():
                pred = t.predict(potential, 0)
            tgt = t.get_target(pred)
            v = pred.value.detach().reshape(-1)
            var = (pred.variance.detach().reshape(-1) if pred.variance is not None else torch.zeros_like(v))
        elif isinstance(t, EnsembleTask):
            proto = replace(t.protocol, seed=t.protocol.seed + protocol_seed_offset) if fresh else t.protocol
            cache = run_sampling(_sampler_for(potential, proto), t.system, t.state, proto,
                                 t.start_positions, t.start_cell)
            ens = ReweightedEnsemble(cache, t.system, t.state, t.chunk_size)
            with torch.no_grad():
                pred = t.observable.predict(potential, ens, "surrogate")
            tgt = t.get_target(pred)
            v = pred.value.detach().reshape(-1)
            var = pred.variance.detach().reshape(-1)
        else:
            raise TypeError(type(t))
        y = tgt.value.reshape(-1)
        sd = torch.sqrt(tgt.meas_sd.reshape(-1) ** 2 + var)
        for k in range(v.numel()):
            rows.append({"label": label, "task": t.name, "family": t.family,
                         "observation": tgt.observation_ids[k] if k < len(tgt.observation_ids) else f"{t.name}[{k}]",
                         "prediction": float(v[k]), "target": float(y[k]), "residual": float(v[k] - y[k]),
                         "z": float((v[k] - y[k]) / sd[k]) if float(sd[k]) > 0 else float("nan"),
                         "valid": pred.valid, "reason": pred.reason, "fresh": fresh})
    return pd.DataFrame(rows)


def compare_reweighted_to_fresh(potential, task: EnsembleTask, cache, n_sigma: float = 3.0) -> dict:
    """Gate: the reweighted estimate on the training cache must agree with a fresh run."""
    ens_rw = ReweightedEnsemble(cache, task.system, task.state, task.chunk_size)
    with torch.no_grad():
        rw = task.observable.predict(potential, ens_rw, "surrogate")
    proto = replace(task.protocol, seed=task.protocol.seed + 777)
    fresh = run_sampling(_sampler_for(potential, proto), task.system, task.state, proto,
                         task.start_positions, task.start_cell)
    ens_f = ReweightedEnsemble(fresh, task.system, task.state, task.chunk_size)
    with torch.no_grad():
        fr = task.observable.predict(potential, ens_f, "surrogate")
    diff = (rw.value.detach() - fr.value.detach()).reshape(-1)
    sd = torch.sqrt(rw.variance.reshape(-1) + fr.variance.reshape(-1))
    z = (diff / sd).abs().max()
    return {"reweighted": rw.value.detach().tolist(), "fresh": fr.value.detach().tolist(),
            "max_abs_z": float(z), "agree": bool(z < n_sigma),
            "ess_fraction": rw.diagnostics.get("ess_fraction")}
