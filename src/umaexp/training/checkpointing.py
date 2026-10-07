"""Training-state checkpoints: trainable weights, optimizer, RNG, identities."""

from __future__ import annotations

import json
import os
import random

import numpy as np
import torch


def save_training_state(path: str, potential, optimizer, step: int, extra: dict | None = None) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    state = {
        "step": step,
        "trainable": {n: p.detach().cpu().clone() for n, p in potential.trainable_named_parameters()},
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()},
        "base_fingerprint": getattr(potential, "base_fingerprint", None),
        "fingerprint": potential.parameter_fingerprint(trainable_only=True),
        "extra": extra or {},
    }
    torch.save(state, path)
    with open(path + ".json", "w") as f:
        json.dump({k: state[k] for k in ("step", "base_fingerprint", "fingerprint")} | {"extra": state["extra"]},
                  f, indent=2, default=str)
    return path


def load_training_state(path: str, potential, optimizer=None) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["base_fingerprint"] != getattr(potential, "base_fingerprint", None):
        raise ValueError("checkpoint was trained from a different base checkpoint")
    named = dict(potential.named_parameters())
    with torch.no_grad():
        for n, v in state["trainable"].items():
            named[n].copy_(v.to(named[n].device, named[n].dtype))
            named[n].requires_grad_(True)
    if optimizer is not None and state["optimizer"] is not None:
        optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["rng"]["torch"])
    np.random.set_state(state["rng"]["numpy"])
    random.setstate(state["rng"]["python"])
    return state
