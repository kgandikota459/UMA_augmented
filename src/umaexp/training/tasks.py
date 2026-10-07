"""Training tasks: one experimental target + the forward model that predicts it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from ..losses.objective import Target
from ..observables.base import Prediction
from ..potential.base import Potential
from ..sampling.cache import SamplingProtocol, ThermoState
from ..systems.system import System


@dataclass
class DirectTask:
    """Observable computed directly from U_w (vibrations, thermochemistry, IE/EA)."""

    name: str
    family: str
    observable: Any  # has .predict(potential) -> Prediction
    target: Target | None = None
    target_fn: Any = None  # callable(prediction) -> Target (e.g. mode-matched targets)
    every: int = 1  # evaluate every n steps (computational scheduling)

    def predict(self, potential: Potential, step: int) -> Prediction:
        return self.observable.predict(potential)

    def get_target(self, pred: Prediction) -> Target:
        return self.target if self.target is not None else self.target_fn(pred)


@dataclass
class EnsembleTask:
    """Observable averaged over an equilibrium ensemble (via a sample cache)."""

    name: str
    family: str
    system: System
    state: ThermoState
    protocol: SamplingProtocol
    observable: Any  # has .predict(potential, ens, mode) -> Prediction
    target: Target
    start_positions: torch.Tensor | None = None
    start_cell: torch.Tensor | None = None
    mode: str = "surrogate"
    chunk_size: int = 32
    every: int = 1
    meta: dict = field(default_factory=dict)

    def get_target(self, pred: Prediction) -> Target:
        return self.target
