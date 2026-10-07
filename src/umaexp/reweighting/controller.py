"""Sampling controller: turn reweighting diagnostics into decisions.

Thresholds are *starting values to be tuned on validation systems*, not
established constants. Nominal ESS cannot reveal a basin the sampler never
visited; that risk is addressed by fresh-simulation validation of saved
checkpoints (validation/evaluation.py), not by this controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .difftre import EnsembleEstimate


class Decision(str, Enum):
    ACCEPT = "accept"  # estimate usable for this step
    EXTEND = "extend"  # usable but statistically too noisy; sample more
    REFRESH = "refresh"  # overlap degraded: resample with the current weights
    REJECT_UPDATE = "reject_update"  # the last update collapsed overlap: revert/shrink step
    ABSTAIN = "abstain"  # no valid estimate (skip the observation this step)


@dataclass
class ControllerConfig:
    ess_refresh: float = 0.5  # refresh cache below this ESS fraction
    ess_reject: float = 0.1  # after an update, below this => reject the update
    min_effective_samples: float = 30.0
    max_weight: float = 0.2  # a single frame dominating the average
    sem_over_sigma_extend: float = 0.5  # sampling SEM vs experimental SD


@dataclass
class ControllerRecord:
    decision: Decision
    reason: str
    ess_fraction: float
    n_eff: float
    max_weight: float


def assess(est: EnsembleEstimate | None, exp_sigma: float | None, cfg: ControllerConfig,
           after_update: bool = False, prediction_sd: float | None = None) -> ControllerRecord:
    """Decide what to do with one ensemble estimate.

    `prediction_sd` is the sampling SD of the *reported* prediction (after the
    observable's transform, e.g. rho = M/<V>) in the same units as
    `exp_sigma`; the raw frame-average SEM is in different units in general.
    """
    if est is None:
        return ControllerRecord(Decision.ABSTAIN, "no estimate", 0.0, 0.0, 1.0)
    n_eff = est.diagnostics.get("n_eff_combined", est.ess)
    r = dict(ess_fraction=est.ess_fraction, n_eff=n_eff, max_weight=est.max_weight)
    if after_update and est.ess_fraction < cfg.ess_reject:
        return ControllerRecord(Decision.REJECT_UPDATE, f"ESS fraction {est.ess_fraction:.3f} < {cfg.ess_reject}", **r)
    if est.ess_fraction < cfg.ess_refresh or est.max_weight > cfg.max_weight:
        return ControllerRecord(Decision.REFRESH, f"ESS fraction {est.ess_fraction:.3f}, max weight {est.max_weight:.3f}", **r)
    if n_eff < cfg.min_effective_samples:
        return ControllerRecord(Decision.EXTEND, f"n_eff {n_eff:.1f} < {cfg.min_effective_samples}", **r)
    if exp_sigma is not None and prediction_sd is not None:
        if prediction_sd > cfg.sem_over_sigma_extend * exp_sigma:
            return ControllerRecord(Decision.EXTEND,
                                    f"sampling SD {prediction_sd:.3g} > {cfg.sem_over_sigma_extend} x sigma_exp", **r)
    return ControllerRecord(Decision.ACCEPT, "ok", **r)
