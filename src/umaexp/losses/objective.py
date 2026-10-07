r"""Combined objective with explicit family normalization.

    L = sum_k lambda_k * l_k / n_k  +  lambda_Q L_quantum  +  lambda_B L_baseline
        + lambda_P L_physical  +  sum_j prior_j(phi_j)

Three things are kept separate on purpose:
  * statistical weighting: inside l_k via the covariance S_k;
  * normalization n_k: so thousands of correlated spectral bins do not
    outweigh a handful of thermodynamic points by count alone
    ("n_obs": number of observations; "eff_dof": effective independent
    count from the correlation matrix, n_eff = (tr C)^2 / tr(C^2));
  * scientific priority lambda_k: an explicit, logged choice.
Computational scheduling (how often a family is evaluated) is handled by the
trainer, not by these weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..observables.base import Prediction
from .likelihood import build_covariance, censored_gaussian_nll, gaussian_nll, student_t_nll


@dataclass
class Target:
    """Experimental target for one prediction (already in prediction units)."""

    observation_ids: list[str]
    family: str
    value: torch.Tensor  # (K,)
    meas_sd: torch.Tensor  # (K,)
    correlation_groups: list[str | None] | None = None
    correlation_rho: float = 0.0
    censoring: tuple[float | None, float | None] | None = None
    likelihood: str = "gaussian"  # gaussian | student_t | censored
    student_nu: float = 4.0


@dataclass
class FamilyWeight:
    priority: float = 1.0
    normalization: str = "n_obs"  # n_obs | eff_dof | n_targets (each target, e.g. a molecule, counts once) | none


@dataclass
class ObjectiveBreakdown:
    total: torch.Tensor
    per_family: dict[str, float] = field(default_factory=dict)
    per_observation: dict[str, float] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)


def _eff_dof(cov: torch.Tensor) -> float:
    d = torch.sqrt(torch.diagonal(cov))
    C = cov / (d[:, None] * d[None, :])
    return float(torch.trace(C) ** 2 / torch.trace(C @ C))


def observation_nll(pred: Prediction, tgt: Target) -> tuple[torch.Tensor, float]:
    value = pred.value.reshape(-1).to(torch.float64)
    extra = pred.variance
    if extra is not None:
        extra = extra.reshape(-1) if extra.dim() < 2 else extra
    if tgt.likelihood == "censored":
        lo, hi = tgt.censoring
        sd = torch.sqrt(tgt.meas_sd**2 + (extra if extra is not None else 0.0))
        return censored_gaussian_nll(value, sd, lo, hi), 1.0
    cov = build_covariance(tgt.meas_sd, extra, tgt.correlation_groups, tgt.correlation_rho)
    r = value - tgt.value.to(torch.float64)
    if tgt.likelihood == "student_t":
        return student_t_nll(r, torch.sqrt(torch.diagonal(cov)), tgt.student_nu), float(r.numel())
    return gaussian_nll(r, cov), _eff_dof(cov)


def combine_objective(
    items: list[tuple[Prediction, Target]],
    weights: dict[str, FamilyWeight] | None = None,
    extra_terms: dict[str, torch.Tensor] | None = None,
) -> ObjectiveBreakdown:
    weights = weights or {}
    fam_terms: dict[str, list[torch.Tensor]] = {}
    fam_count: dict[str, float] = {}
    out = ObjectiveBreakdown(torch.zeros((), dtype=torch.float64))
    for pred, tgt in items:
        key = ",".join(tgt.observation_ids)
        if not pred.valid:
            out.skipped[key] = pred.reason or "invalid prediction"
            continue
        nll, eff = observation_nll(pred, tgt)
        w = weights.get(tgt.family, FamilyWeight())
        fam_terms.setdefault(tgt.family, []).append(nll)
        fam_count[tgt.family] = fam_count.get(tgt.family, 0.0) + (
            eff if w.normalization == "eff_dof" else (len(tgt.observation_ids) if w.normalization == "n_obs" else 1.0))
        out.per_observation[key] = float(nll.detach())
    for fam, terms in fam_terms.items():
        w = weights.get(fam, FamilyWeight())
        n = fam_count[fam] if w.normalization != "none" else 1.0
        val = w.priority * torch.stack(terms).sum() / max(n, 1e-12)
        out.per_family[fam] = float(val.detach())
        out.total = out.total + val
    for name, t in (extra_terms or {}).items():
        out.per_family[name] = float(t.detach())
        out.total = out.total + t
    return out
