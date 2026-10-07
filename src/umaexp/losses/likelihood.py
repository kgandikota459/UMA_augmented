r"""Per-observation likelihoods.

Gaussian (constants dropped):
    l = 1/2 r^T S^{-1} r + 1/2 log det S,   S = S_meas + S_sampling + S_model

S_meas comes from the record (standard sd; correlation groups add shared
systematic terms), S_sampling from the estimator (correlation-aware
n_eff), S_model from the forward model (e.g. anharmonic/NQE discrepancy).
The log-det term matters only when S depends on w or on learned noise;
with fixed S it is a constant and is omitted to keep loss values readable.

Alternatives: Student-t for outlier-robust fits, censored Gaussian for
bounds ("> 10 uM", "< 1 Hz"). Learned noise scales are allowed only with a
prior that penalizes inflation, so difficult observables cannot be
"explained" by making them noisier.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def build_covariance(meas_sd: torch.Tensor, extra_var: torch.Tensor | None = None,
                     correlation_groups: list[str | None] | None = None, rho: float = 0.0) -> torch.Tensor:
    """Diagonal measurement variance + shared-systematic correlation + extra variances."""
    sd = meas_sd.to(torch.float64)
    S = torch.diag(sd**2)
    if correlation_groups is not None and rho > 0:
        for i, gi in enumerate(correlation_groups):
            for j, gj in enumerate(correlation_groups):
                if i != j and gi is not None and gi == gj:
                    S[i, j] = rho * sd[i] * sd[j]
    if extra_var is not None:
        ev = extra_var.to(torch.float64)
        S = S + (ev if ev.dim() == 2 else torch.diag(ev.reshape(-1)))
    return S


def gaussian_nll(residual: torch.Tensor, cov: torch.Tensor, include_logdet: bool = False) -> torch.Tensor:
    r = residual.reshape(-1, 1).to(torch.float64)
    L = torch.linalg.cholesky(cov)
    z = torch.linalg.solve_triangular(L, r, upper=False)
    nll = 0.5 * (z**2).sum()
    if include_logdet:
        nll = nll + torch.log(torch.diagonal(L)).sum()
    return nll


def student_t_nll(residual: torch.Tensor, scale: torch.Tensor, nu: float = 4.0) -> torch.Tensor:
    r = residual.to(torch.float64) / scale
    return (0.5 * (nu + 1.0) * torch.log1p(r**2 / nu) + torch.log(scale)).sum()


def censored_gaussian_nll(pred: torch.Tensor, sd: torch.Tensor, lower: float | None = None,
                          upper: float | None = None) -> torch.Tensor:
    """-log P(bound | pred): lower bound means the true value exceeds `lower`."""
    out = torch.zeros((), dtype=torch.float64)
    if lower is not None:
        out = out - torch.special.log_ndtr((pred - lower) / sd).sum()
    if upper is not None:
        out = out - torch.special.log_ndtr((upper - pred) / sd).sum()
    return out


class LearnedNoise(nn.Module):
    """Extra model-discrepancy SD per family, sigma = exp(log_s), with a
    half-normal prior of scale `prior_scale` (penalizes inflation)."""

    def __init__(self, init: float, prior_scale: float):
        super().__init__()
        self.log_s = nn.Parameter(torch.tensor(math.log(init), dtype=torch.float64))
        self.prior_scale = prior_scale

    def variance(self) -> torch.Tensor:
        return torch.exp(2 * self.log_s)

    def prior_nll(self) -> torch.Tensor:
        s = torch.exp(self.log_s)
        return 0.5 * (s / self.prior_scale) ** 2 - self.log_s  # half-normal on s, Jacobian for log param
