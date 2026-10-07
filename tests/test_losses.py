import math

import numpy as np
import pytest
import torch

from umaexp.losses import (
    FamilyWeight,
    LearnedNoise,
    Target,
    build_covariance,
    censored_gaussian_nll,
    combine_objective,
    gaussian_nll,
)
from umaexp.observables.base import Prediction


def test_gaussian_nll_matches_closed_form():
    r = torch.tensor([0.3, -0.2], dtype=torch.float64)
    S = torch.tensor([[0.04, 0.01], [0.01, 0.09]], dtype=torch.float64)
    ref = 0.5 * float(r @ torch.linalg.solve(S, r))
    assert float(gaussian_nll(r, S)) == pytest.approx(ref, rel=1e-12)
    ref_ld = ref + 0.5 * math.log(float(torch.linalg.det(S)))
    assert float(gaussian_nll(r, S, include_logdet=True)) == pytest.approx(ref_ld, rel=1e-12)


def test_correlation_groups_build_offdiagonal():
    S = build_covariance(torch.tensor([1.0, 2.0, 3.0]), None, ["a", "a", None], rho=0.5)
    assert float(S[0, 1]) == pytest.approx(1.0) and float(S[0, 2]) == 0.0


def test_censored_likelihood_monotone():
    sd = torch.tensor(1.0, dtype=torch.float64)
    far_ok = censored_gaussian_nll(torch.tensor(5.0, dtype=torch.float64), sd, lower=0.0)
    violating = censored_gaussian_nll(torch.tensor(-3.0, dtype=torch.float64), sd, lower=0.0)
    assert float(far_ok) < 1e-5 and float(violating) > 5.0


def test_family_normalization_prevents_count_domination():
    # 1000 correlated "spectral bins" vs one thermodynamic point, equal per-point misfit
    bins = Prediction(torch.full((1000,), 1.0, dtype=torch.float64), torch.zeros(1000, dtype=torch.float64))
    tb = Target([f"b{i}" for i in range(1000)], "scattering", torch.zeros(1000, dtype=torch.float64),
                torch.ones(1000, dtype=torch.float64))
    thermo = Prediction(torch.tensor([1.0], dtype=torch.float64), torch.zeros(1, dtype=torch.float64))
    tt = Target(["t0"], "density", torch.zeros(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64))
    out = combine_objective([(bins, tb), (thermo, tt)], {"scattering": FamilyWeight(1.0, "n_obs"),
                                                          "density": FamilyWeight(1.0, "n_obs")})
    assert out.per_family["scattering"] == pytest.approx(out.per_family["density"])


def test_invalid_predictions_are_skipped_and_recorded():
    bad = Prediction(torch.tensor([1.0], dtype=torch.float64), None, valid=False, reason="imaginary mode")
    t = Target(["x"], "vibrational", torch.zeros(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64))
    out = combine_objective([(bad, t)])
    assert out.skipped == {"x": "imaginary mode"} and float(out.total) == 0.0


def test_learned_noise_prior_resists_inflation():
    ln = LearnedNoise(init=0.1, prior_scale=0.1)
    opt = torch.optim.Adam(ln.parameters(), lr=0.05)
    r = torch.tensor(np.full(5, 3.0))  # large residuals the model "wants" to explain away
    for _ in range(500):
        opt.zero_grad()
        var = 0.01 + ln.variance()
        loss = (0.5 * r**2 / var + 0.5 * torch.log(var)).sum() + 50 * ln.prior_nll()
        loss.backward()
        opt.step()
    # without the prior sigma -> |r| = 3; with it the inflation is strongly limited
    assert math.exp(float(ln.log_s)) < 1.0
