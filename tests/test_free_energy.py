import math

import numpy as np
import pytest
import torch

from umaexp.reweighting import bar_estimator, exp_estimator, mbar_free_energies, mbar_solve
from umaexp.reweighting.free_energy import target_state_free_energy


def _samples(k, n, rng):
    return rng.normal(0.0, 1.0 / math.sqrt(k), size=n)


def test_exp_and_bar_harmonic_exact():
    rng = np.random.default_rng(0)
    k0, k1 = 1.0, 1.6
    x0, x1 = _samples(k0, 20000, rng), _samples(k1, 20000, rng)
    k1_t = torch.tensor(k1, dtype=torch.float64, requires_grad=True)
    wf = 0.5 * (k1_t - k0) * torch.as_tensor(x0**2)
    wr = -0.5 * (k1_t - k0) * torch.as_tensor(x1**2)
    exact = 0.5 * math.log(k1 / k0)
    df_exp = exp_estimator(wf)
    df_bar = bar_estimator(wf, wr)
    assert float(df_exp) == pytest.approx(exact, abs=0.02)
    assert float(df_bar) == pytest.approx(exact, abs=0.01)
    # BAR's attached gradient is the estimator sensitivity at fixed samples:
    # check it against finite differences of the same estimator.
    (g,) = torch.autograd.grad(df_bar, k1_t)
    h = 1e-6
    vals = []
    for kk in (k1 + h, k1 - h):
        vals.append(float(bar_estimator(0.5 * (kk - k0) * torch.as_tensor(x0**2),
                                        -0.5 * (kk - k0) * torch.as_tensor(x1**2))))
    assert float(g) == pytest.approx((vals[0] - vals[1]) / (2 * h), rel=1e-5)
    # EXP with fixed state-0 samples is a valid training gradient: <du1/dk1>_1 = 0.5/k1
    (g_exp,) = torch.autograd.grad(exp_estimator(0.5 * (k1_t - k0) * torch.as_tensor(x0**2)), k1_t)
    assert float(g_exp) == pytest.approx(0.5 / k1, rel=0.05)


def test_bar_matches_pymbar():
    pymbar = pytest.importorskip("pymbar")
    rng = np.random.default_rng(1)
    wf = rng.normal(1.0, 1.2, 3000)
    wr = rng.normal(-0.4, 1.1, 2500)
    ref = pymbar.other_estimators.bar(wf, wr)["Delta_f"]
    ours = float(bar_estimator(torch.as_tensor(wf), torch.as_tensor(wr)))
    assert ours == pytest.approx(ref, abs=1e-8)


def test_mbar_values_and_implicit_gradient():
    pymbar = pytest.importorskip("pymbar")
    rng = np.random.default_rng(2)
    ks = np.array([1.0, 1.5, 2.2, 3.0])
    N_k = np.array([3000, 3000, 3000, 3000])
    x = np.concatenate([_samples(k, n, rng) for k, n in zip(ks, N_k)])
    u_kn = 0.5 * ks[:, None] * x[None, :] ** 2
    f = mbar_solve(u_kn, N_k)
    ref = pymbar.MBAR(u_kn, N_k).compute_free_energy_differences()["Delta_f"][0]
    assert np.allclose(f, ref, atol=1e-7)
    exact = 0.5 * np.log(ks / ks[0])
    assert np.allclose(f, exact, atol=0.02)

    k_t = torch.tensor(ks, dtype=torch.float64, requires_grad=True)
    u_t = 0.5 * k_t[:, None] * torch.as_tensor(x)[None, :] ** 2
    f_t = mbar_free_energies(u_t, N_k)
    (g,) = torch.autograd.grad(f_t[3], k_t)
    # finite differences re-solving MBAR with perturbed k_3 (same samples)
    h = 1e-5
    fp = mbar_solve(0.5 * (ks + h * np.eye(4)[3])[:, None] * x[None, :] ** 2, N_k)[3]
    fm = mbar_solve(0.5 * (ks - h * np.eye(4)[3])[:, None] * x[None, :] ** 2, N_k)[3]
    assert float(g[3]) == pytest.approx((fp - fm) / (2 * h), rel=1e-5)


def test_target_state_free_energy_training_gradient():
    rng = np.random.default_rng(3)
    ks = np.array([1.0, 2.0])
    N_k = np.array([5000, 5000])
    x = np.concatenate([_samples(k, n, rng) for k, n in zip(ks, N_k)])
    u_kn = 0.5 * ks[:, None] * x[None, :] ** 2
    kt = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
    f_t = target_state_free_energy(u_kn, N_k, 0.5 * kt * torch.as_tensor(x) ** 2)
    assert float(f_t) == pytest.approx(0.5 * np.log(1.7), abs=0.02)
    (g,) = torch.autograd.grad(f_t, kt)
    assert float(g) == pytest.approx(0.5 / 1.7, rel=0.03)  # <du_t/dk>_t = 0.5 <x^2>_t


def test_single_sampled_state_is_trivial_not_an_error():
    """Regression: EXP-style use of target_state_free_energy has K = 1 sampled state."""
    rng = np.random.default_rng(4)
    x = rng.normal(0.0, 1.0, 5000)
    u = (0.5 * x**2)[None, :]
    assert np.allclose(mbar_solve(u, [5000]), [0.0])
    f = mbar_free_energies(torch.as_tensor(u), [5000])
    assert f.shape == (1,) and float(f[0]) == 0.0
    kt = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    f_t = target_state_free_energy(u, [5000], 0.5 * kt * torch.as_tensor(x) ** 2)
    assert float(f_t) == pytest.approx(0.5 * math.log(1.5), abs=0.02)
