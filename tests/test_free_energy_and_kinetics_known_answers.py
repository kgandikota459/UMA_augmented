"""Known-answer tests: solvation-cycle endpoint corrections and dynamical estimators."""

import math

import numpy as np
import pytest
import torch
from toy_helpers import cache_from_frames

from umaexp import units
from umaexp.observables.free_energy import EndpointCorrection, SolvationCycle, standard_state_shift, subsystem
from umaexp.observables.kinetics import YEH_HUMMER_XI, msd_diffusion, unrolled_nve_gradient, vacf_spectrum, yeh_hummer
from umaexp.potential.toy import HarmonicWell
from umaexp.systems import System, isotopologue_masses

T = 300.0


def _harmonic_reference(k_ref, n=3, F=20000, seed=0):
    """Exact samples of a reference harmonic 'MM' potential; UMA region = all atoms."""
    beta = units.beta(T)
    pos = np.random.default_rng(seed).normal(0.0, 1.0 / math.sqrt(beta * k_ref), size=(F, n, 3))
    u = 0.5 * k_ref * (pos**2).sum(axis=(1, 2))
    sys = System(f"ref{k_ref}", (6,) * n, isotopologue_masses((6,) * n), task="toy", spin=0,
                 positions=np.zeros((n, 3)))
    cache = cache_from_frames(sys, pos, u=u)
    cache.meta["u_ref_region"] = u.copy()
    return sys, cache


def test_endpoint_correction_matches_analytic_value_and_gradient():
    n, k_ref, k_t = 3, 1.0, 1.3
    sys, cache = _harmonic_reference(k_ref, n)
    target = HarmonicWell(k=k_t)
    pred = EndpointCorrection(cache, sys, list(range(n)), T).predict(target)
    kT = units.KB_EV_K * T
    assert float(pred.value) == pytest.approx(1.5 * n * kT * math.log(k_t / k_ref), rel=0.02)
    (g,) = torch.autograd.grad(pred.value.sum(), target.log_k)
    assert float(g) == pytest.approx(1.5 * n * kT, rel=0.03)  # <dU_t/dlnk>_t = (3n/2) kT
    assert pred.valid and pred.diagnostics["ess_fraction"] > 0.5


def test_solvation_cycle_closes_independently_of_target_potential():
    n = 3
    sol_sys, sol = _harmonic_reference(1.0, n, seed=1)
    gas_sys, gas = _harmonic_reference(1.5, n, seed=2)
    kT = units.KB_EV_K * T
    expected = -0.1 + 1.5 * n * kT * math.log(1.5)  # target-dependence cancels around the cycle
    for k_t in (1.2, 1.4):
        target = HarmonicWell(k=k_t)
        cyc = SolvationCycle(-0.1, 0.02, EndpointCorrection(sol, sol_sys, list(range(n)), T),
                             EndpointCorrection(gas, gas_sys, list(range(n)), T))
        pred = cyc.predict(target)
        assert float(pred.value) == pytest.approx(expected, abs=2e-3)
        assert float(pred.variance[0]) == pytest.approx(0.02**2)
        (g,) = torch.autograd.grad(pred.value.sum(), target.log_k)
        assert abs(float(g)) < 0.01  # both legs share the same <dU/dlnk>


def test_standard_state_shift_between_1atm_and_1M():
    c_gas = 101325.0 / (units.GAS_CONSTANT_J_MOL_K * 298.15) / 1000.0  # mol/L of an ideal gas at 1 atm
    shift = standard_state_shift(c_gas, 1.0, 298.15) * units.EV_TO_KCAL_MOL
    assert shift == pytest.approx(-1.894, abs=2e-3)


def test_subsystem_extracts_atoms_and_drops_periodicity():
    zs = (8, 1, 1, 8, 1, 1)
    sys = System("w2", zs, isotopologue_masses(zs), positions=np.arange(18.0).reshape(6, 3))
    sub = subsystem(sys, [3, 4, 5], "w_b")
    assert sub.atomic_numbers == (8, 1, 1) and np.allclose(sub.positions, np.arange(9.0, 18.0).reshape(3, 3))
    assert sub.masses == sys.masses[3:] and not sub.is_periodic and sub.system_id == "w_b"


def test_msd_diffusion_recovers_brownian_D_and_com_mode():
    rng = np.random.default_rng(0)
    D, dt, F, N = 1e-3, 1.0, 2000, 200
    x = np.cumsum(rng.normal(0.0, math.sqrt(2 * D * dt), size=(F, N, 3)), axis=0)
    out = msd_diffusion(x, dt)
    assert out["D_A2_per_fs"] == pytest.approx(D, rel=0.08)
    assert out["D_cm2_per_s"] == pytest.approx(0.1 * out["D_A2_per_fs"], rel=1e-12)  # 1 A^2/fs = 0.1 cm^2/s
    x2 = np.concatenate([x, x + np.array([1.0, 0.0, 0.0])], axis=1)  # rigid diatomics
    out2 = msd_diffusion(x2, dt, molecule_index=list(range(N)) * 2, masses=np.array([16.0] * N + [1.0] * N))
    assert out2["D_A2_per_fs"] == pytest.approx(out["D_A2_per_fs"], rel=1e-10)


def test_yeh_hummer_finite_size_correction():
    Tk, eta, L = 298.15, 0.89e-3, 30.0
    corr = units.BOLTZMANN_J_K * Tk * YEH_HUMMER_XI / (6 * math.pi * eta * L * 1e-10) * 1e4  # cm^2/s
    assert yeh_hummer(2.0e-5, Tk, eta, L) == pytest.approx(2.0e-5 + corr, rel=1e-12)
    assert corr == pytest.approx(2.32e-6, rel=0.01)


def test_vacf_spectrum_peaks_at_the_oscillation_wavenumber():
    nu, dt = 1000.0, 1.0
    t = np.arange(4000) * dt * 1e-15
    omega = 2 * math.pi * units.SPEED_OF_LIGHT_M_S * 100.0 * nu
    v = np.cos(omega * t)[:, None, None] * np.ones((1, 2, 3))
    out = vacf_spectrum(v, np.array([1.0, 2.0]), dt)
    peak = out["wavenumber"][np.argmax(out["vdos"][1:]) + 1]
    assert peak == pytest.approx(nu, abs=10.0)


def test_unrolled_nve_gradient_matches_finite_differences_and_analytic_orbit():
    sys = System("hw1", (6,), isotopologue_masses((6,)), task="toy", spin=0, positions=np.zeros((1, 3)))
    x0 = torch.tensor([[0.3, -0.1, 0.2]], dtype=torch.float64)
    v0 = torch.tensor([[0.0, 0.01, 0.0]], dtype=torch.float64)
    n_steps, dt = 50, 2.0

    def run(pot):
        return unrolled_nve_gradient(pot, sys, x0, v0, n_steps, dt, observable=lambda traj: (traj[-1] ** 2).sum())

    pot = HarmonicWell(k=1.0)
    (g,) = torch.autograd.grad(run(pot), pot.log_k)
    h = 1e-6
    with torch.no_grad():
        pot.log_k += h
    fp = float(run(pot).detach())
    with torch.no_grad():
        pot.log_k -= 2 * h
    fm = float(run(pot).detach())
    assert float(g) == pytest.approx((fp - fm) / (2 * h), rel=1e-6)
    # the discrete trajectory follows the analytic oscillator (O(dt^2) phase error)
    w = math.sqrt(1.0 / sys.masses[0])
    t = n_steps * dt * units.FS
    x_exact = x0 * math.cos(w * t) + v0 / w * math.sin(w * t)
    traj_end = unrolled_nve_gradient(HarmonicWell(k=1.0), sys, x0, v0, n_steps, dt, observable=lambda tr: tr[-1])
    assert torch.allclose(traj_end.detach(), x_exact, atol=2e-3)
