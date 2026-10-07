import numpy as np
import pytest
import torch

from umaexp import units
from umaexp.observables.vibrational import (
    ExperimentalMode,
    VibrationalConfig,
    VibrationalForwardModel,
    VibrationalObservable,
    assign_modes,
    attach_implicit_geometry,
    degenerate_groups,
    harmonic_modes,
    optimize_geometry,
)
from umaexp.potential.toy import MorseDiatomic, ValenceToy
from umaexp.systems import System, isotopologue_masses


def test_morse_diatomic_analytic():
    D, a, re = 4.6, 2.2, 0.97
    pot = MorseDiatomic(D, a, re)
    sys = System("OH_toy", (8, 1), isotopologue_masses((8, 1)), task="toy", spin=0,
                 positions=np.array([[0, 0, 0], [0, 0, 1.0]]))
    x, info = optimize_geometry(pot, sys)
    assert info["converged"]
    nm = harmonic_modes(pot, sys, x, create_graph=False)
    assert nm.linear and nm.n_external == 5 and nm.wavenumbers.numel() == 1
    m1, m2 = sys.masses
    mu = m1 * m2 / (m1 + m2)
    expected = np.sqrt(2 * D * a**2 / mu) * units.EIGVAL_TO_WAVENUMBER
    assert float(nm.wavenumbers[0]) == pytest.approx(expected, rel=1e-8)


def test_projection_counts_and_isotope_shift(water_toy):
    sys, pot = water_toy
    x, info = optimize_geometry(pot, sys)
    nm = harmonic_modes(pot, sys, x, create_graph=False)
    assert nm.n_external == 6 and nm.wavenumbers.numel() == 3
    assert (nm.wavenumbers > 0).all()
    d2o = sys.with_isotopes({1: 2, 2: 2})
    nm_d = harmonic_modes(pot, d2o, x, create_graph=False)
    ratio = (nm_d.wavenumbers / nm.wavenumbers).numpy()
    # heavier isotopologue on the same surface: all frequencies drop, ratios between 1/sqrt(2) and 1
    assert np.all(ratio < 1.0) and np.all(ratio > 1 / np.sqrt(2) - 1e-9)


def test_rotation_invariance_of_spectrum(water_toy):
    sys, pot = water_toy
    x, _ = optimize_geometry(pot, sys)
    nm = harmonic_modes(pot, sys, x, create_graph=False)
    q = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(3)))[0]
    nm_r = harmonic_modes(pot, sys, x @ q.T + torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64), create_graph=False)
    assert torch.allclose(nm.wavenumbers, nm_r.wavenumbers, rtol=1e-9, atol=1e-7)


def _freqs_at_theta(theta0):
    r0 = 0.9572
    pot = ValenceToy(bonds=[(0, 1), (0, 2)], angles=[(1, 0, 2)], kb=[50.0], r0=[r0], ka=[4.0],
                     theta0=[theta0])
    pos = np.array([[0.0, 0.0, 0.0], [r0, 0.0, 0.0], [r0 * np.cos(1.8), r0 * np.sin(1.8), 0.0]])
    sys = System("t", (8, 1, 1), isotopologue_masses((8, 1, 1)), task="toy", spin=0, positions=pos)
    x, _ = optimize_geometry(pot, sys, fmax=1e-9)
    return pot, sys, x


def test_implicit_geometry_gradient_matches_reoptimized_fd():
    theta0 = np.deg2rad(104.5)
    pot, sys, x = _freqs_at_theta(theta0)
    xw = attach_implicit_geometry(pot, sys, x)
    nm = harmonic_modes(pot, sys, xw, create_graph=True)
    (g_implicit,) = torch.autograd.grad(nm.wavenumbers.sum(), pot.theta0)
    nm_fixed = harmonic_modes(pot, sys, x.detach(), create_graph=True)
    (g_fixed,) = torch.autograd.grad(nm_fixed.wavenumbers.sum(), pot.theta0)
    h = 1e-5
    vals = []
    for t in (theta0 + h, theta0 - h):
        p2, s2, x2 = _freqs_at_theta(t)
        vals.append(float(harmonic_modes(p2, s2, x2, create_graph=False).wavenumbers.sum()))
    fd = (vals[0] - vals[1]) / (2 * h)
    assert float(g_implicit) == pytest.approx(fd, rel=1e-5)
    # the fixed-geometry approximation misses the geometry response here
    assert abs(float(g_fixed) - fd) > 1e-3 * abs(fd)


def test_degenerate_linear_bend_and_assignment():
    # linear symmetric triatomic: bonds + linear bend -> doubly degenerate bend
    pos = np.array([[-1.16, 0.0, 0.0], [0.0, 0.0, 0.0], [1.16, 0.0, 0.0]])
    sys = System("CO2_toy", (8, 6, 8), isotopologue_masses((8, 6, 8)), task="toy", spin=0, positions=pos)
    pot = ValenceToy(bonds=[(0, 1), (1, 2)], angles=[(0, 1, 2)], kb=[100.0], r0=[1.16], ka=[3.0],
                     theta0=[np.pi])
    nm = harmonic_modes(pot, sys, torch.tensor(pos), create_graph=False)
    nu = nm.wavenumbers.numpy()
    assert nm.linear and nu.size == 4
    groups = degenerate_groups(nu)
    assert sorted(len(g) for g in groups) == [1, 1, 2]
    exp = [ExperimentalMode("bend", float(np.mean(nu[[g for g in groups if len(g) == 2][0]])), 1.0, 2, "pi_u"),
           ExperimentalMode("sym", float(np.sort(nu)[2]), 1.0, 1, "sigma_g+"),
           ExperimentalMode("asym", float(np.sort(nu)[3]), 1.0, 1, "sigma_u+")]
    m = assign_modes(nu, exp)
    assert [len(x.comp_indices) for x in m] == [2, 1, 1]


def test_observable_gradient_flows_to_parameters(water_toy):
    sys, pot = water_toy
    exp = [ExperimentalMode("bend", 1500.0, 1.0), ExperimentalMode("s", 3600.0, 1.0),
           ExperimentalMode("a", 3700.0, 1.0)]
    obs = VibrationalObservable(sys, exp, VibrationalConfig(forward=VibrationalForwardModel(kind="harmonic")))
    pred = obs.predict(pot)
    assert pred.valid, pred.reason
    y, var = obs.targets()
    loss = (((pred.value - y) ** 2) / var).sum()
    grads = torch.autograd.grad(loss, [pot.kb, pot.ka, pot.r0, pot.theta0])
    assert all(torch.isfinite(g).all() for g in grads)
    assert float(grads[0].abs().sum()) > 0 and float(grads[1].abs().sum()) > 0


@pytest.mark.parametrize("scale", [None, 0.96])
def test_hvp_backward_equals_full_graph(water_toy, scale):
    """Memory-bounded Hessian-vector backward == full create_graph backward."""
    import copy

    sys, pot0 = water_toy
    exp = [ExperimentalMode("bend", 1500.0, 1.0), ExperimentalMode("s", 3600.0, 1.0),
           ExperimentalMode("a", 3700.0, 1.0)]
    grads = {}
    for mode in ("graph", "hvp"):
        pot = copy.deepcopy(pot0)
        s = None if scale is None else torch.nn.Parameter(torch.tensor(scale, dtype=torch.float64))
        fwd = VibrationalForwardModel("fundamental_scaled" if s is not None else "harmonic", scale=s, rel_model_sd=0.0)
        obs = VibrationalObservable(sys, exp, VibrationalConfig(forward=fwd, memory_mode=mode))
        pred = obs.predict(pot)
        y, var = obs.targets()
        loss = (((pred.value - y) ** 2) / var).sum()
        if mode == "graph":
            loss.backward()
        else:
            (gv,) = torch.autograd.grad(loss, pred.value, retain_graph=True)
            loss.backward()
            pred.backward(gv)
        grads[mode] = [p.grad.clone() for p in (pot.kb, pot.ka, pot.r0, pot.theta0)] + (
            [s.grad.clone()] if s is not None else [])
    for a, b in zip(grads["graph"], grads["hvp"]):
        assert torch.allclose(a, b, rtol=1e-8, atol=1e-10), (a, b)
