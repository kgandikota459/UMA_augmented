import math

import numpy as np
import pytest
import torch

from umaexp import units
from umaexp.potential.base import Potential
from umaexp.potential.toy import HarmonicWell
from umaexp.reweighting import ReweightedEnsemble, covariance_gradient
from umaexp.sampling import SampleCache, SamplingProtocol, ThermoState
from umaexp.systems import System

T = 300.0
BETA = units.beta(T)


def _exact_harmonic_cache(k0, n_atoms=4, n_frames=4000, seed=0):
    """Exact canonical samples of a 3D harmonic well at k0 (no MD error)."""
    rng = np.random.default_rng(seed)
    sd = math.sqrt(1.0 / (BETA * k0))
    pos = rng.normal(0.0, sd, size=(n_frames, n_atoms, 3))
    u = 0.5 * k0 * (pos**2).sum(axis=(1, 2))
    sys = System("hw", (18,) * n_atoms, (39.948,) * n_atoms, task="toy", spin=0,
                 positions=np.zeros((n_atoms, 3)))
    cache = SampleCache(sys.system_id, sys.atom_order_hash(), ThermoState(T), SamplingProtocol(),
                        "exact", pos, None, u)
    return sys, cache


def _r2(p, c):
    return (p**2).sum(dim=(1, 2))[:, None]


def test_reweighted_mean_and_gradient_match_exact():
    k0 = 2.0
    sys, cache = _exact_harmonic_cache(k0)
    pot = HarmonicWell(k=2.3)  # target differs from sampling
    ens = ReweightedEnsemble(cache, sys)
    est = ens.estimate(pot, _r2, mode="graph")
    n = sys.n_atoms
    exact = 3 * n / (BETA * 2.3)
    assert float(est.value) == pytest.approx(exact, rel=0.03)
    (g,) = torch.autograd.grad(est.value.sum(), pot.log_k)
    # d<r2>/dlogk = -<r2> exactly
    assert float(g) == pytest.approx(-exact, rel=0.06)
    assert 0.5 < est.ess_fraction <= 1.0


def test_autograd_equals_covariance_formula_at_reference():
    sys, cache = _exact_harmonic_cache(1.5, n_frames=2000, seed=1)
    pot = HarmonicWell(k=1.5)
    ens = ReweightedEnsemble(cache, sys)
    est = ens.estimate(pot, _r2, mode="graph")
    (g_auto,) = torch.autograd.grad(est.value.sum(), pot.log_k)
    A = _r2(ens.pos, None)[:, 0]
    (g_cov,) = covariance_gradient(pot, ens, A, [pot.log_k])
    assert float(g_auto) == pytest.approx(float(g_cov), rel=1e-10)


def test_surrogate_backward_equals_graph_mode():
    sys, cache = _exact_harmonic_cache(1.0, n_frames=1000, seed=2)
    pot = HarmonicWell(k=1.2)
    ens = ReweightedEnsemble(cache, sys, chunk_size=64)
    est = ens.estimate(pot, _r2, mode="graph")
    loss = (est.value - 10.0) ** 2
    (g_graph,) = torch.autograd.grad(loss.sum(), pot.log_k)
    est2 = ens.estimate(pot, _r2, mode="surrogate")
    gv = 2 * (est2.value - 10.0)
    pot.zero_grad()
    ens.surrogate_backward(pot, _r2, est2, gv)
    assert float(pot.log_k.grad) == pytest.approx(float(g_graph), rel=1e-10)


class Zero(Potential):
    def energy(self, system, positions, cell=None):
        p = positions if positions.dim() == 3 else positions[None]
        e = 0.0 * p.sum(dim=(1, 2)).to(torch.float64)
        return e if positions.dim() == 3 else e[0]


def test_npt_pressure_reweighting_ideal_gas():
    # ln V-proposal barostat samples V^N exp(-beta p V): <V> = (N+1)/(beta p)
    n = 20
    ps, pt = 100.0 * units.BAR_TO_EV_A3, 130.0 * units.BAR_TO_EV_A3
    rng = np.random.default_rng(0)
    V = rng.gamma(n + 1, 1.0 / (BETA * ps), size=20000)
    L = V ** (1 / 3)
    cells = np.einsum("f,ij->fij", L, np.eye(3))
    sys = System("ig", (18,) * n, (39.948,) * n, task="toy", spin=0, pbc=(True, True, True),
                 positions=np.zeros((n, 3)), cell=np.eye(3))
    cache = SampleCache(sys.system_id, sys.atom_order_hash(), ThermoState(T, 100.0, "NPT"),
                        SamplingProtocol(), "exact", np.zeros((V.size, n, 3)), cells, np.zeros(V.size))
    ens = ReweightedEnsemble(cache, sys, target=ThermoState(T, 130.0, "NPT"), chunk_size=4096)
    est = ens.estimate(Zero(), lambda p, c: torch.linalg.det(c).abs()[:, None], mode="graph")
    assert float(est.value) == pytest.approx((n + 1) / (BETA * pt), rel=0.01)
