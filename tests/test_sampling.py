import numpy as np
import pytest
import torch

from umaexp import units
from umaexp.potential.base import Potential
from umaexp.potential.toy import HarmonicWell, LennardJones, lj_argon_system
from umaexp.sampling import (
    ForceEvaluator,
    MDState,
    SamplingProtocol,
    ThermoState,
    VelocityVerlet,
    run_sampling,
    statistical_inefficiency,
)
from umaexp.sampling.integrators import maxwell_boltzmann
from umaexp.systems import System


def test_statistical_inefficiency_matches_pymbar():
    pymbar = pytest.importorskip("pymbar")
    rng = np.random.default_rng(0)
    x = np.zeros(5000)
    for i in range(1, x.size):
        x[i] = 0.9 * x[i - 1] + rng.normal()
    ours = statistical_inefficiency(x)
    ref = pymbar.timeseries.statistical_inefficiency(x)
    assert ours == pytest.approx(ref, rel=1e-6)


def test_baoab_harmonic_equipartition():
    k = 1.0
    sys = System("hw", (18,) * 8, (39.948,) * 8, task="toy", spin=0, positions=np.zeros((8, 3)))
    proto = SamplingProtocol(dt_fs=5.0, friction_per_ps=5.0, n_equilibration=500, n_production=20000,
                             stride=5, seed=3)
    cache = run_sampling(HarmonicWell(k=k), sys, ThermoState(300.0), proto)
    r2 = (cache.positions**2).sum(axis=(1, 2)).mean()
    exact = 3 * 8 * units.KB_EV_K * 300.0 / k
    assert r2 == pytest.approx(exact, rel=0.05)
    assert cache.diagnostics["mean_temperature_k"] == pytest.approx(300.0, rel=0.05)


def test_nve_energy_conservation_lj():
    sys = lj_argon_system(n_side=3, density_g_cm3=1.2)
    pot = LennardJones(cutoff=float(np.linalg.det(sys.cell) ** (1 / 3)) / 2 - 0.01)
    ev = ForceEvaluator(pot, sys)
    m = sys.mass_tensor()
    v = maxwell_boltzmann(m, 90.0, torch.Generator().manual_seed(0))
    pos = sys.positions_tensor()
    e, f = ev(pos, sys.cell_tensor())
    s = MDState(pos, v, sys.cell_tensor(), f, e)
    integ = VelocityVerlet(ev, m, dt_fs=2.0)
    tot = []
    for _ in range(400):
        s = integ.step(s)
        tot.append(s.energy + 0.5 * float((m[:, None] * s.velocities**2).sum()))
    tot = np.array(tot)
    assert np.abs(tot - tot[0]).max() < 2e-3 * sys.n_atoms * units.KB_EV_K * 90.0


class Zero(Potential):
    def energy(self, system, positions, cell=None):
        p = positions if positions.dim() == 3 else positions[None]
        e = 0.0 * p.sum(dim=(1, 2)).to(torch.float64)
        return e if positions.dim() == 3 else e[0]


@pytest.mark.slow
def test_mc_barostat_ideal_gas_volume():
    n = 10
    sys = System("ig", (18,) * n, (39.948,) * n, task="toy", spin=0, pbc=(True, True, True),
                 molecule_index=tuple(range(n)), positions=np.random.default_rng(0).uniform(0, 20, (n, 3)),
                 cell=np.eye(3) * 20.0)
    p_bar = 200.0
    proto = SamplingProtocol(dt_fs=5.0, n_equilibration=2000, n_production=40000, stride=5,
                             barostat_every=1, barostat_max_dlnv=0.3, seed=1)
    cache = run_sampling(Zero(), sys, ThermoState(300.0, p_bar, "NPT"), proto)
    beta_p = units.beta(300.0) * p_bar * units.BAR_TO_EV_A3
    assert cache.volumes.mean() == pytest.approx((n + 1) / beta_p, rel=0.05)
