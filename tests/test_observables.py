import math

import numpy as np
import pytest
import torch

from umaexp import units
from umaexp.observables.binding import AssayModelRequired, BoreschRestraint, affinity_to_dg
from umaexp.observables.density import DensityObservable
from umaexp.observables.fluctuation import IsothermalCompressibility, harmonic_cv_quantum_correction
from umaexp.observables.nmr import KARPLUS, JCoupling
from umaexp.observables.response import FixedChargeDipole, ir_intensities
from umaexp.observables.structure import RadialDistribution, dihedral
from umaexp.observables.thermochemistry import check_balance, rrho
from umaexp.potential.toy import LennardJones, MorseDiatomic, lj_argon_system
from umaexp.reweighting import ReweightedEnsemble
from umaexp.sampling import SampleCache, SamplingProtocol, ThermoState, run_sampling
from umaexp.systems import System, isotopologue_masses


@pytest.fixture(scope="module")
def lj_npt():
    sys = lj_argon_system(n_side=3, density_g_cm3=1.3)
    L = float(np.linalg.det(sys.cell) ** (1 / 3))
    pot = LennardJones(cutoff=0.45 * L * 0.9)
    proto = SamplingProtocol(dt_fs=5.0, friction_per_ps=5.0, n_equilibration=1000, n_production=4000, stride=10,
                             barostat_every=5, barostat_max_dlnv=0.02, seed=0)
    cache = run_sampling(pot, sys, ThermoState(100.0, 400.0, "NPT"), proto)
    return sys, pot, cache


def test_density_graph_surrogate_and_fd(lj_npt):
    sys, pot, cache = lj_npt
    ens = ReweightedEnsemble(cache, sys, chunk_size=64)
    obs = DensityObservable(sys)
    pred = obs.predict(pot, ens, mode="graph")
    expected = sys.total_mass / cache.volumes.mean() * units.U_PER_A3_TO_G_CM3
    assert float(pred.value.detach()) == pytest.approx(expected, rel=1e-10)  # uniform weights at w0
    (g_graph,) = torch.autograd.grad(pred.value.sum(), pot.log_eps)
    pot.zero_grad()
    pred_s = obs.predict(pot, ens, mode="surrogate")
    pred_s.backward(torch.ones(1, dtype=torch.float64))
    assert float(pot.log_eps.grad) == pytest.approx(float(g_graph), rel=1e-8)
    h = 1e-4
    with torch.no_grad():
        pot.log_eps += h
        rp = float(obs.predict(pot, ens, "surrogate").value)
        pot.log_eps -= 2 * h
        rm = float(obs.predict(pot, ens, "surrogate").value)
        pot.log_eps += h
    assert float(g_graph) == pytest.approx((rp - rm) / (2 * h), rel=1e-5)


def test_compressibility_transform_on_synthetic_volumes():
    rng = np.random.default_rng(0)
    n, T = 30, 300.0
    beta_p = units.beta(T) * 1000 * units.BAR_TO_EV_A3
    V = rng.gamma(n + 1, 1 / beta_p, size=20000)
    sys = System("ig", (18,) * n, (39.948,) * n, task="toy", spin=0, pbc=(True,) * 3,
                 positions=np.zeros((n, 3)), cell=np.eye(3))
    cells = np.einsum("f,ij->fij", V ** (1 / 3), np.eye(3))
    cache = SampleCache("ig", sys.atom_order_hash(), ThermoState(T, 1000.0, "NPT"), SamplingProtocol(), "x",
                        np.zeros((V.size, n, 3)), cells, np.zeros(V.size))

    class Zero(torch.nn.Module):
        def energy(self, s, p, c=None):
            return torch.zeros(p.shape[0], dtype=torch.float64)

    pred = IsothermalCompressibility(T).predict(Zero(), ReweightedEnsemble(cache, sys, chunk_size=5000))
    # ideal-gas (N+1) model: Var V / (kT <V>) = 1/(beta p) * (N+1)/(N+1) / (kT ...) -> 1/p
    kappa_exact = 1.0 / (1000 * units.BAR_TO_EV_A3) / units.EV_A3_TO_GPA
    assert float(pred.value) == pytest.approx(kappa_exact, rel=0.03)


def test_rdf_ideal_gas_is_one():
    n = 300
    rng = np.random.default_rng(1)
    L = 20.0
    pos = rng.uniform(0, L, size=(20, n, 3))
    sys = System("ig", (18,) * n, (39.948,) * n, task="toy", spin=0, pbc=(True,) * 3,
                 positions=pos[0], cell=np.eye(3) * L)
    rdf = RadialDistribution(sys, 18, 18, r_max=8.0, n_bins=16)
    g = rdf.frames(torch.as_tensor(pos), torch.as_tensor(np.stack([np.eye(3) * L] * 20)))
    assert float(g.mean(0)[4:].mean()) == pytest.approx(1.0, abs=0.02)


def test_quantum_cv_correction_limits():
    nu = torch.tensor([3000.0, 50.0], dtype=torch.float64)
    corr = harmonic_cv_quantum_correction(nu, 298.15)
    # stiff mode: ~ -R ; soft mode: ~ 0
    assert float(corr) == pytest.approx(-units.GAS_CONSTANT_J_MOL_K, rel=0.01)


def test_rrho_water_standard_entropy():
    # experimental fundamentals and r_e geometry of H2O; NIST S deg(298.15 K, 1 bar) = 188.84 J/mol/K
    r, th = 0.9572, math.radians(104.52)
    pos = torch.tensor([[0, 0, 0], [r, 0, 0], [r * math.cos(th), r * math.sin(th), 0]], dtype=torch.float64)
    masses = torch.tensor(isotopologue_masses((8, 1, 1)), dtype=torch.float64)
    th_ = rrho(torch.tensor([1595.0, 3657.0, 3756.0], dtype=torch.float64), pos, masses, 298.15, 1.0,
               symmetry_number=2, qrrho_cutoff_cm=None)
    s = float(th_["s"]) * units.EV_J * units.AVOGADRO
    assert s == pytest.approx(188.84, abs=0.3)


def test_rrho_argon_translational_entropy():
    th_ = rrho(torch.zeros(0, dtype=torch.float64), torch.zeros(1, 3, dtype=torch.float64),
               torch.tensor([39.948], dtype=torch.float64), 298.15, 1.0)
    assert float(th_["s"]) * units.EV_J * units.AVOGADRO == pytest.approx(154.85, abs=0.05)


def test_reaction_balance_guard():
    h2 = System("H2", (1, 1), isotopologue_masses((1, 1)), positions=np.zeros((2, 3)))
    o2 = System("O2", (8, 8), isotopologue_masses((8, 8)), spin=3, positions=np.zeros((2, 3)))
    w = System("H2O", (8, 1, 1), isotopologue_masses((8, 1, 1)), positions=np.zeros((3, 3)))
    check_balance([(h2, -2), (o2, -1), (w, 2)])
    with pytest.raises(ValueError):
        check_balance([(h2, -1), (o2, -1), (w, 2)])


def test_karplus_beta_sheet_value():
    pos = torch.tensor([[[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.5], [1.0, -1.0, 1.5]]],
                       dtype=torch.float64)
    phi = math.degrees(float(dihedral(pos, 0, 1, 2, 3)[0]))
    j = float(JCoupling((0, 1, 2, 3), KARPLUS["3J_HN_HA_VuisterBax1993"]).frames(pos, None)[0, 0])
    p = KARPLUS["3J_HN_HA_VuisterBax1993"]
    c = math.cos(math.radians(phi - 60))
    assert j == pytest.approx(p.A * c * c + p.B * c + p.C, rel=1e-12)


def test_ir_intensity_diatomic_fixed_charges():
    q = 0.3
    pot = MorseDiatomic(4.0, 2.0, 1.1)
    sys = System("CO_toy", (6, 8), isotopologue_masses((6, 8)), task="toy", spin=0,
                 positions=np.array([[0, 0, 0], [0, 0, 1.1]]))
    pred = ir_intensities(pot, sys, torch.tensor([[0, 0, 0], [0, 0, 1.1]], dtype=torch.float64),
                          FixedChargeDipole([q, -q]))
    m1, m2 = sys.masses
    mu = m1 * m2 / (m1 + m2)
    assert float(pred.value[0]) == pytest.approx(q**2 / mu * 974.8801, rel=1e-6)


def test_binding_conversions_and_guards():
    dg = affinity_to_dg(1e-6, "Kd", 298.15)
    assert dg * units.EV_TO_KCAL_MOL == pytest.approx(-8.185, abs=0.005)
    with pytest.raises(AssayModelRequired):
        affinity_to_dg(1e-6, "IC50", 298.15)
    ok = affinity_to_dg(2e-6, "IC50", 298.15, assay={"mechanism": "competitive", "S_M": 1e-5, "Km_M": 1e-5})
    assert ok == pytest.approx(dg, rel=1e-12)
    kc = 10.0 * units.KCAL_MOL_TO_EV
    rest = BoreschRestraint(5.0, 1.5, 1.5, kc, kc, kc, kc, kc, kc)
    assert rest.dg_release(298.15) < 0
