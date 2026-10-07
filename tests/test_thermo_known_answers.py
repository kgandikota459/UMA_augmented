"""Known-answer tests: fluctuation responses, virial pressure, vaporization and reaction thermochemistry."""

import math

import numpy as np
import pytest
import torch
from toy_helpers import PairMorse, SizeDependent, Zero, cache_from_frames

from umaexp import units
from umaexp.observables.density import VirialPressureObservable
from umaexp.observables.fluctuation import HeatCapacityP, ThermalExpansion
from umaexp.observables.thermochemistry import ReactionThermochemistry, SpeciesThermo, VaporizationEnthalpy
from umaexp.potential.derivatives import virial_pressure
from umaexp.potential.toy import LennardJones, lj_argon_system
from umaexp.reweighting import ReweightedEnsemble
from umaexp.sampling import ThermoState
from umaexp.systems import System, isotopologue_masses

T = 300.0


def _ideal_gas_npt(n=20, p_bar=500.0, F=40000, seed=0):
    """Exact NPT volumes of N point particles: V ~ Gamma(N+1, 1/(beta p))."""
    beta, p = units.beta(T), p_bar * units.BAR_TO_EV_A3
    V = np.random.default_rng(seed).gamma(n + 1, 1.0 / (beta * p), size=F)
    cells = np.einsum("f,ij->fij", V ** (1 / 3), np.eye(3))
    sys = System("ig", (18,) * n, (39.948,) * n, task="toy", spin=0, pbc=(True,) * 3,
                 molecule_index=tuple(range(n)), positions=np.zeros((n, 3)), cell=np.eye(3))
    cache = cache_from_frames(sys, np.zeros((F, n, 3)), cells, state=ThermoState(T, p_bar, "NPT"))
    return sys, ReweightedEnsemble(cache, sys, chunk_size=10000), p_bar


def test_ideal_gas_heat_capacity_and_thermal_expansion():
    sys, ens, p_bar = _ideal_gas_npt()
    n = sys.n_atoms
    cp = HeatCapacityP(sys, T, p_bar).predict(Zero(), ens)
    # (3N/2) k kinetic + (N+1) k from <d(pV)^2>/kT^2 (the +1 is the V^N measure), per molecule
    assert float(cp.value.detach()) == pytest.approx((2.5 + 1.0 / n) * units.GAS_CONSTANT_J_MOL_K, rel=0.03)
    alpha = ThermalExpansion(T, p_bar).predict(Zero(), ens)
    assert float(alpha.value.detach()) == pytest.approx(1.0 / T, rel=0.03)  # ideal gas: 1/T
    qc = HeatCapacityP(sys, T, p_bar, quantum_correction_j_mol_k=-5.0).predict(Zero(), ens)
    assert float(qc.value.detach() - cp.value.detach()) == pytest.approx(-5.0, abs=1e-9)


def test_virial_pressure_ideal_gas_and_independent_pairwise_lj_virial():
    L, n = 12.0, 10
    ig = System("ig", (18,) * n, (39.948,) * n, task="toy", spin=0, pbc=(True,) * 3,
                positions=np.random.default_rng(1).uniform(0, L, (n, 3)), cell=np.eye(3) * L)
    P = virial_pressure(Zero(), ig, ig.positions_tensor(), ig.cell_tensor(), T)
    assert float(P) == pytest.approx(n * units.KB_EV_K * T / L**3, rel=1e-12)

    lj = lj_argon_system(n_side=3, density_g_cm3=1.2)
    Lb = float(np.linalg.det(lj.cell)) ** (1 / 3)
    rc = 0.49 * Lb
    pot = LennardJones(cutoff=rc)
    gen = torch.Generator().manual_seed(0)
    pos = lj.positions_tensor() + 0.05 * torch.randn(lj.n_atoms, 3, dtype=torch.float64, generator=gen)
    eps, sig = float(pot.log_eps.exp()), float(pot.log_sigma.exp())

    def dphi(r):
        sr6 = (sig / r) ** 6
        return 4 * eps * (-12 * sr6 * sr6 + 6 * sr6) / r

    x = pos.numpy()
    W = 0.0
    for i in range(lj.n_atoms):
        for j in range(i + 1, lj.n_atoms):
            d = x[i] - x[j]
            d -= Lb * np.round(d / Lb)
            r = float(np.linalg.norm(d))
            if r < rc:
                W += r * (dphi(r) - dphi(rc))  # shifted-force derivative
    P_ref = lj.n_atoms * units.KB_EV_K * T / Lb**3 - W / (3 * Lb**3)
    assert float(virial_pressure(pot, lj, pos, lj.cell_tensor(), T)) == pytest.approx(P_ref, rel=1e-9)

    frames = torch.stack([pos, pos + 0.02 * torch.randn(pos.shape, dtype=torch.float64, generator=gen)])
    cells = np.stack([lj.cell] * 2)
    with torch.no_grad():
        u = pot.energy(lj, frames, torch.as_tensor(cells)).numpy()
    ens = ReweightedEnsemble(cache_from_frames(lj, frames.numpy(), cells, u=u), lj)
    pred = VirialPressureObservable(lj, T).predict(pot, ens)
    per_frame = [float(virial_pressure(pot, lj, f, lj.cell_tensor(), T)) for f in frames]
    assert float(pred.value.detach()) == pytest.approx(np.mean(per_frame) / units.BAR_TO_EV_A3, rel=1e-10)


@pytest.mark.parametrize("mode", ["graph", "surrogate"])
def test_vaporization_enthalpy_value_and_exact_gradient(mode):
    kT = units.KB_EV_K * T
    n, n_mol, k, c = 3, 4, 1.0, 0.002
    rng = np.random.default_rng(3)

    def ensemble(n_atoms, F):
        pos = rng.normal(0.0, math.sqrt(kT / k), size=(F, n_atoms, 3))
        u = 0.5 * k * (pos**2).sum(axis=(1, 2)) + c * n_atoms**2
        sys = System(f"s{n_atoms}", (6,) * n_atoms, isotopologue_masses((6,) * n_atoms), task="toy", spin=0,
                     positions=np.zeros((n_atoms, 3)))
        return ReweightedEnsemble(cache_from_frames(sys, pos, u=u), sys, chunk_size=5000)

    gas, liq = ensemble(n, 20000), ensemble(n * n_mol, 20000)
    pot = SizeDependent(k=k, c=c)
    pred = VaporizationEnthalpy(n_mol, T).predict(pot, gas, liq, mode=mode)
    # harmonic parts cancel per molecule in expectation: dH_vap = c n^2 (1 - N) + kT
    assert float(pred.value.detach()) == pytest.approx(c * n * n * (1 - n_mol) + kT, abs=2e-3)
    pot.zero_grad()
    pred.backward(torch.ones(1, dtype=torch.float64))
    assert float(pot.c.grad) == pytest.approx(n * n * (1 - n_mol), rel=1e-9)


def _h2_dissociation():
    D, a, re = 4.75, 1.94, 0.741
    pot = PairMorse(D, a, re)
    h2 = System("H2", (1, 1), isotopologue_masses((1, 1)), task="toy", spin=0,
                positions=np.array([[0, 0, 0], [0, 0, 0.75]]))
    h = System("H", (1,), isotopologue_masses((1,)), task="toy", spin=0, positions=np.zeros((1, 3)))
    species = [SpeciesThermo(h2, -1.0, symmetry_number=2, spin_multiplicity=1),
               SpeciesThermo(h, 2.0, spin_multiplicity=2)]
    return pot, h2, h, species, (D, a, re)


def test_reaction_thermochemistry_h2_dissociation_known_answers():
    pot, h2, h, species, (D, a, re) = _h2_dissociation()
    m1, m2 = h2.masses
    mu = m1 * m2 / (m1 + m2)
    nu = math.sqrt(2 * D * a * a / mu) * units.EIGVAL_TO_WAVENUMBER
    hnu = nu * units.WAVENUMBER_TO_EV
    zpe = 0.5 * hnu
    Tk = 298.15
    kT = units.KB_EV_K * Tk

    el = ReactionThermochemistry(species, Tk, quantity="electronic").predict(pot)
    assert float(el.value) == pytest.approx(D, rel=1e-8)
    (gD,) = torch.autograd.grad(el.value.sum(), pot.D)
    assert float(gD) == pytest.approx(1.0, rel=1e-8)  # envelope theorem

    z = ReactionThermochemistry(species, Tk, quantity="zpe_corrected").predict(pot)
    assert float(z.value) == pytest.approx(D - zpe, rel=1e-8)
    (gz,) = torch.autograd.grad(z.value.sum(), pot.D)
    assert float(gz) == pytest.approx(1.0 - zpe / (2 * D), rel=1e-6)  # zpe ~ sqrt(D)

    x = hnu / kT
    e_vib = hnu / math.expm1(x)
    hT = ReactionThermochemistry(species, Tk, quantity="enthalpy", qrrho_cutoff_cm=None).predict(pot)
    # 2 H(trans 5/2 kT) - H2(trans 5/2 kT + rot kT + ZPE + thermal vib) + D
    assert float(hT.value) == pytest.approx(D - zpe - e_vib + 1.5 * kT, rel=1e-8)

    # independent ideal-gas entropies (Sackur-Tetrode, rigid rotor, harmonic, electronic)
    kB, h_pl = units.BOLTZMANN_J_K, units.PLANCK_J_S
    p_pa = 1e5

    def s_trans(m_u):
        m = m_u * units.DALTON_KG
        return math.log((2 * math.pi * m * kB * Tk / h_pl**2) ** 1.5 * kB * Tk / p_pa) + 2.5

    inertia = mu * units.DALTON_KG * (re * 1e-10) ** 2
    s_rot = math.log(8 * math.pi**2 * inertia * kB * Tk / (2 * h_pl**2)) + 1.0
    s_vib = x / math.expm1(x) - math.log(-math.expm1(-x))
    dS = units.KB_EV_K * (2 * (s_trans(m1) + math.log(2)) - (s_trans(m1 + m2) + s_rot + s_vib))
    g = ReactionThermochemistry(species, Tk, quantity="gibbs", qrrho_cutoff_cm=None).predict(pot)
    assert float(g.value) == pytest.approx(float(hT.value) - Tk * dS, rel=1e-7)


def test_gibbs_requires_explicit_multiplicity_for_non_omol_species():
    pot, h2, h, _, _ = _h2_dissociation()
    species = [SpeciesThermo(h2, -1.0, symmetry_number=2), SpeciesThermo(h, 2.0)]  # multiplicities omitted
    with pytest.raises(ValueError, match="spin_multiplicity"):
        ReactionThermochemistry(species, quantity="gibbs").predict(pot)
    # quantities without electronic entropy still work
    assert math.isfinite(float(ReactionThermochemistry(species, quantity="enthalpy").predict(pot).value))
