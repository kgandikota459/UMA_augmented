"""Known-answer tests for VPT2/GVPT2 against exact spectra of the same Hamiltonians.

Exact levels: H = sum_i w_i (n_i + 1/2) + 1/6 sum phi_ijk q_i q_j q_k
+ 1/24 sum_i phi_iiii q_i^4 + 1/4 sum_{i<j} phi_iijj q_i^2 q_j^2, diagonalized in a
direct-product harmonic-oscillator basis large enough for the low states.
"""

import itertools

import numpy as np
import pytest
import torch

from umaexp.observables.vpt2 import _detect_resonances, chi_matrix, vpt2_core


def exact(omega, phi3, phi4, nb):
    n = len(omega)
    a = np.diag(np.sqrt(np.arange(1, nb)), 1)
    q1 = (a + a.T) / np.sqrt(2.0)
    eye = np.eye(nb)

    def emb(m, i):
        out = np.ones((1, 1))
        for t in range(n):
            out = np.kron(out, m if t == i else eye)
        return out

    Q = [emb(q1, i) for i in range(n)]
    H = sum(omega[i] * emb(np.diag(np.arange(nb) + 0.5), i) for i in range(n))
    for i, j, k in itertools.product(range(n), repeat=3):
        if phi3[i, j, k]:
            H = H + phi3[i, j, k] / 6.0 * Q[i] @ Q[j] @ Q[k]
    for i in range(n):
        H = H + phi4[i, i] / 24.0 * np.linalg.matrix_power(Q[i], 4)
        for j in range(i + 1, n):
            H = H + phi4[i, j] / 4.0 * Q[i] @ Q[i] @ Q[j] @ Q[j]
    E, V = np.linalg.eigh(H)

    def idx(v):
        return int(np.ravel_multi_index(tuple(v), (nb,) * n))

    g = int(np.argmax(V[idx([0] * n)] ** 2))
    fund = []
    for i in range(n):
        v = [0] * n
        v[i] = 1
        fund.append(E[int(np.argmax(V[idx(v)] ** 2))] - E[g])
    return np.array(fund)


def sym3(entries, n):
    t = np.zeros((n, n, n))
    for (i, j, k), v in entries.items():
        for p in set(itertools.permutations((i, j, k))):
            t[p] = v
    return t


def test_partial_fraction_form_equals_mills_formula():
    rng = np.random.default_rng(1)
    w = rng.uniform(800, 3500, 4)
    phi3 = sym3({(i, j, k): rng.normal(0, 50) for i, j, k in itertools.combinations_with_replacement(range(4), 3)}, 4)
    phi4 = np.zeros((4, 4))
    chi = chi_matrix(w, phi3, phi4)
    for i, j in [(0, 1), (1, 3), (0, 2)]:
        mills = -sum(phi3[i, i, k] * phi3[j, j, k] / w[k] for k in range(4)) / 4.0
        for k in range(4):
            D = (w[i] + w[j] + w[k]) * (w[i] - w[j] - w[k]) * (-w[i] + w[j] - w[k]) * (-w[i] - w[j] + w[k])
            mills += 2 * w[k] * (w[i] ** 2 + w[j] ** 2 - w[k] ** 2) * phi3[i, j, k] ** 2 / D / 4.0
        assert chi[i, j] == pytest.approx(mills, rel=1e-10)
    for i in range(4):
        mills = -sum(phi3[i, i, k] ** 2 * (8 * w[i] ** 2 - 3 * w[k] ** 2) / (w[k] * (4 * w[i] ** 2 - w[k] ** 2))
                     for k in range(4)) / 16.0
        assert chi[i, i] == pytest.approx(mills, rel=1e-10)


def test_one_dimensional_anharmonic_oscillator():
    for lam, tol in [(1.0, 0.1), (0.5, 0.01)]:
        w = np.array([1000.0])
        phi3 = np.full((1, 1, 1), -120.0 * lam)
        phi4 = np.full((1, 1), 60.0 * lam**2)
        r = vpt2_core(w, phi3, phi4)
        assert r.nu[0] == pytest.approx(exact(w, phi3, phi4, 60)[0], abs=tol)


def test_coupled_two_mode_oscillator_and_error_scaling():
    errs = []
    for lam in (1.0, 0.5):
        w = np.array([1200.0, 3100.0])
        phi3 = sym3({(0, 0, 0): -60 * lam, (1, 1, 1): -150 * lam, (0, 0, 1): 40 * lam, (0, 1, 1): -50 * lam}, 2)
        phi4 = np.array([[20.0, -15.0], [-15.0, 60.0]]) * lam**2
        r = vpt2_core(w, phi3, phi4)
        assert not r.resonances
        errs.append(np.abs(r.nu - exact(w, phi3, phi4, 24)).max())
    assert errs[0] < 0.5 and errs[0] / errs[1] > 8  # residual is higher order (~lambda^4)


def test_morse_oscillator_is_reproduced_exactly_at_realistic_anharmonicity():
    # Morse Taylor constants in dimensionless coordinates: phi3 = -3 sqrt(2 we wexe), phi4 = 14 wexe;
    # VPT2 is exact for Morse: nu = we - 2 wexe (here a CH-stretch-sized shift of -130 cm-1)
    we, wexe = 3000.0, 65.0
    r = vpt2_core(np.array([we]), np.full((1, 1, 1), -3 * np.sqrt(2 * we * wexe)), np.full((1, 1), 14 * wexe))
    assert r.nu[0] == pytest.approx(we - 2 * wexe, abs=1e-9)


@pytest.mark.parametrize("lam,mu,nb,tol", [(20.0, 6.0, 26, 0.05), (150.0, 0.0, 40, 0.5)])
def test_degenerate_henon_heiles_pair_through_cartesian_components(lam, mu, nb, tol):
    w = np.array([1000.0, 1000.0]) if lam < 100 else np.array([1500.0, 1500.0])
    phi3 = sym3({(0, 0, 1): 2 * lam, (1, 1, 1): -2 * lam}, 2)  # V3 = lam (q1^2 q2 - q2^3 / 3)
    phi4 = np.array([[mu, mu / 3], [mu / 3, mu]])  # isotropic mu/24 (q1^2 + q2^2)^2
    r = vpt2_core(w, phi3, phi4)
    ex = exact(w, phi3, phi4, nb)
    assert r.nu[0] == pytest.approx(r.nu[1], abs=1e-9)  # components stay degenerate
    assert r.nu[0] == pytest.approx(ex[0], abs=tol) and ex[0] == pytest.approx(ex[1], abs=1e-6)


def test_strong_fermi_dyad_energies_match_and_label_ambiguity_is_flagged():
    # CH stretch ~ 2 x bend with a realistic coupling: GVPT2 must reproduce BOTH dyad states; which one is
    # "the fundamental" is ill-defined near 50/50 mixing, so the protocol excludes weights < 0.7
    w = np.array([1460.0, 2950.0])
    phi3 = sym3({(0, 0, 1): 140.0, (1, 1, 1): -1400.0, (0, 0, 0): -100.0}, 2)
    phi4 = np.array([[30.0, -60.0], [-60.0, 800.0]])
    r = vpt2_core(w, phi3, phi4)
    ev, wt = r.blocks[1]
    nb = 36
    a = np.diag(np.sqrt(np.arange(1, nb)), 1)
    q = (a + a.T) / np.sqrt(2.0)
    eye = np.eye(nb)
    Q = [np.kron(q, eye), np.kron(eye, q)]
    H = w[0] * np.kron(np.diag(np.arange(nb) + 0.5), eye) + w[1] * np.kron(eye, np.diag(np.arange(nb) + 0.5))
    for i, j, k in itertools.product(range(2), repeat=3):
        if phi3[i, j, k]:
            H = H + phi3[i, j, k] / 6 * Q[i] @ Q[j] @ Q[k]
    H = H + phi4[0, 0] / 24 * np.linalg.matrix_power(Q[0], 4) + phi4[1, 1] / 24 * np.linalg.matrix_power(Q[1], 4) \
        + phi4[0, 1] / 4 * Q[0] @ Q[0] @ Q[1] @ Q[1]
    E, V = np.linalg.eigh(H)
    g = int(np.argmax(V[0] ** 2))
    dyad = sorted(np.argsort(-(V[1] ** 2 + V[2 * nb] ** 2))[:2])  # basis |0,1> = index 1, |2,0> = index 2 nb
    assert np.allclose(sorted(ev), sorted(E[dyad] - E[g]), atol=12.0)
    assert r.weight[1] < 0.7  # -> excluded by PROTOCOL 3.2


def test_fermi_type_one_resonance_is_deperturbed_and_diagonalized():
    w = np.array([1000.0, 2010.0])
    phi3 = sym3({(0, 0, 1): 60.0, (0, 0, 0): -40.0, (1, 1, 1): -90.0}, 2)
    phi4 = np.array([[10.0, -5.0], [-5.0, 30.0]])
    res1, _ = _detect_resonances(w, phi3)
    assert res1 == {(0, 1)}
    r = vpt2_core(w, phi3, phi4)
    ex = exact(w, phi3, phi4, 30)
    assert r.weight[1] < 0.95  # a genuinely mixed state
    assert r.nu[1] == pytest.approx(ex[1], abs=1.5)
    assert r.nu[0] == pytest.approx(ex[0], abs=1.5)
    plain = vpt2_core(w, phi3, phi4, threshold=np.inf)  # no resonance treatment
    assert abs(plain.nu[1] - ex[1]) > 3 * abs(r.nu[1] - ex[1])


def test_fermi_type_two_resonance():
    w = np.array([1000.0, 1500.0, 2512.0])
    phi3 = sym3({(0, 1, 2): 50.0, (0, 0, 0): -30.0, (1, 1, 1): -40.0, (2, 2, 2): -70.0}, 3)
    phi4 = np.diag([8.0, 12.0, 25.0])
    _, res2 = _detect_resonances(w, phi3)
    assert res2 == {(0, 1, 2)}
    r = vpt2_core(w, phi3, phi4)
    ex = exact(w, phi3, phi4, 11)
    assert r.nu[2] == pytest.approx(ex[2], abs=1.5)
    assert r.weight[2] < 0.95


def test_force_field_by_finite_differences_matches_autograd_third_derivatives():
    from toy_helpers import PairMorse

    from umaexp.observables.vibrational import modes_from_hessian, optimize_geometry
    from umaexp.observables.vpt2 import EV_TO_CM, normal_mode_force_field
    from umaexp.potential.derivatives import hessian
    from umaexp.systems import System
    from umaexp import units

    # Morse triangle (D3h minimum, e' pair) with unequal masses -> C2v-like mode structure
    zs = [8, 1, 1]
    sysm = System(system_id="tri", atomic_numbers=zs, masses=[16.0, 1.008, 2.014], charge=0, spin=1)
    pot = PairMorse(2.0, 1.6, 1.1)
    x0 = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.55, 0.95, 0.0]], dtype=torch.float64)
    x, info = optimize_geometry(pot, sysm, x0, fmax=1e-9)
    ff = normal_mode_force_field(pot, sysm, x, step=0.02)
    assert ff.harmonic_check < 1e-8
    # exact cubic constants via autograd along the same normal coordinates
    masses = sysm.mass_tensor(dtype=torch.float64)
    nm = modes_from_hessian(hessian(pot, sysm, x, create_graph=False), x, masses)
    L = torch.tensor(ff.modes)
    mih = masses.repeat_interleave(3).rsqrt()
    omega = torch.tensor(ff.omega)
    gamma = omega * EV_TO_CM / units.EIGVAL_TO_WAVENUMBER**2
    D = (mih[:, None] * L) / gamma.sqrt()[None, :]  # Cartesian displacement per unit q_i
    xx = x.clone().requires_grad_(True)
    e = pot.energy(sysm, xx)
    g = torch.autograd.grad(e, xx, create_graph=True)[0].reshape(-1)
    n = L.shape[1]
    phi3 = np.zeros((n, n, n))
    for i in range(n):
        gi = g @ D[:, i]
        hi = torch.autograd.grad(gi, xx, create_graph=True)[0].reshape(-1)
        for j in range(n):
            hij = hi @ D[:, j]
            tij = torch.autograd.grad(hij, xx, retain_graph=True)[0].reshape(-1)
            phi3[i, j] = (tij @ D).detach().numpy() * EV_TO_CM
    assert np.abs(ff.phi3 - phi3).max() < 1e-3 * np.abs(phi3).max()
    # step convergence of the resulting anharmonic constants
    ff2 = normal_mode_force_field(pot, sysm, x, step=0.04)
    r1 = vpt2_core(ff.omega, ff.phi3, ff.phi4)
    r2 = vpt2_core(ff2.omega, ff2.phi3, ff2.phi4)
    assert np.abs(r1.nu - r2.nu).max() < 0.05


def test_triply_degenerate_f_type_oscillator():
    # Td-invariant cubic coupling of an F2 triple, V3 = kappa q1 q2 q3, plus an isotropic quartic
    kappa, mu = 60.0, 9.0
    w = np.array([1300.0, 1300.0, 1300.0])
    phi3 = sym3({(0, 1, 2): kappa}, 3)
    phi4 = np.full((3, 3), mu / 3) + np.eye(3) * (mu - mu / 3)
    r = vpt2_core(w, phi3, phi4)
    ex = exact(w, phi3, phi4, 12)
    assert np.ptp(r.nu) < 1e-9 and np.ptp(ex) < 1e-6  # the triple stays degenerate
    assert r.nu[0] == pytest.approx(ex[0], abs=0.2)


def test_decoupled_modes_get_no_correction_and_leave_the_others_consistent():
    w = np.array([300.0, 1500.0, 3000.0])
    phi3 = sym3({(0, 0, 1): 40.0, (0, 0, 2): 30.0, (1, 1, 2): 60.0, (2, 2, 2): -900.0}, 3)
    phi4 = np.diag([50.0, 20.0, 400.0])
    full = vpt2_core(w, phi3, phi4)
    dec = vpt2_core(w, phi3, phi4, active=[False, True, True])
    sub = vpt2_core(w[1:], phi3[1:, 1:, 1:], phi4[1:, 1:])
    assert dec.nu[0] == w[0] and np.allclose(dec.nu[1:], sub.nu)
    assert not np.allclose(dec.nu[1:], full.nu[1:])  # the soft mode's couplings did matter
