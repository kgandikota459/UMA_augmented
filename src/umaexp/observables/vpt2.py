"""Second-order vibrational perturbation theory (VPT2/GVPT2) on a model surface.

Forward model of experiments/vib_v1/PROTOCOL.md 3.1: experimental fundamentals
are compared with omega_model + Delta_anh, where Delta_anh = nu_GVPT2 - omega
is computed ONCE from the unchanged base model and held fixed. This file
provides:

  * ``vpt2_core``: anharmonic constants chi and (G)VPT2 fundamentals from
    harmonic wavenumbers, cubic constants phi_ijk, semi-diagonal quartic
    constants phi_iijj (dimensionless normal coordinates, cm^-1), and optional
    Coriolis constants. Asymmetric-top (Cartesian) formulas of Mills (1972),
    written in the partial-fraction form; deperturbation and variational
    treatment of Fermi resonances following Martin et al., J. Chem. Phys. 103,
    2589 (1995) and Barone, J. Chem. Phys. 122, 014108 (2005). Degenerate modes
    are treated through their Cartesian components (validated against exact
    spectra of a degenerate 2-D oscillator in tests/test_vpt2.py).
  * ``normal_mode_force_field``: those constants from a Potential by central
    differences of analytic Hessians along rectilinear normal coordinates.
  * ``anharmonic_corrections``: Delta_anh per vibrational mode plus diagnostics.

Hamiltonian convention: H = sum_i omega_i/2 (p_i^2 + q_i^2)
  + 1/6 sum_ijk phi_ijk q_i q_j q_k + 1/24 sum_ijkl phi_ijkl q_i q_j q_k q_l.
Fundamental: nu_i = omega_i + 2 chi_ii + 1/2 sum_{j != i} chi_ij.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .. import units

EV_TO_CM = 8065.543937  # cm^-1 per eV (CODATA 2018)
ROT_CONST = 16.857629206  # h / (8 pi^2 c) in amu A^2 cm^-1


@dataclass
class VPT2Result:
    omega: np.ndarray
    chi: np.ndarray  # deperturbed anharmonic constants (cm^-1)
    nu_deperturbed: np.ndarray
    nu: np.ndarray  # GVPT2 fundamentals
    weight: np.ndarray  # |<1_k|state>|^2 of the state taken as fundamental k (1 if not in resonance)
    resonances: list[tuple] = field(default_factory=list)  # ("I", i, i, k) or ("II", i, j, k), with Delta
    blocks: dict = field(default_factory=dict)  # k -> (eigenvalues, |<1_k|state>|^2) of its resonance block


def _detect_resonances(omega, phi3, max_gap=200.0, threshold=1.0):
    n = len(omega)
    res1, res2 = set(), set()
    for k in range(n):
        for i in range(n):
            if i == k:
                continue
            d = 2 * omega[i] - omega[k]
            if abs(d) < max_gap and phi3[i, i, k] ** 4 > threshold * 256 * abs(d) ** 3:  # no division at d = 0
                res1.add((i, k))
            for j in range(i + 1, n):
                if j == k:
                    continue
                d = omega[i] + omega[j] - omega[k]
                if abs(d) < max_gap and phi3[i, j, k] ** 4 > threshold * 64 * abs(d) ** 3:
                    res2.add((i, j, k))
    return res1, res2


def chi_matrix(omega, phi3, phi4, zeta=None, B=None, res1=frozenset(), res2=frozenset()) -> np.ndarray:
    """Anharmonic constants with the listed Fermi-resonant terms removed (deperturbed)."""
    w = np.asarray(omega, float)
    n = len(w)
    chi = np.zeros((n, n))
    for i in range(n):
        s = phi4[i, i] / 16.0
        for k in range(n):
            f = 2.0 / w[k] + 0.5 / (2 * w[i] + w[k])
            if (i, k) not in res1:
                f -= 0.5 / (2 * w[i] - w[k])
            s -= phi3[i, i, k] ** 2 * f / 16.0
        chi[i, i] = s
    for i in range(n):
        for j in range(i + 1, n):
            s = phi4[i, j] / 4.0
            for k in range(n):
                s -= phi3[i, i, k] * phi3[j, j, k] / (4.0 * w[k])
                br = 1.0 / (w[i] + w[j] + w[k])
                # -1/(wi+wj-wk): singular when w_k ~ w_i + w_j
                if not _is_res(res1, res2, i, j, k):
                    br -= 1.0 / (w[i] + w[j] - w[k])
                # +1/(wi+wk-wj): singular when w_j ~ w_i + w_k (type I if k == i)
                if not _is_res(res1, res2, i, k, j):
                    br += 1.0 / (w[i] - w[j] + w[k])
                # +1/(wj+wk-wi): singular when w_i ~ w_j + w_k (type I if k == j)
                if not _is_res(res1, res2, j, k, i):
                    br -= 1.0 / (w[i] - w[j] - w[k])
                s -= phi3[i, j, k] ** 2 * br / 8.0
            if zeta is not None and B is not None:
                s += float((B * zeta[:, i, j] ** 2).sum()) * (w[i] / w[j] + w[j] / w[i])
            chi[i, j] = chi[j, i] = s
    return chi


def _is_res(res1, res2, a, b, c) -> bool:
    """Is the denominator (w_a + w_b - w_c) a listed Fermi resonance (w_c ~ w_a + w_b)?"""
    if a == b:
        return (a, c) in res1
    lo, hi = min(a, b), max(a, b)
    return (lo, hi, c) in res2


def _level(chi, omega, v):
    v = np.asarray(v, float)
    e = float(omega @ v)
    n = len(omega)
    for i in range(n):
        e += chi[i, i] * (v[i] ** 2 + v[i])
        for j in range(i + 1, n):
            e += chi[i, j] * (v[i] * v[j] + 0.5 * v[i] + 0.5 * v[j])
    return e


def vpt2_core(omega, phi3, phi4, zeta=None, B=None, max_gap=200.0, threshold=1.0, active=None) -> VPT2Result:
    """active: boolean mask; inactive (decoupled) modes are removed from every sum and get no correction."""
    if active is not None:
        act = np.flatnonzero(np.asarray(active, bool))
        sub = vpt2_core(np.asarray(omega, float)[act], phi3[np.ix_(act, act, act)], phi4[np.ix_(act, act)],
                        None if zeta is None else zeta[:, act][:, :, act], B, max_gap, threshold)
        n_all = len(omega)
        nu = np.asarray(omega, float).copy()
        nu_d = nu.copy()
        weight = np.ones(n_all)
        chi = np.zeros((n_all, n_all))
        nu[act], nu_d[act], weight[act] = sub.nu, sub.nu_deperturbed, sub.weight
        chi[np.ix_(act, act)] = sub.chi
        remap = {k: int(act[k]) for k in range(len(act))}
        res = [(r[0], *[remap[i] for i in r[1:4]], r[4]) for r in sub.resonances]
        blocks = {remap[k]: v for k, v in sub.blocks.items()}
        return VPT2Result(np.asarray(omega, float), chi, nu_d, nu, weight, res, blocks)
    w = np.asarray(omega, float)
    n = len(w)
    res1, res2 = _detect_resonances(w, phi3, max_gap, threshold)
    chi = chi_matrix(w, phi3, phi4, zeta, B, res1, res2)
    nu_d = np.array([w[i] + 2 * chi[i, i] + 0.5 * (chi[i].sum() - chi[i, i]) for i in range(n)])
    nu = nu_d.copy()
    weight = np.ones(n)
    blocks = {}
    reslist = [("I", i, i, k, 2 * w[i] - w[k]) for i, k in sorted(res1)] + \
              [("II", i, j, k, w[i] + w[j] - w[k]) for i, j, k in sorted(res2)]
    for k in range(n):
        partners = []
        for i, kk in res1:
            if kk == k:
                v = np.zeros(n)
                v[i] = 2
                partners.append((v, phi3[i, i, k] / 4.0))
        for i, j, kk in res2:
            if kk == k:
                v = np.zeros(n)
                v[i] = v[j] = 1
                partners.append((v, phi3[i, j, k] / (2.0 * np.sqrt(2.0))))
        if not partners:
            continue
        m = len(partners) + 1
        Hm = np.zeros((m, m))
        Hm[0, 0] = nu_d[k]
        for a, (v, cpl) in enumerate(partners, start=1):
            Hm[a, a] = _level(chi, w, v)
            Hm[0, a] = Hm[a, 0] = cpl
        ev, U = np.linalg.eigh(Hm)
        t = int(np.argmax(U[0] ** 2))
        nu[k], weight[k] = ev[t], U[0, t] ** 2
        blocks[k] = (ev, U[0] ** 2)
    return VPT2Result(w, chi, nu_d, nu, weight, reslist, blocks)


# ---------------------------------------------------------------------------
# force field of a Potential in dimensionless normal coordinates
# ---------------------------------------------------------------------------


@dataclass
class NormalModeForceField:
    omega: np.ndarray  # cm^-1
    phi3: np.ndarray  # (n, n, n) cm^-1
    phi4: np.ndarray  # (n, n) semi-diagonal phi_iijj, cm^-1
    zeta: np.ndarray  # (3, n, n) Coriolis constants in the principal-axis frame
    B: np.ndarray  # (3,) rotational constants, cm^-1
    modes: np.ndarray  # (3N, n) mass-weighted eigenvectors
    harmonic_check: float  # max |phi_ii - omega_i| / omega_i (unit consistency)


def normal_mode_force_field(potential, system, x_star: torch.Tensor, step: float = 0.025,
                            nm=None) -> NormalModeForceField:
    """Cubic and semi-diagonal quartic constants by central differences of analytic Hessians.

    step: displacement in dimensionless normal coordinates (q = sqrt(gamma) Q).
    nm: NormalModes at x_star to use (so mode indices agree with an assignment
        made on the same modes); computed here if None.
    """
    from ..potential.derivatives import hessian
    from .vibrational import modes_from_hessian

    x0 = x_star.detach().to(torch.float64)
    masses = system.mass_tensor(dtype=torch.float64)
    H0 = nm.hessian.detach() if nm is not None else hessian(potential, system, x0, create_graph=False).detach()
    if nm is None:
        nm = modes_from_hessian(H0, x0, masses)
    if nm.linear:
        raise ValueError("VPT2 forward model is implemented for nonlinear molecules only")
    omega = nm.wavenumbers.detach().numpy().astype(float)
    if (omega <= 0).any():
        raise ValueError(f"non-positive harmonic wavenumbers {omega[omega <= 0]}: not a minimum")
    L = nm.modes.numpy()
    mih = 1.0 / np.sqrt(masses.repeat_interleave(3).numpy())
    gamma = omega * EV_TO_CM / units.EIGVAL_TO_WAVENUMBER**2  # q = sqrt(gamma) Q, Q in sqrt(amu) A

    def hq(H):
        return L.T @ (mih[:, None] * H * mih[None, :]) @ L

    HQ0 = hq(H0.numpy())
    n = len(omega)
    dQ = step / np.sqrt(gamma)
    d3 = np.zeros((n, n, n))
    d4 = np.zeros((n, n))
    for k in range(n):
        disp = torch.tensor((mih * L[:, k] * dQ[k]).reshape(-1, 3))
        Hp = hq(hessian(potential, system, x0 + disp, create_graph=False).detach().numpy())
        Hm = hq(hessian(potential, system, x0 - disp, create_graph=False).detach().numpy())
        d3[:, :, k] = (Hp - Hm) / (2 * dQ[k])
        d4[:, k] = np.diag(Hp + Hm - 2 * HQ0) / dQ[k] ** 2
    d3 = sum(d3.transpose(p) for p in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]) / 6.0
    s = 1.0 / np.sqrt(gamma)
    phi3 = d3 * s[:, None, None] * s[None, :, None] * s[None, None, :] * EV_TO_CM
    phi4 = d4 * (s**2)[:, None] * (s**2)[None, :] * EV_TO_CM
    phi4 = 0.5 * (phi4 + phi4.T)  # d2 H_ii / dQ_k2 and d2 H_kk / dQ_i2 are the same constant
    harm = np.diag(HQ0) / gamma * EV_TO_CM
    # Coriolis constants and rotational constants in the principal-axis frame
    m = masses.numpy()
    xc = x0.numpy() - (m[:, None] * x0.numpy()).sum(0) / m.sum()
    inertia = np.eye(3) * (m[:, None] * xc * xc).sum() - (m[:, None] * xc).T @ xc
    I, U = np.linalg.eigh(inertia)
    Lp = np.einsum("ab,nbi->nai", U.T, L.reshape(-1, 3, n))  # (N, 3, n) in the principal frame
    zeta = np.zeros((3, n, n))
    for a, (b, c) in enumerate([(1, 2), (2, 0), (0, 1)]):
        zeta[a] = Lp[:, b, :].T @ Lp[:, c, :] - Lp[:, c, :].T @ Lp[:, b, :]
    return NormalModeForceField(omega, phi3, phi4, zeta, ROT_CONST / I, L,
                                float(np.max(np.abs(harm - omega) / omega)))


@dataclass
class AnharmonicCorrections:
    omega: np.ndarray
    nu: np.ndarray
    delta: np.ndarray  # nu - omega, per vibrational mode (degenerate components averaged)
    weight: np.ndarray
    resonances: list
    modes: np.ndarray
    harmonic_check: float
    step: float


def anharmonic_corrections(potential, system, x_star: torch.Tensor, step: float = 0.025,
                           degenerate_tol: float = 0.5, nm=None, decouple=None, coriolis: bool = True,
                           ff: NormalModeForceField | None = None,
                           groups: list[list[int]] | None = None) -> AnharmonicCorrections:
    """groups: symmetry mode groups (irrep-pure degenerate sets, observables/symmetry.py). Components are
    averaged only within a group; without groups, within `degenerate_tol` (which can merge different
    species that are accidentally near-degenerate, critic report #3 P1)."""
    """decouple: boolean mask of modes removed from the VPT2 sums (large-amplitude modes; PROTOCOL 3.1).
    coriolis: include the Mills asymmetric-top Coriolis terms in chi_ij (primary) or not (sensitivity)."""
    ff = ff or normal_mode_force_field(potential, system, x_star, step, nm=nm)
    active = None if decouple is None else ~np.asarray(decouple, bool)
    r = vpt2_core(ff.omega, ff.phi3, ff.phi4, ff.zeta if coriolis else None, ff.B if coriolis else None,
                  active=active)
    delta = r.nu - ff.omega
    weight = r.weight.copy()
    if groups is not None:
        grp = [list(g) for g in groups]
    else:
        order = np.argsort(ff.omega)  # average Cartesian components of degenerate modes
        grp = []
        for i in order:
            if grp and abs(ff.omega[i] - ff.omega[grp[-1][-1]]) <= degenerate_tol:
                grp[-1].append(int(i))
            else:
                grp.append([int(i)])
    for g in grp:
        delta[g] = delta[g].mean()
        weight[g] = weight[g].min()
    return AnharmonicCorrections(ff.omega, ff.omega + delta, delta, weight, r.resonances, ff.modes,
                                 ff.harmonic_check, step)
