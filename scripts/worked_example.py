"""Worked small-system examples with numbers (used in docs/DESIGN.md section 6).

A. Morse diatomic: every quantity analytic, so autograd can be checked exactly.
B. Toy bent triatomic: implicit vs fixed-geometry frequency gradients vs
   re-optimized finite differences.
C. DiffTRe identity on a harmonic well: autograd == -beta Cov(A, dU/dw).
D. Water on the UMA-S architecture (random weights unless --checkpoint):
   Hessian -> projected frequencies -> Gaussian NLL vs NIST fundamentals ->
   gradient into UMA's readout weights, with a directional FD check.

  python scripts/worked_example.py [--checkpoint uma-s-1p1]
"""

from __future__ import annotations

import argparse
import json
import logging
import math

import numpy as np
import torch

from umaexp import units
from umaexp.observables.vibrational import attach_implicit_geometry, harmonic_modes, optimize_geometry
from umaexp.potential.toy import HarmonicWell, MorseDiatomic, ValenceToy
from umaexp.reweighting import ReweightedEnsemble, covariance_gradient
from umaexp.sampling import SampleCache, SamplingProtocol, ThermoState
from umaexp.systems import System, isotopologue_masses

logging.getLogger("fairchem").setLevel(logging.ERROR)


def part_a():
    D, a, re = 4.6, 2.2, 0.97
    pot = MorseDiatomic(D, a, re)
    sys = System("OH_morse", (8, 1), isotopologue_masses((8, 1)), task="toy", spin=0,
                 positions=np.array([[0, 0, 0], [0, 0, 1.0]]))
    x, _ = optimize_geometry(pot, sys, fmax=1e-10)
    xw = attach_implicit_geometry(pot, sys, x)
    nu = harmonic_modes(pot, sys, xw).wavenumbers[0]
    gD, ga, gre = torch.autograd.grad(nu, [pot.D, pot.a, pot.re])
    m1, m2 = sys.masses
    mu = m1 * m2 / (m1 + m2)
    nu_exact = math.sqrt(2 * D * a * a / mu) * units.EIGVAL_TO_WAVENUMBER
    nu = nu.detach()
    return {"nu_cm": float(nu), "nu_exact_cm": nu_exact, "dnu_dD": float(gD), "exact_dnu_dD": nu_exact / (2 * D),
            "dnu_da": float(ga), "exact_dnu_da": nu_exact / a, "dnu_dre": float(gre), "exact_dnu_dre": 0.0}


def part_b():
    def at(theta0):
        r0 = 0.9572
        pot = ValenceToy([(0, 1), (0, 2)], [(1, 0, 2)], [50.0], [r0], [4.0], [theta0])
        pos = np.array([[0.0, 0, 0], [r0, 0, 0], [r0 * math.cos(1.8), r0 * math.sin(1.8), 0]])
        sys = System("tri", (8, 1, 1), isotopologue_masses((8, 1, 1)), task="toy", spin=0, positions=pos)
        x, _ = optimize_geometry(pot, sys, fmax=1e-10)
        return pot, sys, x

    th = math.radians(104.5)
    pot, sys, x = at(th)
    nu_imp = harmonic_modes(pot, sys, attach_implicit_geometry(pot, sys, x)).wavenumbers
    (g_imp,) = torch.autograd.grad(nu_imp.sum(), pot.theta0)
    nu_fix = harmonic_modes(pot, sys, x.detach()).wavenumbers
    (g_fix,) = torch.autograd.grad(nu_fix.sum(), pot.theta0)
    h = 1e-5
    vals = []
    for t in (th + h, th - h):
        p2, s2, x2 = at(t)
        vals.append(float(harmonic_modes(p2, s2, x2, create_graph=False).wavenumbers.sum()))
    return {"wavenumbers_cm": nu_imp.detach().tolist(), "d_sum_nu_d_theta0_implicit": float(g_imp),
            "d_sum_nu_d_theta0_fixed_geometry": float(g_fix),
            "d_sum_nu_d_theta0_reoptimized_fd": (vals[0] - vals[1]) / (2 * h)}


def part_c():
    T, k0, n = 300.0, 1.5, 4
    beta = units.beta(T)
    rng = np.random.default_rng(1)
    pos = rng.normal(0, 1 / math.sqrt(beta * k0), size=(4000, n, 3))
    sys = System("hw", (18,) * n, (39.948,) * n, task="toy", spin=0, positions=np.zeros((n, 3)))
    cache = SampleCache("hw", sys.atom_order_hash(), ThermoState(T), SamplingProtocol(), "exact", pos, None,
                        0.5 * k0 * (pos**2).sum(axis=(1, 2)))
    pot = HarmonicWell(k=k0)
    ens = ReweightedEnsemble(cache, sys)
    est = ens.estimate(pot, lambda p, c: (p**2).sum(dim=(1, 2))[:, None], "graph")
    (g_auto,) = torch.autograd.grad(est.value.sum(), pot.log_k)
    (g_cov,) = covariance_gradient(pot, ens, (ens.pos**2).sum(dim=(1, 2)), [pot.log_k])
    return {"<r^2>": float(est.value.detach()), "exact_<r^2>": 3 * n / (beta * k0),
            "d<r^2>/dlogk_autograd": float(g_auto), "d<r^2>/dlogk_covariance": float(g_cov),
            "exact": -3 * n / (beta * k0)}


def part_d(checkpoint):
    from ase.build import molecule

    from umaexp.potential import UMAPotential, select_parameters
    from umaexp.potential.derivatives import hessian
    from umaexp.validation.gradcheck import check_parameter_gradient

    if checkpoint:
        pot = UMAPotential.from_checkpoint(checkpoint, task="omol")
    else:
        pot = UMAPotential.random_architecture(element_references=torch.zeros(101, dtype=torch.float64))
    select_parameters(pot.model, preset="readout")
    a = molecule("H2O")
    sys = System.from_ase(a, system_id="H2O", charge=0, spin=1)
    x = torch.tensor(a.get_positions())
    nu_exp = torch.tensor([1595.0, 3657.0, 3756.0], dtype=torch.float64)  # NIST/Shimanouchi fundamentals
    sd = torch.full((3,), 0.5, dtype=torch.float64)

    # The model-discrepancy variance uses omega at the current weights with a
    # stop-gradient (VibrationalForwardModel): the fit must not be able to
    # lower its loss by moving frequencies to inflate its own error bars.
    # For the finite-difference check it is therefore held fixed.
    with torch.enable_grad():
        omega0 = torch.sort(harmonic_modes(pot, sys, x, create_graph=False).wavenumbers).values.detach()
    var = sd**2 + (0.015 * omega0) ** 2

    def loss():
        nm = harmonic_modes(pot, sys, x, create_graph=True)  # fixed geometry here, for a clean FD check
        omega = torch.sort(nm.wavenumbers).values
        pred = 0.955 * omega
        return 0.5 * (((pred - nu_exp) ** 2) / var).sum()

    L = loss()
    params = pot.trainable_parameters()
    grads = torch.autograd.grad(L, params, allow_unused=True)
    H = hessian(pot, sys, x, create_graph=False)
    chk = check_parameter_gradient(loss, params, n_probe=2, h=1e-6, rtol=1e-3)
    return {"random_weights": pot.provenance.random_init, "hessian_shape": list(H.shape),
            "wavenumbers_cm": harmonic_modes(pot, sys, x, create_graph=False).wavenumbers.tolist(),
            "loss": float(L), "grad_norms": {n: (None if g is None else float(g.norm()))
                                             for (n, _), g in zip(pot.trainable_named_parameters(), grads)},
            "directional_fd_check": str(chk)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default="runs/worked_example.json")
    args = ap.parse_args()
    out = {"A_morse": part_a(), "B_implicit_geometry": part_b(), "C_difftre_identity": part_c(),
           "D_water_uma": part_d(args.checkpoint)}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
