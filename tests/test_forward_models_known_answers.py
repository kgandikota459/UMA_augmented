"""Known-answer tests: scattering, NMR and structural forward models."""

import math

import numpy as np
import pytest
import torch
from toy_helpers import Zero, cache_from_frames, toy_system

from umaexp.observables.nmr import KARPLUS, ChemicalShift, JCoupling, NOEDistance
from umaexp.observables.scattering import (
    CROMER_MANN,
    DebyeSAXS,
    profile_scale_offset,
    structure_factor_from_rdf,
    xray_form_factors,
)
from umaexp.observables.structure import DistanceAverage, StatePopulation
from umaexp.reweighting import ReweightedEnsemble
from umaexp.systems import System


def _ens(system, frames, cells=None):
    return ReweightedEnsemble(cache_from_frames(system, frames, cells), system)


def _dihedral_frames(phis_deg):
    out = []
    for phi in np.deg2rad(phis_deg):
        out.append([[1.0, 0, 0], [0, 0, 0], [0, 0, 1.5], [math.cos(phi), math.sin(phi), 1.5]])
    return np.array(out)


# ---------------------------------------------------------------- scattering

def test_form_factors_at_q0_equal_atomic_number():
    zs = list(CROMER_MANN)
    f0 = xray_form_factors(zs, torch.zeros(1, dtype=torch.float64))[0]
    # the published Cromer-Mann fits reproduce Z only to ~0.1% (N sums to 6.9946)
    assert torch.allclose(f0, torch.tensor(zs, dtype=torch.float64), rtol=1e-3)


def test_debye_two_atoms_closed_form_forward_limit_and_ensemble_average():
    q = torch.linspace(0.0, 2.0, 21, dtype=torch.float64)
    d1, d2 = 1.2, 1.6
    frames = np.array([[[0, 0, 0], [0, 0, d1]], [[0, 0, 0], [0, 0, d2]]], dtype=float)
    sys = toy_system((6, 8), frames[0])
    saxs = DebyeSAXS(sys, q)
    intensity = saxs.frames(torch.as_tensor(frames), None)
    f = xray_form_factors([6, 8], q)
    fc, fo = f[:, 0], f[:, 1]

    def closed(d):
        qd = q * d
        s = torch.where(qd > 0, torch.sin(qd) / qd.clamp_min(1e-300), torch.ones_like(qd))
        return fc**2 + fo**2 + 2 * fc * fo * s

    assert torch.allclose(intensity[0], closed(d1), rtol=1e-12)
    assert torch.allclose(intensity[1], closed(d2), rtol=1e-12)
    assert float(intensity[0, 0]) == pytest.approx(float((fc[0] + fo[0]) ** 2), rel=1e-12)  # I(0) = (sum f)^2
    pred = saxs.predict(Zero(), _ens(sys, frames))
    assert torch.allclose(pred.value.detach(), 0.5 * (closed(d1) + closed(d2)), rtol=1e-12)


def test_profile_scale_offset_recovers_exact_parameters():
    model = torch.linspace(1.0, 5.0, 30, dtype=torch.float64)
    sd = torch.full_like(model, 0.1)
    fitted, c, b = profile_scale_offset(model, 2.5 * model + 0.3, sd)
    assert float(c) == pytest.approx(2.5, rel=1e-12) and float(b) == pytest.approx(0.3, abs=1e-12)
    _, c2, b2 = profile_scale_offset(model, 1.7 * model, sd, fit_offset=False)
    assert float(c2) == pytest.approx(1.7, rel=1e-12) and float(b2) == 0.0


def test_structure_factor_hard_sphere_low_density_limit():
    rho, sig, dr = 0.01, 3.0, 1e-3
    r = torch.arange(dr / 2, 12.0, dr, dtype=torch.float64)
    g = (r >= sig).to(torch.float64)
    q = torch.tensor([0.5, 1.0, 2.0, 3.0], dtype=torch.float64)
    s = structure_factor_from_rdf(r, g, rho, q, lorch=False)
    exact = -4 * math.pi * rho * (torch.sin(q * sig) - q * sig * torch.cos(q * sig)) / q**3
    assert torch.allclose(s, exact, rtol=2e-3, atol=1e-6)


# ---------------------------------------------------------------------- NMR

def test_jcoupling_ensemble_average_and_model_variance():
    p = KARPLUS["3J_HN_HA_VuisterBax1993"]
    phis = [-120.0, -60.0, 60.0]
    frames = _dihedral_frames(phis)
    sys = toy_system((1, 7, 6, 1), frames[0])
    pred = JCoupling((0, 1, 2, 3), p).predict(Zero(), _ens(sys, frames))

    def karplus(phi):
        c = math.cos(math.radians(phi + p.offset_deg))
        return p.A * c * c + p.B * c + p.C

    assert float(pred.value.detach()) == pytest.approx(float(np.mean([karplus(x) for x in phis])), rel=1e-12)
    assert float(pred.variance) >= p.model_sd_hz**2


def test_noe_r6_r3_and_equivalent_proton_summation():
    frames = np.array([[[0, 0, 0], [0, 0, 2.0], [0, 1.5, 0]], [[0, 0, 0], [0, 0, 3.0], [0, 1.5, 0]]], dtype=float)
    sys = toy_system((1, 1, 1), frames[0])
    ens = _ens(sys, frames)
    r6 = NOEDistance([0], [1]).predict(Zero(), ens)
    assert float(r6.value.detach()) == pytest.approx(((2.0**-6 + 3.0**-6) / 2) ** (-1 / 6), rel=1e-12)
    r3 = NOEDistance([0], [1], regime="r-3").predict(Zero(), ens)
    assert float(r3.value.detach()) == pytest.approx(((2.0**-3 + 3.0**-3) / 2) ** (-1 / 3), rel=1e-12)
    grp = NOEDistance([0], [1, 2]).frames(torch.as_tensor(frames[:1]), None)
    assert float(grp[0, 0]) == pytest.approx(2.0**-6 + 1.5**-6, rel=1e-12)  # r^-6 summation


def test_chemical_shift_frozen_predictor_average_and_model_variance():
    frames = np.array([[[0.1, 0, 0]], [[0.3, 0, 0]]])
    sys = toy_system((6,), frames[0])
    cs = ChemicalShift(lambda p: 100.0 + 10.0 * p[:, :1, 0], model_sd_ppm=1.5, name="toy-linear")
    pred = cs.predict(Zero(), _ens(sys, frames))
    assert float(pred.value.detach()) == pytest.approx(102.0, rel=1e-12)
    assert float(pred.variance) >= 1.5**2


# ---------------------------------------------------------------- structure

def test_distance_averages_and_minimum_image():
    frames = np.array([[[0, 0, 0], [1.0, 0, 0]], [[0, 0, 0], [3.0, 0, 0]]], dtype=float)
    sys = toy_system((6, 6), frames[0])
    ens = _ens(sys, frames)
    assert float(DistanceAverage(0, 1, 1.0).predict(Zero(), ens).value.detach()) == pytest.approx(2.0, rel=1e-12)
    assert float(DistanceAverage(0, 1, 2.0).predict(Zero(), ens).value.detach()) == pytest.approx(math.sqrt(5.0))
    L = 10.0
    pf = np.array([[[0.5, 0, 0], [9.5, 0, 0]]])
    psys = System("p", (18, 18), (39.948, 39.948), task="toy", spin=0, pbc=(True,) * 3, positions=pf[0],
                  cell=np.eye(3) * L)
    pe = ReweightedEnsemble(cache_from_frames(psys, pf, np.array([np.eye(3) * L])), psys)
    assert float(DistanceAverage(0, 1, 1.0, periodic=True).predict(Zero(), pe).value.detach()) == pytest.approx(1.0)


def test_state_populations_from_dihedral_windows():
    frames = _dihedral_frames([60.0, 180.0, -60.0, 179.0])
    sys = toy_system((6, 6, 6, 6), frames[0])
    sp = StatePopulation((0, 1, 2, 3), {"g+": [(0.0, 120.0)], "t": [(120.0, 180.1), (-180.1, -120.0)],
                                        "g-": [(-120.0, 0.0)]})
    pred = sp.predict(Zero(), _ens(sys, frames))
    assert torch.allclose(pred.value.detach(), torch.tensor([0.25, 0.5, 0.25], dtype=torch.float64))
