"""Tests against the REAL uma-s-1p1 weights (skipped when the gated checkpoint is not cached).

Run with:  pytest -m uma_real
"""

import copy
import glob
import os

import numpy as np
import pytest
import torch
from ase.build import molecule

from umaexp.potential import UMAPotential, select_parameters
from umaexp.systems import System
from umaexp.validation.gradcheck import check_hessian, check_no_detached_path

CACHE = os.path.expanduser("~/.cache/fairchem/models--facebook--UMA/snapshots/*/checkpoints/uma-s-1p1.pt")
_paths = glob.glob(CACHE)
pytestmark = [pytest.mark.uma, pytest.mark.uma_real,
              pytest.mark.skipif(not _paths, reason="uma-s-1p1 not cached (gated: needs approved HF access)")]


@pytest.fixture(scope="module")
def uma_real():
    return UMAPotential.from_checkpoint(_paths[0], task="omol", dtype=torch.float64)


def _sys(name, charge=0, spin=1):
    a = molecule(name)
    a.info.update(charge=charge, spin=spin)
    return System.from_ase(a, system_id=name, charge=charge, spin=spin), torch.tensor(a.get_positions())


def test_parity_with_fairchem_predictor_on_equilibrium_molecules(uma_real):
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

    settings = InferenceSettings(merge_mole=False, compile=False, activation_checkpointing=False,
                                 base_precision_dtype=torch.float64, execution_mode="general")
    calc = FAIRChemCalculator(load_predict_unit(uma_real._checkpoint_path, inference_settings=settings, device="cpu"),
                              task_name="omol")
    for name, q, s in [("H2O", 0, 1), ("CH4", 0, 1), ("NH3", 0, 1), ("CH3CN", 0, 1), ("CH3", 0, 2)]:
        system, pos = _sys(name, q, s)
        a = system.to_ase()
        a.calc = calc
        e, f = uma_real.energy_and_forces(system, pos)
        assert float(e) == pytest.approx(a.get_potential_energy(), abs=1e-7)
        assert np.abs(f.numpy() - a.get_forces()).max() < 1e-7


def test_symmetry_extensivity_and_hessian(uma_real):
    from umaexp.validation.physics import extensivity_check, symmetry_check

    system, pos = _sys("CH3CN")
    rep = symmetry_check(uma_real, system, pos)
    assert max(rep.values()) < 1e-7, rep
    w, wpos = _sys("H2O")
    assert abs(extensivity_check(uma_real, w, wpos, separation=20.0)["deviation_eV"]) < 1e-7
    r = check_hessian(uma_real, w, wpos, h=1e-4, rtol=1e-5, atol=1e-5)
    assert r.passed, str(r)


def test_water_frequencies_are_physical_and_gradient_reaches_readout(uma_real):
    from umaexp.observables.vibrational import (
        ExperimentalMode,
        VibrationalConfig,
        VibrationalForwardModel,
        VibrationalObservable,
    )

    pot = copy.deepcopy(uma_real)
    select_parameters(pot.model, preset="readout")
    system, _ = _sys("H2O")
    exp = [ExperimentalMode("nu2", 1595.0, 0.5), ExperimentalMode("nu1", 3657.0, 0.5),
           ExperimentalMode("nu3", 3756.0, 0.5)]
    obs = VibrationalObservable(system, exp, VibrationalConfig(fmax=1e-6,
                                forward=VibrationalForwardModel("fundamental_scaled", rel_model_sd=0.015)))
    pred = obs.predict(pot)
    assert pred.valid, pred.reason
    nu = sorted(pred.diagnostics["all_wavenumbers"])
    # harmonic water: bend ~1600-1700, stretches ~3700-4000 cm^-1 (experimental omega_e 1648, 3832, 3943)
    assert 1580 < nu[0] < 1720 and 3650 < nu[1] < 3950 and 3750 < nu[2] < 4050
    y, var = obs.targets()
    loss = (((pred.value - y) ** 2) / (var + pred.variance)).sum()
    named = pot.trainable_named_parameters()
    diag = check_no_detached_path(loss, [p for _, p in named])
    assert not diag["nonfinite"]
    assert {named[i][0].split("head.")[-1] for i in diag["none"] + diag["zero"]} <= {"energy_block.4.bias"}


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no Apple MPS")
def test_mps_matches_cpu_on_real_weights(uma_real):
    mps = uma_real.to_device("mps", torch.float32)
    system, pos = _sys("CH3CN")
    e_c, f_c = uma_real.energy_and_forces(system, pos)
    e_m, f_m = mps.energy_and_forces(system, pos.to(torch.float32).to("mps"))
    assert float(e_m) == pytest.approx(float(e_c), abs=1e-3)  # fp32 on ~ -4218 eV totals
    assert (f_m.cpu().double() - f_c).abs().max() < 1e-4


@pytest.mark.skipif(__import__("shutil").disk_usage(os.path.expanduser("~")).free < 6e9,
                    reason="writes a ~1.2 GB float64 checkpoint; needs >= 6 GB free disk")
def test_export_roundtrip_with_real_base_checkpoint(uma_real, tmp_path):
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

    pot = copy.deepcopy(uma_real)
    select_parameters(pot.model, preset="readout")
    with torch.no_grad():
        for p in pot.trainable_parameters():
            p.add_(1e-3 * torch.randn_like(p))
    out = pot.export_inference_checkpoint(str(tmp_path / "tuned.pt"), dtype=torch.float64)
    settings = InferenceSettings(merge_mole=False, compile=False, activation_checkpointing=False,
                                 base_precision_dtype=torch.float64, execution_mode="general")
    calc = FAIRChemCalculator(load_predict_unit(out, inference_settings=settings, device="cpu"), task_name="omol")
    system, pos = _sys("H2O")
    a = system.to_ase()
    a.calc = calc
    assert a.get_potential_energy() == pytest.approx(float(pot.energy(system, pos)), abs=1e-7)
    assert a.get_potential_energy() != pytest.approx(float(uma_real.energy(system, pos)), abs=1e-6)
    os.remove(out)


def test_hvp_and_graph_gradients_agree_with_implicit_geometry_and_corrected_forward_model(uma_real):
    """The memory-bounded HVP backward used in Vib-v1 training gives the full-graph gradient (real weights)."""
    from umaexp.observables.vibrational import (
        ExperimentalMode,
        VibrationalConfig,
        VibrationalForwardModel,
        VibrationalObservable,
    )

    system, _ = _sys("H2O")
    exp = [ExperimentalMode("a11", 3657.0, 0.5, species="a1", number=1, anharmonic_correction=-164.3),
           ExperimentalMode("a12", 1595.0, 0.5, species="a1", number=2, anharmonic_correction=-50.9),
           ExperimentalMode("b13", 3756.0, 0.5, species="b1", number=3, anharmonic_correction=-173.3)]
    grads = {}
    for mode in ("graph", "hvp"):
        pot = copy.deepcopy(uma_real)
        select_parameters(pot.model, preset="readout")
        fwd = VibrationalForwardModel("fundamental_corrected", rel_model_sd=0.0, abs_model_sd=10.0)
        obs = VibrationalObservable(system, exp, VibrationalConfig(fmax=1e-6, memory_mode=mode, assignment="symmetry",
                                                                    point_group="C2v", forward=fwd))
        pred = obs.predict(pot)
        assert pred.valid, pred.reason
        y, var = obs.targets()
        loss = (((pred.value - y) ** 2) / (var + pred.variance)).sum()
        if pred.backward_fn is not None:
            pred.backward(torch.autograd.grad(loss, pred.value)[0])
        else:
            loss.backward()
        grads[mode] = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                                 for p in pot.trainable_parameters()])
    rel = (grads["hvp"] - grads["graph"]).norm() / grads["graph"].norm()
    assert rel < 1e-6, float(rel)


@pytest.mark.parametrize("name", ["CH4", "CH3Cl"])
def test_vpt2_is_invariant_to_rotations_inside_degenerate_subspaces(uma_real, name):
    """PROTOCOL 3.1 / critic report #2 M2: Cartesian-component VPT2 does not depend on the basis chosen
    inside degenerate subspaces, up to the O(step^2) finite-difference error of the force field
    (development molecules; real weights)."""
    from dataclasses import replace

    from umaexp.observables.vibrational import degenerate_groups, harmonic_modes, optimize_geometry
    from umaexp.observables.vpt2 import normal_mode_force_field, vpt2_core

    system, pos = _sys(name)
    x, _ = optimize_geometry(uma_real, system, pos, fmax=1e-7)
    nm = harmonic_modes(uma_real, system, x, create_graph=False)
    groups = [g for g in degenerate_groups(nm.wavenumbers.detach().numpy()) if len(g) > 1]
    assert groups, "expected degenerate modes"
    rng = np.random.default_rng(1)
    V = nm.modes.clone()
    for g in groups:
        q, _ = np.linalg.qr(rng.normal(size=(len(g), len(g))))
        V[:, g] = V[:, g] @ torch.tensor(q)
    worst = {}
    for step in (0.05, 0.025):
        res = []
        for modes in (nm, replace(nm, modes=V)):
            ff = normal_mode_force_field(uma_real, system, x, step=step, nm=modes)
            res.append(vpt2_core(ff.omega, ff.phi3, ff.phi4, ff.zeta, ff.B).nu)
        worst[step] = max(max(np.ptp(r[g]) for r in res for g in groups),
                          max(abs(res[0][g].mean() - res[1][g].mean()) for g in groups))
    assert worst[0.025] < 0.05  # production step (PROTOCOL 3.1)
    assert worst[0.025] < 0.5 * worst[0.05] or worst[0.05] < 0.01  # shrinks like step^2: numerical, not systematic
