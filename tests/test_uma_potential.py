"""UMA architecture tests (random weights; the gated checkpoint is not required).

Marked `uma`: each test instantiates or evaluates the 146.6M-parameter UMA-S
network on CPU.
"""

import copy

import numpy as np
import pytest
import torch
from ase.build import molecule

from umaexp.potential import UMAPotential, select_parameters, write_synthetic_checkpoint
from umaexp.potential.adapters import add_lowrank_adapters, has_adapters, merge_adapters
from umaexp.systems import System
from umaexp.validation.gradcheck import (
    check_forces,
    check_hessian,
    check_no_detached_path,
    check_parameter_gradient,
)

pytestmark = pytest.mark.uma


def _sys(name, charge=0, spin=1):
    a = molecule(name)
    return System.from_ase(a, system_id=name, charge=charge, spin=spin), torch.tensor(a.get_positions())


def _rot(seed):
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(seed)))
    return q * torch.sign(torch.det(q))


@pytest.mark.skipif(__import__("shutil").disk_usage(__import__("os").path.expanduser("~")).free < 6e9,
                    reason="writes UMA-sized checkpoints (~0.6-1.2 GB); needs >= 6 GB free disk")
def test_parity_with_fairchem_predictor(uma_random, tmp_path):
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

    path = write_synthetic_checkpoint(uma_random, str(tmp_path / "synthetic.pt"))
    ours = UMAPotential.from_checkpoint(path, dtype=torch.float64)
    settings = InferenceSettings(merge_mole=False, compile=False, activation_checkpointing=False,
                                 base_precision_dtype=torch.float64, execution_mode="general")
    calc = FAIRChemCalculator(load_predict_unit(path, inference_settings=settings, device="cpu"),
                              task_name="omol")
    for name, q, s in [("H2O", 0, 1), ("CH3", 0, 2), ("CH4", 0, 1)]:
        system, pos = _sys(name, q, s)
        a = system.to_ase()
        a.calc = calc
        e, f = ours.energy_and_forces(system, pos)
        assert float(e) == pytest.approx(a.get_potential_energy(), abs=1e-7)
        assert np.abs(f.numpy() - a.get_forces()).max() < 1e-7


def test_symmetries(uma_random):
    system, pos = _sys("CH3CH2OH")
    e0, f0 = uma_random.energy_and_forces(system, pos)
    R = _rot(1)
    e1, f1 = uma_random.energy_and_forces(system, pos @ R.T + torch.tensor([3.0, -1.0, 2.0], dtype=torch.float64))
    assert float(e1) == pytest.approx(float(e0), abs=1e-8)
    assert torch.allclose(f1, f0 @ R.T, atol=1e-8)
    perm = torch.randperm(system.n_atoms, generator=torch.Generator().manual_seed(0))
    sp = System.from_ase(system.to_ase()[perm.numpy()], system_id="perm")
    e2 = uma_random.energy(sp, pos[perm])
    assert float(e2) == pytest.approx(float(e0), abs=1e-8)


def test_charge_and_spin_change_energy(uma_random):
    s1, pos = _sys("H2O", 0, 1)
    s3, _ = _sys("H2O", 0, 3)
    sc, _ = _sys("H2O", 1, 2)
    e1, e3, ec = (float(uma_random.energy(s, pos)) for s in (s1, s3, sc))
    assert len({round(e1, 8), round(e3, 8), round(ec, 8)}) == 3


def test_force_and_hessian_finite_differences(uma_random):
    system, pos = _sys("H2O")
    assert check_forces(uma_random, system, pos, h=1e-4).passed
    r = check_hessian(uma_random, system, pos, h=1e-4, rtol=1e-5, atol=1e-6)
    assert r.passed, str(r)


def test_parameter_gradient_directional_check(uma_random):
    pot = copy.deepcopy(uma_random)
    rep = select_parameters(pot.model, preset="readout_norm")
    params = pot.trainable_parameters()
    assert rep.n_tensors == len(params) > 0
    system, pos = _sys("NH3")

    def loss():
        e, f = pot.energy_and_forces(system, pos, create_graph=True)
        return e + (f**2).sum()  # energy + force term exercises mixed derivatives

    r = check_parameter_gradient(loss, params, n_probe=3, h=1e-6, rtol=1e-4)
    assert r.passed, str(r)
    diag = check_no_detached_path(loss(), params)
    assert diag["ok"] and not diag["nonfinite"]


def test_adapters_preserve_parity_symmetry_and_merge(uma_random):
    pot = copy.deepcopy(uma_random)
    system, pos = _sys("CH3CH2OH")
    e_ref = float(pot.energy(system, pos))
    names = add_lowrank_adapters(pot.model, [r"blocks\.3\..*fc_m0\.weights$", r"energy_block\.2\.weight$"], rank=2)
    assert len(names) == 3 and has_adapters(pot.model)  # so2_conv_1/2.fc_m0 + readout layer
    assert float(pot.energy(system, pos)) == pytest.approx(e_ref, abs=1e-10)  # B = 0 init
    with torch.no_grad():
        for p in pot.trainable_parameters():
            p.normal_(0.0, 0.05)
    e0, f0 = pot.energy_and_forces(system, pos)
    assert abs(float(e0) - e_ref) > 1e-6  # adapters now change the surface
    R = _rot(5)
    e1, f1 = pot.energy_and_forces(system, pos @ R.T)
    assert float(e1) == pytest.approx(float(e0), abs=1e-8)
    assert torch.allclose(f1, f0 @ R.T, atol=1e-8)
    merge_adapters(pot.model)
    assert not has_adapters(pot.model)
    assert float(pot.energy(system, pos)) == pytest.approx(float(e0), abs=1e-9)


@pytest.mark.skipif(__import__("shutil").disk_usage(__import__("os").path.expanduser("~")).free < 6e9,
                    reason="writes UMA-sized checkpoints (~0.6-1.2 GB); needs >= 6 GB free disk")
def test_export_roundtrip_through_fairchem(uma_random, tmp_path):
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

    base = write_synthetic_checkpoint(uma_random, str(tmp_path / "base.pt"))
    pot = UMAPotential.from_checkpoint(base, dtype=torch.float64)
    select_parameters(pot.model, preset="readout")
    with torch.no_grad():
        for p in pot.trainable_parameters():
            p.add_(0.01 * torch.randn_like(p))
    out = pot.export_inference_checkpoint(str(tmp_path / "tuned.pt"), dtype=torch.float64)
    settings = InferenceSettings(merge_mole=False, compile=False, activation_checkpointing=False,
                                 base_precision_dtype=torch.float64, execution_mode="general")
    calc = FAIRChemCalculator(load_predict_unit(out, inference_settings=settings, device="cpu"), task_name="omol")
    system, pos = _sys("H2O")
    a = system.to_ase()
    a.calc = calc
    assert a.get_potential_energy() == pytest.approx(float(pot.energy(system, pos)), abs=1e-8)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no Apple MPS")
def test_mps_float32_matches_cpu(uma_random):
    mps = uma_random.to_device("mps", torch.float32)
    system, pos = _sys("CH3CH2OH")
    e_c, f_c = uma_random.energy_and_forces(system, pos)
    e_m, f_m = mps.energy_and_forces(system, pos.to("mps", torch.float32))
    assert float(e_m) == pytest.approx(float(e_c), abs=5e-5)
    assert (f_m.cpu().double() - f_c).abs().max() < 1e-4


def test_vibrational_loss_reaches_uma_weights(uma_random):
    from umaexp.observables.vibrational import (
        ExperimentalMode,
        VibrationalConfig,
        VibrationalForwardModel,
        VibrationalObservable,
    )

    pot = copy.deepcopy(uma_random)
    select_parameters(pot.model, preset="readout")
    system, _ = _sys("H2O")
    exp = [ExperimentalMode("nu2", 1595.0, 1.0, species="a1"), ExperimentalMode("nu1", 3657.0, 1.0, species="a1"),
           ExperimentalMode("nu3", 3756.0, 1.0, species="b1")]
    cfg = VibrationalConfig(fmax=1e-4, forward=VibrationalForwardModel(kind="harmonic", rel_model_sd=0.0))
    obs = VibrationalObservable(system, exp, cfg)
    pred = obs.predict(pot)
    y, var = obs.targets()
    # random weights: the geometry may be unphysical, but the gradient path must exist
    loss = (((pred.value - y) ** 2) / var).sum()
    named = pot.trainable_named_parameters()
    diag = check_no_detached_path(loss, [p for _, p in named])
    assert not diag["nonfinite"]
    # Only the readout's final bias may be disconnected: it adds a per-atom
    # constant, which cannot change a Hessian.
    assert {named[i][0].split("head.")[-1] for i in diag["none"] + diag["zero"]} <= {"energy_block.4.bias"}


def test_extensivity_of_distant_copies(uma_random):
    from umaexp.validation.physics import extensivity_check

    system, pos = _sys("H2O")
    rep = extensivity_check(uma_random, system, pos, separation=20.0)
    assert abs(rep["deviation_eV"]) < 1e-8, rep


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no Apple MPS")
def test_mps_sampling_reweights_exactly_to_cpu_fp64(uma_random):
    """Regression: sampling on Metal (fp32) + reweighting in CPU fp64."""
    from umaexp.reweighting import ReweightedEnsemble
    from umaexp.sampling import SamplingProtocol, ThermoState, run_sampling

    system, _ = _sys("CH3CH2OH")
    mps = uma_random.to_device("mps", torch.float32)
    cache = run_sampling(mps, system, ThermoState(300.0),
                         SamplingProtocol(dt_fs=0.25, n_equilibration=2, n_production=20, stride=5,
                                          device="mps", dtype="float32"))
    ens = ReweightedEnsemble(cache, system)
    du = ens.energies(uma_random, with_graph=False).numpy() - cache.u_sampling
    assert np.abs(du).max() < 1e-5
    est = ens.estimate(uma_random, lambda p, c: p[:, :1, 0], "surrogate")
    assert est.ess_fraction > 0.99


@pytest.mark.slow
@pytest.mark.skipif(__import__("shutil").disk_usage(__import__("os").path.expanduser("~")).free < 6e9,
                    reason="writes UMA-sized checkpoints (~0.6-1.2 GB); needs >= 6 GB free disk")
def test_capability_audit_passes_on_random_architecture(tmp_path):
    from umaexp.validation.audit import run_audit

    rep = run_audit(None, "omol", out_dir=str(tmp_path), benchmark=False, workdir=str(tmp_path))
    assert rep["random_architecture"] and rep["passed"], rep["derivative_checks"]
    assert rep["parity"]["same_settings"]["max_abs_energy_diff_eV"] < 1e-6
    assert rep["presets"]["readout"]["params"] == 33153
