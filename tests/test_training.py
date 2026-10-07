import copy

import numpy as np
import pytest
import torch

from umaexp import units
from umaexp.losses import Target
from umaexp.observables.ensemble import ensemble_prediction
from umaexp.observables.vibrational import (
    ExperimentalMode,
    VibrationalConfig,
    VibrationalForwardModel,
    VibrationalObservable,
)
from umaexp.potential.toy import HarmonicWell, ValenceToy
from umaexp.reweighting import ControllerConfig
from umaexp.sampling import SamplingProtocol, ThermoState
from umaexp.systems import System, isotopologue_masses
from umaexp.training import CacheManager, DirectTask, EnsembleTask, Trainer, TrainerConfig
from umaexp.training.checkpointing import load_training_state, save_training_state


def _toy_water(kb=50.0, ka=4.0):
    r0, th0 = 0.9572, np.deg2rad(104.5)
    pos = np.array([[0, 0, 0], [r0, 0, 0], [r0 * np.cos(th0), r0 * np.sin(th0), 0]]) + 0.01
    sys = System("toy_H2O", (8, 1, 1), isotopologue_masses((8, 1, 1)), task="toy", spin=0, positions=pos)
    return sys, ValenceToy([(0, 1), (0, 2)], [(1, 0, 2)], [kb], [r0], [ka], [th0])


def test_direct_vibrational_training_recovers_targets():
    sys, truth = _toy_water(62.0, 5.1)
    true_nu = VibrationalObservable(sys, [ExperimentalMode(n, 1.0, 1.0) for n in "abc"],
                                    VibrationalConfig(forward=VibrationalForwardModel("harmonic", rel_model_sd=0.0))
                                    ).predict(truth).value.detach()
    exp = [ExperimentalMode(n, float(v), 2.0) for n, v in zip("abc", sorted(true_nu.tolist()))]
    _, pot = _toy_water()
    obs = VibrationalObservable(sys, exp, VibrationalConfig(forward=VibrationalForwardModel("harmonic", rel_model_sd=0.0)))

    def target_fn(pred):
        y, var = obs.targets()
        return Target(pred.meta["labels"], "vibrational", y, var.sqrt())

    task = DirectTask("vib", "vibrational", obs, target_fn=target_fn)
    cfg = TrainerConfig(steps=300, lr=1.0, lr_decay=0.98, max_relative_update=None, grad_clip=None)
    for p in (pot.r0, pot.theta0):
        p.requires_grad_(False)
    hist = Trainer(pot, [task], cfg).fit()
    assert hist[-1]["loss"] < 1e-3 * hist[0]["loss"]
    assert float(pot.kb) == pytest.approx(62.0, rel=1e-3) and float(pot.ka) == pytest.approx(5.1, rel=1e-3)


def test_ensemble_training_with_refresh_and_trust_region():
    T = 300.0
    n = 4
    sys = System("hw", (18,) * n, (39.948,) * n, task="toy", spin=0, positions=np.zeros((n, 3)))
    k_true = 2.0
    target_r2 = 3 * n * units.KB_EV_K * T / k_true
    pot = HarmonicWell(k=1.0)

    class R2:
        def predict(self, potential, ens, mode):
            return ensemble_prediction(ens, potential, lambda p, c: (p**2).sum(dim=(1, 2))[:, None], None, mode)

    proto = SamplingProtocol(dt_fs=5.0, friction_per_ps=5.0, n_equilibration=300, n_production=6000, stride=5, seed=0)
    task = EnsembleTask("r2", "structure", sys, ThermoState(T), proto, R2(),
                        Target(["r2"], "structure", torch.tensor([target_r2], dtype=torch.float64),
                               torch.tensor([0.02 * target_r2], dtype=torch.float64)), mode="surrogate")
    cm = CacheManager()
    cfg = TrainerConfig(steps=40, lr=0.1, lr_decay=0.95, max_relative_update=0.2, grad_clip=None,
                        controller=ControllerConfig(ess_refresh=0.6, ess_reject=0.2, min_effective_samples=20,
                                                    sem_over_sigma_extend=5.0))
    hist = Trainer(pot, [task], cfg, cache_manager=cm).fit()
    assert float(pot.k.detach()) == pytest.approx(k_true, rel=0.08)
    actions = {e.action for e in cm.events}
    assert "build" in actions and "refresh" in actions  # overlap-driven resampling happened
    assert hist[-1]["loss"] < hist[0]["loss"]


def test_checkpoint_roundtrip(tmp_path):
    _, pot = _toy_water()
    opt = torch.optim.AdamW(pot.trainable_parameters(), lr=1e-3)
    path = save_training_state(str(tmp_path / "s.pt"), pot, opt, 7)
    before = {n: p.detach().clone() for n, p in pot.trainable_named_parameters()}
    with torch.no_grad():
        for p in pot.trainable_parameters():
            p.add_(1.0)
    st = load_training_state(path, pot, opt)
    assert st["step"] == 7
    for n, p in pot.trainable_named_parameters():
        assert torch.equal(p.detach(), before[n])


@pytest.mark.uma
def test_uma_training_step_updates_weights(uma_random, tmp_path):
    """Training mechanics on the UMA architecture (random weights).

    Uses an always-valid direct observable (energy difference between two
    fixed water geometries); vibrational predictions on random weights may
    legitimately abstain (no valid minimum), which the trainer must respect.
    """
    from ase.build import molecule

    from umaexp.observables.base import Prediction
    from umaexp.potential import select_parameters
    from umaexp.systems import System as S

    pot = copy.deepcopy(uma_random)
    baseline = uma_random.frozen_copy()
    select_parameters(pot.model, preset="readout")
    a = molecule("H2O")
    sys = S.from_ase(a, system_id="H2O")
    x0 = torch.tensor(a.get_positions())
    x1 = x0.clone()
    x1[1] = x1[0] + 1.1 * (x1[1] - x1[0])

    class DeltaE:
        def predict(self, potential):
            return Prediction((potential.energy(sys, x1) - potential.energy(sys, x0)).reshape(1),
                              torch.zeros(1, dtype=torch.float64))

    tgt = Target(["dE"], "thermochemistry", torch.tensor([0.25], dtype=torch.float64),
                 torch.tensor([0.01], dtype=torch.float64))
    tr = Trainer(pot, [DirectTask("dE", "thermochemistry", DeltaE(), target=tgt)],
                 TrainerConfig(steps=3, lr=1e-3, checkpoint_dir=str(tmp_path), checkpoint_every=3), baseline=baseline)
    assert tr.verify_initialization([(sys, x0)])["max_abs_energy_diff_eV"] == 0.0
    hist = tr.fit()
    assert all(h["n_params_with_grad"] == 6 for h in hist)  # readout: 3 weights + 3 biases
    assert all(h["update"].startswith("accepted") for h in hist)
    assert hist[-1]["loss"] < hist[0]["loss"]
    assert pot.parameter_fingerprint(trainable_only=True) != baseline.parameter_fingerprint(trainable_only=True)
    assert (tmp_path / "step00003.pt").exists()


def test_surrogate_nonlinear_transform_through_trainer():
    """Regression: the trainer frees the loss graph before deferred surrogate
    callbacks run; nonlinear transforms (rho = M/<V>) must still backprop."""
    from umaexp.observables.density import DensityObservable
    from umaexp.potential.toy import LennardJones, lj_argon_system

    sys = lj_argon_system(n_side=3, density_g_cm3=1.3)
    L = float(np.linalg.det(sys.cell) ** (1 / 3))
    pot = LennardJones(cutoff=0.4 * L)
    proto = SamplingProtocol(dt_fs=5.0, n_equilibration=200, n_production=1000, stride=10, barostat_every=5, seed=0)
    task = EnsembleTask("rho", "density", sys, ThermoState(100.0, 400.0, "NPT"), proto, DensityObservable(sys),
                        Target(["rho"], "density", torch.tensor([1.2], dtype=torch.float64),
                               torch.tensor([0.01], dtype=torch.float64)), mode="surrogate")
    tr = Trainer(pot, [task], TrainerConfig(steps=1, lr=1e-3, controller=ControllerConfig(
        ess_refresh=0.0, min_effective_samples=1, sem_over_sigma_extend=1e9)))
    rec = tr.step(0)
    assert rec["n_params_with_grad"] == 2 and rec["update"].startswith("accepted")


def test_l2sp_anchors_to_initial_weights_and_nuisance_not_decayed():
    """L2-SP pulls weights back toward w0 (not toward zero); nuisances are never decayed."""
    from umaexp.observables.base import Prediction

    _, pot = _toy_water()

    class Null:  # observable with zero loss gradient: only L2-SP acts
        def predict(self, potential):
            return Prediction((0.0 * potential.kb).sum().reshape(1), torch.zeros(1, dtype=torch.float64))

    scale = torch.nn.Parameter(torch.tensor(0.95, dtype=torch.float64))
    tgt = Target(["x"], "misc", torch.zeros(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64))
    tr = Trainer(pot, [DirectTask("n", "misc", Null(), target=tgt)],
                 TrainerConfig(steps=60, lr=0.5, l2sp=1.0, weight_decay=0.5, max_relative_update=None, grad_clip=None),
                 nuisance={"s": scale})
    w0 = float(pot.kb)  # recorded by the trainer at construction
    with torch.no_grad():
        pot.kb.add_(3.0)
    tr.fit()
    assert float(pot.kb) == pytest.approx(w0, abs=0.3)  # back to w0, not decayed toward 0
    assert float(scale) == pytest.approx(0.95, abs=1e-12)


def test_lm_optimizer_direct_and_ensemble():
    """Observation-space Levenberg-Marquardt converges in few steps on both routes."""
    # direct (vibrational) route: recover force constants of the toy triatomic
    sys, truth = _toy_water(62.0, 5.1)
    fm = VibrationalForwardModel("harmonic", rel_model_sd=0.0)
    true_nu = VibrationalObservable(sys, [ExperimentalMode(n, 1.0, 1.0) for n in "abc"],
                                    VibrationalConfig(forward=fm)).predict(truth).value.detach()
    exp = [ExperimentalMode(n, float(v), 2.0) for n, v in zip("abc", sorted(true_nu.tolist()))]
    _, pot = _toy_water()
    for p in (pot.r0, pot.theta0):
        p.requires_grad_(False)
    obs = VibrationalObservable(sys, exp, VibrationalConfig(forward=fm, memory_mode="hvp"))

    def target_fn(pred):
        y, var = obs.targets()
        return Target(pred.meta["labels"], "vibrational", y, var.sqrt())

    hist = Trainer(pot, [DirectTask("vib", "vibrational", obs, target_fn=target_fn)],
                   TrainerConfig(steps=8, optimizer="lm", max_relative_update=None)).fit()
    assert float(pot.kb) == pytest.approx(62.0, rel=1e-4) and float(pot.ka) == pytest.approx(5.1, rel=1e-4)
    assert hist[-1]["loss"] < 1e-6

    # ensemble (reweighting) route: recover k of a harmonic well from <r^2>
    T, n = 300.0, 4
    s2 = System("hw", (18,) * n, (39.948,) * n, task="toy", spin=0, positions=np.zeros((n, 3)))
    target_r2 = 3 * n * units.KB_EV_K * T / 2.0
    well = HarmonicWell(k=1.0)

    class R2:
        def predict(self, potential, ens, mode):
            return ensemble_prediction(ens, potential, lambda p, c: (p**2).sum(dim=(1, 2))[:, None], None, mode)

    proto = SamplingProtocol(dt_fs=5.0, friction_per_ps=5.0, n_equilibration=300, n_production=6000, stride=5, seed=0)
    task = EnsembleTask("r2", "structure", s2, ThermoState(T), proto, R2(),
                        Target(["r2"], "structure", torch.tensor([target_r2], dtype=torch.float64),
                               torch.tensor([0.02 * target_r2], dtype=torch.float64)))
    Trainer(well, [task], TrainerConfig(steps=8, optimizer="lm", max_relative_update=0.5,
                                        controller=ControllerConfig(ess_refresh=0.6, ess_reject=0.2,
                                                                    min_effective_samples=20,
                                                                    sem_over_sigma_extend=5.0))).fit()
    assert float(well.k.detach()) == pytest.approx(2.0, rel=0.08)


def test_nuisance_parameters_get_their_own_learning_rate():
    _, pot = _toy_water()
    scale = torch.nn.Parameter(torch.tensor(0.95, dtype=torch.float64))
    tr = Trainer(pot, [], TrainerConfig(lr=1e-5, nuisance_lr=2e-3), nuisance={"s": scale})
    lrs = {id(p): g["lr"] for g in tr.opt.param_groups for p in g["params"]}
    assert lrs[id(scale)] == 2e-3 and lrs[id(pot.kb)] == 1e-5
