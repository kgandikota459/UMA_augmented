"""Known-answer tests: anchors, the additive-correction comparator and infrastructure."""

import copy
import datetime as dt

import numpy as np
import pytest
import torch
from ase.build import molecule
from ase.io import write
from toy_helpers import InverseR, Zero

from umaexp.data.schema import ValueKind
from umaexp.data.sources import freesolv
from umaexp.data.sources.catalog import CATALOG, GAPS
from umaexp.losses import Target
from umaexp.losses.anchors import AnchorSet, GaussianPrior, baseline_drift_loss, quantum_anchor_loss, \
    short_range_repulsion_penalty
from umaexp.observables.base import Prediction
from umaexp.observables.ensemble import ensemble_prediction
from umaexp.potential.correction import CorrectedPotential, PairCorrection
from umaexp.potential.derivatives import hessian
from umaexp.potential.toy import HarmonicWell, LennardJones, ValenceToy
from umaexp.sampling import SampleCache, SamplingProtocol, ThermoState
from umaexp.sampling.diagnostics import block_standard_error, detect_equilibration, replica_consistency
from umaexp.systems import System, from_file, from_smiles, isotopologue_masses, liquid_box, \
    molecule_index_from_connectivity
from umaexp.training import CacheManager, DirectTask, EnsembleTask
from umaexp.validation.evaluation import compare_reweighted_to_fresh, evaluate_tasks


def _water():
    pos = np.array([[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]], dtype=float)
    sys = System("w", (8, 1, 1), isotopologue_masses((8, 1, 1)), task="toy", spin=0, positions=pos)
    return sys, ValenceToy([(0, 1), (0, 2)], [(1, 0, 2)], [50.0], [0.96], [4.0], [1.82])


def _frames(sys, M=6, sd=0.03, seed=0):
    x0 = sys.positions_tensor()
    return x0 + sd * torch.randn(M, *x0.shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


# ------------------------------------------------------------------ anchors

def test_quantum_anchor_loss_is_gauge_invariant_and_detects_drift():
    sys, pot = _water()
    X = _frames(sys)
    with torch.no_grad():
        e = pot.energy(sys, X)
    F = torch.stack([pot.energy_and_forces(sys, x)[1] for x in X])
    anchors = [AnchorSet(sys, X, e + 123.456, F, theory="toy", source="self")]  # constant offset: gauge
    assert float(quantum_anchor_loss(pot, anchors)) == pytest.approx(0.0, abs=1e-12)
    with torch.no_grad():
        pot.kb += 5.0
    loss = quantum_anchor_loss(pot, anchors)
    assert float(loss) > 1e-3
    (g,) = torch.autograd.grad(loss, pot.kb)
    assert float(g.abs().sum()) > 0


def test_baseline_drift_zero_when_identical_positive_after_change():
    sys, pot = _water()
    base = copy.deepcopy(pot)
    for p in base.parameters():
        p.requires_grad_(False)
    X = _frames(sys)
    assert float(baseline_drift_loss(pot, base, sys, X)) == pytest.approx(0.0, abs=1e-14)
    with torch.no_grad():
        pot.ka += 1.0
    loss = baseline_drift_loss(pot, base, sys, X)
    assert float(loss) > 0
    (g,) = torch.autograd.grad(loss, pot.ka)
    assert float(g.abs().sum()) > 0


def test_short_range_repulsion_penalty_and_prior():
    sys = System("d", (18, 18), (39.948, 39.948), task="toy", spin=0, positions=np.array([[0, 0, 0], [0, 0, 3.8]]))
    rs = torch.linspace(2.6, 3.6, 11, dtype=torch.float64)
    assert float(short_range_repulsion_penalty(LennardJones(cutoff=8.0), sys, 0, 1, sys.positions_tensor(), rs)) == 0.0
    assert float(short_range_repulsion_penalty(InverseR(A=1.0), sys, 0, 1, sys.positions_tensor(), rs)) > 0.0
    prior = GaussianPrior(0.955, 0.02)
    assert float(prior.nll(torch.tensor(0.995, dtype=torch.float64))) == pytest.approx(2.0, rel=1e-12)


# --------------------------------------------------- additive correction model

def test_pair_correction_zero_init_analytic_value_cutoff_symmetry_and_frozen_base():
    import math

    sys, base = _water()
    corr = PairCorrection([1, 8], n_basis=4, r_min=0.8, cutoff=6.0)
    cp = CorrectedPotential(base, corr)
    x = sys.positions_tensor()
    assert float(cp.energy(sys, x)) == pytest.approx(float(base.energy(sys, x)), abs=1e-14)
    assert cp.updates_uma_weights is False and all(not p.requires_grad for p in base.parameters())
    hi = int(corr.pair_index(torch.tensor([1]), torch.tensor([8])))
    assert hi == int(corr.pair_index(torch.tensor([8]), torch.tensor([1])))
    k = 2
    with torch.no_grad():
        corr.coef[hi, k] = 0.7
    dimer = System("OH", (8, 1), isotopologue_masses((8, 1)), task="toy", spin=0,
                   positions=np.array([[0, 0, 0], [0, 0, 1.1]]))
    cz = CorrectedPotential(Zero(), corr)
    r, c, w = 1.1, float(corr.centers[k]), corr.width
    exact = 0.7 * math.exp(-(((r - c) / w) ** 2)) * 0.5 * (math.cos(math.pi * r / 6.0) + 1)
    assert float(cz.energy(dimer, dimer.positions_tensor())) == pytest.approx(exact, rel=1e-12)
    assert float(cz.energy(dimer, torch.tensor([[0, 0, 0], [0, 0, 6.5]], dtype=torch.float64))) == 0.0
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(1)))
    assert float(cp.energy(sys, x @ q.T + 1.0)) == pytest.approx(float(cp.energy(sys, x)), abs=1e-12)
    (g,) = torch.autograd.grad(cz.energy(dimer, dimer.positions_tensor()), corr.coef)
    assert float(g.abs().sum()) > 0


# ------------------------------------------------------------ infrastructure

def test_sample_cache_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    sys = System("c", (6, 6), isotopologue_masses((6, 6)), task="toy", spin=0, pbc=(True,) * 3,
                 positions=np.zeros((2, 3)), cell=np.eye(3) * 10)
    cache = SampleCache("c", sys.atom_order_hash(), ThermoState(300.0, 1.0, "NPT"), SamplingProtocol(seed=7), "fp",
                        rng.normal(size=(5, 2, 3)), np.stack([np.eye(3) * 10] * 5), rng.normal(size=5),
                        bias=rng.normal(size=5), replica=2, diagnostics={"g": 3.0}, meta={"note": "x"})
    stem = cache.save(str(tmp_path))
    back = SampleCache.load(stem)
    for a in ("positions", "cells", "u_sampling", "bias"):
        assert np.array_equal(getattr(back, a), getattr(cache, a))
    assert back.state == cache.state and back.protocol == cache.protocol and back.replica == 2
    assert back.diagnostics == {"g": 3.0} and back.meta == {"note": "x"}


def test_block_error_replica_consistency_and_equilibration_detection():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 2.0, 20000)
    assert block_standard_error(x, 20) == pytest.approx(2.0 / np.sqrt(20000), rel=0.35)
    sems = np.full(8, 0.1)
    assert replica_consistency(rng.normal(0, 0.1, 8), sems) < 3.0
    assert replica_consistency(np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]), sems) > 10.0
    series = np.concatenate([np.linspace(5.0, 0.0, 300), rng.normal(0, 0.1, 3000)])  # initial transient
    t0, _, _ = detect_equilibration(series)
    assert 200 <= t0 <= 450


def test_freesolv_loader_selects_experimental_fields_and_flags():
    db = {
        "mobley_1": {"smiles": "CCO", "expt": -5.0, "d_expt": 0.6, "expt_reference": "10.1/abc", "iupac": "ethanol",
                     "groups": ["alcohol"], "calc": -4.1,
                     "notes": ["Experimental uncertainty not presently available, so assigned a default value."]},
        "mobley_2": {"smiles": "CCl", "expt": -0.5, "d_expt": 0.2, "expt_reference": "10.1/xyz", "iupac": "ClMe",
                     "groups": [], "calc": 0.1, "notes": []},
    }
    info = {"url": "file://synthetic", "sha256": "0", "retrieval_date": dt.date.today().isoformat()}
    obs = freesolv.to_observations(db, info, elements={"H", "C", "N", "O"})
    assert [o.observation_id for o in obs] == ["freesolv:mobley_1"]  # Cl filtered out
    o = obs[0]
    from umaexp import units

    assert o.value == pytest.approx(-5.0 * units.KCAL_MOL_TO_EV) and o.value_kind == ValueKind.EVALUATED
    assert "default_uncertainty_assigned" in o.quality_flags and o.groups.publication_group == "10.1/abc"
    assert o.labels["expt_kcal_mol"] == -5.0 and "calc" not in o.labels  # computed field never used


def test_source_catalog_is_complete():
    assert set(CATALOG) == {f"D{i}" for i in range(1, 12)}
    for s in CATALOG.values():
        assert s.name and s.url and s.role and s.curation and s.status and s.license
    assert GAPS


def test_connectivity_file_builder_and_packing_failure(tmp_path):
    w2 = molecule("H2O")
    w2.translate([5.0, 0, 0])
    both = molecule("H2O") + w2
    assert molecule_index_from_connectivity(both) == [0, 0, 0, 1, 1, 1]
    path = tmp_path / "w.xyz"
    write(str(path), both)
    sys = from_file(str(path), system_id="w2", molecule_index=[0, 0, 0, 1, 1, 1])
    assert sys.atomic_numbers == (8, 1, 1, 8, 1, 1) and sys.n_molecules == 2
    assert sys.provenance["builder"] == "ase.io.read"
    with pytest.raises(RuntimeError, match="Packing failed"):
        liquid_box(from_smiles("CCCCCC", "hexane"), 30, 2.0, max_attempts=200)


def test_vmap_hessian_equals_row_hessian():
    sys, pot = _water()
    x = sys.positions_tensor()
    assert torch.allclose(hessian(pot, sys, x, create_graph=False, method="rows"),
                          hessian(pot, sys, x, create_graph=False, method="vmap"), rtol=1e-12, atol=1e-12)


def _r2_task(n=4, T=300.0):
    from umaexp import units

    sys = System("hw", (18,) * n, (39.948,) * n, task="toy", spin=0, positions=np.zeros((n, 3)))

    class R2:
        def predict(self, potential, ens, mode):
            return ensemble_prediction(ens, potential, lambda p, c: (p**2).sum(dim=(1, 2))[:, None], None, mode)

    target = 3 * n * units.KB_EV_K * T / 1.0
    return EnsembleTask("r2", "structure", sys, ThermoState(T),
                        SamplingProtocol(dt_fs=5.0, friction_per_ps=5.0, n_equilibration=300, n_production=4000,
                                         stride=5, seed=0), R2(),
                        Target(["r2"], "structure", torch.tensor([target], dtype=torch.float64),
                               torch.tensor([0.05 * target], dtype=torch.float64)))


def test_cache_manager_extend_then_refresh_after_weight_change():
    pot = HarmonicWell(k=1.0)
    task = _r2_task()
    cm = CacheManager()
    c1 = cm.get(task, pot)
    c2 = cm.extend(task, pot, "more statistics")
    assert c2.n_frames == 2 * c1.n_frames and c2.sampling_fingerprint == c1.sampling_fingerprint
    assert cm.events[-1].action == "extend"
    with torch.no_grad():
        pot.log_k += 0.1
    c3 = cm.extend(task, pot, "more statistics")
    assert cm.events[-1].action == "refresh" and c3.sampling_fingerprint != c1.sampling_fingerprint


def test_fresh_evaluation_and_reweighted_vs_fresh_gate():
    pot = HarmonicWell(k=1.0)
    task = _r2_task()
    sys = task.system

    class EnergyAt:
        def predict(self, potential):
            x = torch.full((sys.n_atoms, 3), 0.1, dtype=torch.float64)
            return Prediction(potential.energy(sys, x).reshape(1), torch.zeros(1, dtype=torch.float64))

    direct = DirectTask("e", "misc", EnergyAt(), target=Target(["e"], "misc", torch.tensor([0.06], dtype=torch.float64),
                                                                torch.tensor([0.01], dtype=torch.float64)))
    df = evaluate_tasks(pot, [direct, task], label="t")
    assert list(df["task"]) == ["e", "r2"] and df["fresh"].all() and np.isfinite(df["z"]).all()
    assert df.loc[0, "prediction"] == pytest.approx(0.5 * 1.0 * 0.01 * 3 * sys.n_atoms, rel=1e-12)
    gate = compare_reweighted_to_fresh(pot, task, CacheManager().get(task, pot))
    assert gate["agree"] and gate["max_abs_z"] < 3.0
