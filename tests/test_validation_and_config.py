import os

import numpy as np
import pytest
import torch

from umaexp.config import ExperimentConfig, load_config
from umaexp.potential.toy import LennardJones, ValenceToy, lj_argon_system
from umaexp.systems import System, isotopologue_masses
from umaexp.validation.physics import cutoff_smoothness, nve_drift, symmetry_check

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_config_merge_and_typo_rejection(tmp_path):
    cfg = load_config(os.path.join(ROOT, "configs", "vib_pilot.yaml"), overrides=["training.steps=3"])
    assert isinstance(cfg, ExperimentConfig) and cfg.training.steps == 3 and cfg.potential.preset == "readout"
    bad = tmp_path / "bad.yaml"
    bad.write_text("training:\n  stepz: 3\n")
    from omegaconf.errors import ConfigKeyError

    with pytest.raises(ConfigKeyError):
        load_config(str(bad))


def test_symmetry_and_drift_checks_on_toys():
    pos = np.array([[0.0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    sys = System("t", (8, 1, 1), isotopologue_masses((8, 1, 1)), task="toy", spin=0, positions=pos)
    pot = ValenceToy([(0, 1), (0, 2)], [(1, 0, 2)], [50.0], [0.96], [4.0], [1.82])
    rep = symmetry_check(pot, sys, torch.tensor(pos))
    assert max(rep.values()) < 1e-10
    lj = lj_argon_system(n_side=3, density_g_cm3=1.2)
    L = float(np.linalg.det(lj.cell) ** (1 / 3))
    d = nve_drift(LennardJones(cutoff=0.49 * L), lj, lj.positions_tensor(), lj.cell_tensor(), temperature_k=60.0,
                  dt_fs=2.0, n_steps=300)
    assert abs(d["drift_eV_per_atom_per_ps"]) < 1e-3


def test_cutoff_smoothness_shifted_force_lj():
    sys = System("dimer", (18, 18), (39.948, 39.948), task="toy", spin=0, positions=np.array([[0, 0, 0], [0, 0, 4.0]]))
    pot = LennardJones(cutoff=8.0)
    rep = cutoff_smoothness(pot, sys, sys.positions_tensor(), 0, 1, 7.5, 8.5, n=400)
    assert rep["max_energy_step"] < 1e-5 and rep["max_force_step"] < 1e-4
    assert rep["max_force_path_inconsistency"] < 1e-4


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "data", "processed", "vibrational.parquet")),
                    reason="pilot data not built (scripts/build_pilot_dataset.py)")
def test_pipeline_builds_vibrational_tasks_from_registry():
    from umaexp.pipeline import Built, vibrational_tasks

    cfg = load_config(os.path.join(ROOT, "configs", "vib_pilot.yaml"),
                      overrides=[f"data.processed_dir={ROOT}/data/processed",
                                 f"data.splits={ROOT}/data/manifests/pilot_splits.json"])
    built = Built(potential=None, baseline=None, selection="", nuisance={}, priors={})
    tasks = vibrational_tasks(cfg, built, "train")
    assert tasks and all(t.family == "vibrational" for t in tasks)
    for t in tasks:
        modes = t.observable.exp_modes
        assert all(m.sigma > 0 and m.degeneracy in (1, 2, 3) for m in modes)
        # excluded flags must not leak into targets (e.g. Fermi resonance / Coriolis-observed)
        assert t.observable.system.task == "omol"


@pytest.mark.uma
def test_adiabatic_ie_pipeline_on_uma_architecture(uma_random):
    from ase.build import molecule

    from umaexp.observables.electronic import AdiabaticIonization

    a = molecule("NH3")
    neutral = System.from_ase(a, system_id="NH3", charge=0, spin=1)
    ion = AdiabaticIonization.ion_of(neutral, +1, 2)
    obs = AdiabaticIonization(neutral, ion, "IE", include_zpe=False)
    with pytest.raises(ValueError):
        AdiabaticIonization(neutral, AdiabaticIonization.ion_of(neutral, -1, 2), "IE")
    pred = obs.predict(uma_random)
    assert pred.value.shape == (1,) and torch.isfinite(pred.value).all()


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "data", "processed", "density.parquet")),
                    reason="pilot data not built (scripts/build_pilot_dataset.py)")
def test_hard_packing_falls_back_to_looser_insertion():
    from umaexp.pipeline import pack_liquid
    from umaexp.systems import from_smiles

    hexane = from_smiles("CCCCCC", system_id="n-hexane")
    box = pack_liquid(hexane, 16, 0.7, seed=0, system_id="hexane_box16", acknowledge_periodic_omol=True)
    assert box.is_periodic and len(box.atomic_numbers) == 16 * 20


def test_pipeline_builds_density_tasks_and_refuses_test_split():
    from umaexp.pipeline import Built, density_tasks

    cfg = load_config(os.path.join(ROOT, "configs", "density_pilot.yaml"),
                      overrides=["density.systems=[methanol,water]", "density.n_molecules=16",
                                 f"data.processed_dir={ROOT}/data/processed",
                                 f"data.splits={ROOT}/data/manifests/pilot_splits.json"])
    built = Built(potential=None, baseline=None, selection="", nuisance={}, priors={})
    tasks = [t for split in ("train", "val") for t in density_tasks(cfg, built, split)]
    assert {t.name for t in tasks} == {"density:methanol", "density:water"}
    with pytest.raises(PermissionError):  # held-out data are only built by the unblinding script
        density_tasks(cfg, built, "test")
    for t in tasks:
        assert t.system.is_periodic and t.state.ensemble == "NPT"
        assert t.system.provenance["periodic_omol_acknowledged"] is True


def test_reserved_held_out_candidates_are_refused_by_every_loader():
    from umaexp.data.reserved import assert_not_reserved, reserved_blocks
    from umaexp.pipeline import Built, density_tasks

    blocks = reserved_blocks()
    assert len(blocks) == 73  # G2 candidates after the label-free offline steps 1-4
    with pytest.raises(PermissionError):
        assert_not_reserved([sorted(blocks)[0]], "test")
    cfg = load_config(os.path.join(ROOT, "configs", "density_pilot.yaml"),
                      overrides=["density.systems=[acetone]", "density.n_molecules=16",
                                 f"data.processed_dir={ROOT}/data/processed",
                                 f"data.splits={ROOT}/data/manifests/pilot_splits.json"])
    built = Built(potential=None, baseline=None, selection="", nuisance={}, priors={})
    with pytest.raises(PermissionError):  # acetone is a G2 candidate: reserved until unblinding
        density_tasks(cfg, built, "train")
