"""Build potentials and training tasks from a config + curated registries."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import torch

from .config import ExperimentConfig
from .data.registry import ExperimentRegistry
from .data.schema import Observation
from .data.reserved import assert_not_reserved
from .losses.anchors import GaussianPrior
from .losses.objective import Target
from .observables.density import DensityObservable
from .observables.vibrational import (
    ExperimentalMode,
    VibrationalConfig,
    VibrationalForwardModel,
    VibrationalObservable,
)
from .potential import UMAPotential, select_parameters
from .potential.adapters import add_lowrank_adapters
from .sampling.cache import SamplingProtocol, ThermoState
from .systems import from_smiles, liquid_box
from .training.tasks import DirectTask, EnsembleTask

log = logging.getLogger(__name__)
_DT = {"float64": torch.float64, "float32": torch.float32}


@dataclass
class Built:
    potential: UMAPotential
    baseline: UMAPotential
    selection: str
    nuisance: dict
    priors: dict


def build_potential(cfg: ExperimentConfig) -> Built:
    pc = cfg.potential
    if pc.checkpoint:
        pot = UMAPotential.from_checkpoint(pc.checkpoint, task=pc.task, dtype=_DT[pc.dtype], device=pc.device)
    else:
        log.warning("No checkpoint configured: RANDOM UMA architecture (mechanics only, no physics).")
        pot = UMAPotential.random_architecture(task=pc.task, dtype=_DT[pc.dtype], device=pc.device, seed=cfg.seed)
    baseline = pot.frozen_copy()
    if pc.adapters.enabled:
        names = add_lowrank_adapters(pot.model, pc.adapters.patterns, pc.adapters.rank, pc.adapters.per_expert)
        selection = f"adapters(rank={pc.adapters.rank}) on {len(names)} tensors"
    else:
        rep = select_parameters(pot.model, preset=pc.preset, include=pc.include, exclude=pc.exclude)
        selection = rep.summary()
    nuisance, priors = {}, {}
    if cfg.vibrational.enabled and cfg.vibrational.target_kind == "fundamental_scaled" and cfg.vibrational.fit_scale:
        nuisance["freq_scale"] = torch.nn.Parameter(torch.tensor(cfg.vibrational.scale_prior_mean, dtype=torch.float64))
        priors["freq_scale"] = GaussianPrior(cfg.vibrational.scale_prior_mean, cfg.vibrational.scale_prior_sd)
    return Built(pot, baseline, selection, nuisance, priors)


def load_split(cfg: ExperimentConfig, family: str) -> tuple[ExperimentRegistry, dict]:
    reg = ExperimentRegistry.from_parquet(os.path.join(cfg.data.processed_dir, f"{family}.parquet"))
    with open(cfg.data.splits) as f:
        splits = json.load(f)[family]
    return reg, splits


def _usable(o: Observation, exclude: list[str]) -> bool:
    from .data.curation import usable

    return usable(o, exclude)


def pack_liquid(mol, n_molecules: int, rho_guess: float, seed: int, system_id: str,
                acknowledge_periodic_omol: bool = False):
    """Random-insertion box; flexible/elongated molecules may not pack near the liquid
    density, so pack looser and let NPT equilibration compress."""
    for frac in (0.9, 0.75, 0.6, 0.5):
        try:
            return liquid_box(mol, n_molecules, rho_guess * frac, seed=seed, system_id=system_id,
                              acknowledge_periodic_omol=acknowledge_periodic_omol)
        except RuntimeError:
            log.info("packing %s at %.2f x rho_guess failed; trying looser", system_id, frac)
    raise RuntimeError(f"could not pack {n_molecules} x {system_id}; reduce n_molecules")


def _refuse_test(split: str) -> None:
    if split == "test" and os.environ.get("UMAEXP_UNBLINDING") != "1":
        raise PermissionError("test-split tasks are only built by the unblinding script (DECISIONS.md D-13)")


def vibrational_tasks(cfg: ExperimentConfig, built: Built, split: str = "train") -> list[DirectTask]:
    _refuse_test(split)
    reg, splits = load_split(cfg, "vibrational")
    ids = set(splits[split])
    by_sys: dict[str, list[Observation]] = {}
    for o in reg:
        if o.observation_id in ids and _usable(o, cfg.data.exclude_flags):
            by_sys.setdefault(o.system_ids[0], []).append(o)
    assert_not_reserved([o.groups.molecule_group for obs in by_sys.values() for o in obs], f"vibrational/{split}")
    vc = cfg.vibrational
    tasks = []
    for sid, obs in sorted(by_sys.items()):
        if vc.systems and sid not in vc.systems:
            continue
        smiles = obs[0].labels.get("smiles") or _smiles_from_registry(obs[0])
        system = from_smiles(smiles, system_id=sid, task=cfg.potential.task)
        modes = [ExperimentalMode(label=f"{o.labels['species']}{o.labels['mode_number']}", value=float(o.value),
                                  sigma=float(o.uncertainty.standard_sd()), degeneracy=int(o.labels["degeneracy"]),
                                  species=o.labels["species"]) for o in obs]
        fwd = VibrationalForwardModel(kind=vc.target_kind, scale=built.nuisance.get("freq_scale"),
                                      rel_model_sd=vc.rel_model_sd)
        observable = VibrationalObservable(system, modes, VibrationalConfig(geometry_response=vc.geometry_response,
                                                                            fmax=vc.fmax, max_opt_steps=vc.max_opt_steps,
                                                                            memory_mode=vc.memory_mode, forward=fwd))

        def target_fn(pred, _obs=observable, _ids=[o.observation_id for o in obs]):
            y, var = _obs.targets()
            used = [_ids[i] for i in pred.meta["exp_indices"]]
            return Target(used, "vibrational", y, var.sqrt())

        tasks.append(DirectTask(f"vib:{sid}", "vibrational", observable, target_fn=target_fn))
    return tasks


def _smiles_from_registry(o: Observation) -> str:
    # system_id -> SMILES comes from the pilot definition file
    import yaml

    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pilot = yaml.safe_load(open(os.path.join(here, "data", "pilot", "vibrational_pilot.yaml")))
    table = {m["system_id"]: m["smiles"] for m in pilot["molecules"]}
    return table[o.system_ids[0]]


class RepeatedMeasurement:
    """One model prediction compared with n independent measurements of the same state.

    The sampling error of the prediction is common to all n comparisons, so
    it enters as a fully correlated block (var * 1 1^T), not n independent terms.
    """

    def __init__(self, inner, n: int):
        self.inner, self.n = inner, n

    def predict(self, potential, ens, mode):
        from .observables.base import Prediction

        p = self.inner.predict(potential, ens, mode)
        v = p.value.reshape(-1).repeat(self.n)
        var = p.variance.reshape(()) * torch.ones(self.n, self.n, dtype=torch.float64)
        bf = None
        if p.backward_fn is not None:
            def bf(g, _p=p):
                _p.backward(g.sum().reshape(1))
        return Prediction(v, var, p.valid, p.reason, p.diagnostics, p.meta, bf)


def density_tasks(cfg: ExperimentConfig, built: Built, split: str = "train") -> list[EnsembleTask]:
    _refuse_test(split)
    import numpy as np
    import yaml

    dc = cfg.density
    reg, splits = load_split(cfg, "density")
    ids = set(splits[split])
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    comp = yaml.safe_load(open(os.path.join(here, "data", "pilot", "density_pilot.yaml")))["compounds"]
    smiles_of = {c["system_id"]: c["smiles"] for c in comp.values()}
    tasks = []
    for sid in dc.systems:
        obs = [o for o in reg if o.system_ids[0] == sid and o.observation_id in ids
               and abs((o.conditions.temperature_k or 0) - dc.temperature_k) < 0.5]
        if not obs:
            log.warning("no %s density records for %s at %.2f K", split, sid, dc.temperature_k)
            continue
        assert_not_reserved([o.groups.molecule_group for o in obs], f"density/{split}/{sid}")
        mol = from_smiles(smiles_of[sid], system_id=sid, task=cfg.potential.task)
        # OPEN ISSUE (DECISIONS.md D-15): the starting density uses the measured values, i.e. labels
        # enter the simulation setup; must become a label-free estimate before any density evaluation
        rho0 = float(np.mean([o.value for o in obs]))
        box = pack_liquid(mol, dc.n_molecules, rho0, seed=cfg.seed, system_id=f"{sid}_box{dc.n_molecules}",
                          acknowledge_periodic_omol=dc.acknowledge_periodic_omol)
        state = ThermoState(dc.temperature_k, dc.pressure_bar, "NPT")
        proto = SamplingProtocol(dt_fs=dc.dt_fs, n_equilibration=dc.n_equilibration, n_production=dc.n_production,
                                 stride=dc.stride, seed=cfg.seed, device=cfg.sampling.device, dtype=cfg.sampling.dtype)
        tgt = Target([o.observation_id for o in obs], "density",
                     torch.tensor([o.value for o in obs], dtype=torch.float64),
                     torch.tensor([o.uncertainty.standard_sd() or 1e-3 for o in obs], dtype=torch.float64))

        tasks.append(EnsembleTask(f"density:{sid}", "density", box, state, proto,
                                  RepeatedMeasurement(DensityObservable(box), len(obs)), tgt))
    return tasks
