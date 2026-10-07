"""Stage-3 equilibrium proof on a system with KNOWN ground truth.

A synthetic "experiment" is generated from a reference Lennard-Jones liquid
(eps*, sigma*). A mis-parameterized model is fitted to (i) NPT density and
(ii) configurational energy per atom, through DiffTRe-style reweighting with
the sampling controller and trust region. Then:
  * parameter recovery is compared with the truth,
  * reweighted training-cache estimates are compared with FRESH simulations,
  * a density-only fit shows the identifiability problem (a ridge of
    (eps, sigma) pairs reproduces one density).
The same trainer/controller/cache code drives UMA; only the potential and
observables change. Numbers are synthetic and demonstrate the machinery.

  python scripts/run_equilibrium_pilot.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from umaexp.losses import FamilyWeight, Target
from umaexp.observables.density import DensityObservable
from umaexp.observables.ensemble import ensemble_prediction
from umaexp.potential.toy import LennardJones, lj_argon_system
from umaexp.reweighting import ControllerConfig, ReweightedEnsemble
from umaexp.sampling import SamplingProtocol, ThermoState, run_sampling
from umaexp.training import CacheManager, EnsembleTask, Trainer, TrainerConfig
from umaexp.validation.evaluation import compare_reweighted_to_fresh


class EnergyPerAtom:
    def __init__(self, n):
        self.n = n

    def predict(self, potential, ens, mode):
        def fn(p, c, U):
            return (U / self.n)[:, None]

        fn.uses_energy = True
        return ensemble_prediction(ens, potential, fn, None, mode, meta={"units": "eV/atom"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="runs/equilibrium_pilot")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lr-decay", type=float, default=0.9)
    ap.add_argument("--trust", type=float, default=0.01)
    ap.add_argument("--only", choices=["joint", "density"], default=None)
    ap.add_argument("--optimizer", choices=["adamw", "lm"], default="adamw")
    ap.add_argument("--lm-lambda", type=float, default=1e-3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    system = lj_argon_system(n_side=5, density_g_cm3=1.35)
    cutoff = 8.0
    # 85 K / 50 bar: dense liquid for shifted-force LJ (rc = 8 A); measured
    # volume statistical inefficiency ~25-45 frames (physical density relaxation)
    state = ThermoState(85.0, 50.0, "NPT")
    n_prod = 12000 if a.quick else 24000
    proto = SamplingProtocol(dt_fs=10.0, friction_per_ps=2.0, n_equilibration=3000, n_production=n_prod, stride=20,
                             barostat_every=10, barostat_max_dlnv=0.01, seed=1)

    # 1) synthetic experiment from the reference potential
    truth = LennardJones(epsilon=0.0104, sigma=3.40, cutoff=cutoff)
    ref_cache = run_sampling(truth, system, state, SamplingProtocol(**{**proto.__dict__, "n_production": 2 * n_prod,
                                                                       "seed": 99}))
    ens = ReweightedEnsemble(ref_cache, system)
    rho_true = float(DensityObservable(system).predict(truth, ens, "surrogate").value)
    u_true = float(EnergyPerAtom(system.n_atoms).predict(truth, ens, "surrogate").value)
    rng = np.random.default_rng(7)
    # synthetic measurement noise at the resolution a 125-atom box can match
    rho_exp = rho_true * (1 + 0.002 * rng.normal())
    u_exp = u_true + 0.0003 * rng.normal()
    targets = {"rho_true": rho_true, "u_true": u_true, "rho_exp": rho_exp, "u_exp": u_exp}

    def tasks():
        return [
            EnsembleTask("density", "density", system, state, proto, DensityObservable(system),
                         Target(["rho"], "density", torch.tensor([rho_exp], dtype=torch.float64),
                                torch.tensor([0.003 * rho_exp], dtype=torch.float64))),
            EnsembleTask("energy", "thermochemistry", system, state, proto, EnergyPerAtom(system.n_atoms),
                         Target(["u"], "thermochemistry", torch.tensor([u_exp], dtype=torch.float64),
                                torch.tensor([0.0005], dtype=torch.float64))),
        ]

    results = {"targets": targets}
    runs = [("joint_density_energy", ["density", "energy"]), ("density_only", ["density"])]
    if a.only:
        runs = [r for r in runs if r[0].startswith(a.only)]
    for label, use in runs:
        model = LennardJones(epsilon=0.0104 * 0.85, sigma=3.40 * 1.03, cutoff=cutoff)
        ts = [t for t in tasks() if t.name in use]
        cfg = TrainerConfig(steps=a.steps or (10 if a.quick else 20), lr=0.03, lr_decay=a.lr_decay, grad_clip=None,
                            max_relative_update=a.trust, trust_region_norm_floor=1.0,
                            family_weights={"density": FamilyWeight(1.0), "thermochemistry": FamilyWeight(1.0)},
                            controller=ControllerConfig(ess_refresh=0.5, ess_reject=0.15, min_effective_samples=15,
                                                        sem_over_sigma_extend=3.0),
                            max_sampling_attempts=2,
                            optimizer=a.optimizer, lm_lambda=a.lm_lambda,
                            log_path=os.path.join(a.out, f"{label}.jsonl"))
        cm = CacheManager()
        hist = Trainer(model, ts, cfg, cache_manager=cm).fit()
        eps, sig = float(model.log_eps.exp()), float(model.log_sigma.exp())
        gates = {t.name: compare_reweighted_to_fresh(model, t, cm.caches[t.name]) for t in ts}
        results[label] = {
            "eps_eV": eps, "sigma_A": sig, "eps_true": 0.0104, "sigma_true": 3.40,
            "eps_rel_err": eps / 0.0104 - 1, "sigma_rel_err": sig / 3.40 - 1,
            "loss_first_last": [hist[0]["loss"], hist[-1]["loss"]],
            "updates": [h.get("update") for h in hist],
            "cache_events": [(e.action, e.reason) for e in cm.events],
            "fresh_vs_reweighted": gates,
        }
    results["wall_s"] = time.time() - t0
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
