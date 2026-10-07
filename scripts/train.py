"""Config-driven fitting run with before/after evaluation on train and validation splits.

Test splits are never loaded or scored here (experiments/vib_v1/DECISIONS.md D-13):
held-out data are scored only by a one-shot unblinding script.

  python scripts/train.py configs/vib_pilot.yaml [key=value ...]
  python scripts/train.py configs/vib_pilot.yaml potential.checkpoint=null   # random weights (mechanics)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict

import torch

from umaexp.config import load_config
from umaexp.losses.objective import FamilyWeight
from umaexp.pipeline import build_potential, density_tasks, vibrational_tasks
from umaexp.reweighting.controller import ControllerConfig
from umaexp.training import CacheManager, Trainer, TrainerConfig
from umaexp.validation.evaluation import evaluate_tasks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logging.getLogger("fairchem").setLevel(logging.ERROR)


def replay_set(tasks, n=6, sd=0.03, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for t in tasks:
        sys_ = getattr(t.observable, "system", None) or getattr(t, "system", None)
        if sys_ is None or sys_.positions is None or sys_.is_periodic:
            continue
        x0 = sys_.positions_tensor()
        out.append((sys_, x0 + sd * torch.randn(n, *x0.shape, generator=g, dtype=torch.float64)))
    return out


def main(argv):
    cfg = load_config(argv[0], overrides=argv[1:])
    run_dir = os.path.join(cfg.out_dir, cfg.name, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    built = build_potential(cfg)
    train, val = [], []
    if cfg.vibrational.enabled:
        train += vibrational_tasks(cfg, built, "train")
        val += vibrational_tasks(cfg, built, "val")
    if cfg.density.enabled:
        train += density_tasks(cfg, built, "train")
        val += density_tasks(cfg, built, "val")
    tc = cfg.training
    trainer_cfg = TrainerConfig(
        steps=tc.steps, lr=tc.lr, lr_decay=tc.lr_decay, weight_decay=tc.weight_decay, grad_clip=tc.grad_clip,
        max_relative_update=tc.max_relative_update, lambda_baseline=tc.lambda_baseline,
        lambda_quantum=tc.lambda_quantum, l2sp=tc.l2sp, optimizer=tc.optimizer, lm_lambda=tc.lm_lambda,
        nuisance_lr=tc.nuisance_lr,
        family_weights={k: FamilyWeight(v, "n_obs") for k, v in tc.family_priority.items()},
        controller=ControllerConfig(ess_refresh=tc.ess_refresh, ess_reject=tc.ess_reject),
        checkpoint_dir=os.path.join(run_dir, "checkpoints"), checkpoint_every=tc.checkpoint_every,
        log_path=os.path.join(run_dir, "train_log.jsonl"), seed=cfg.seed)
    cm = CacheManager(os.path.join(run_dir, "caches"), cfg.sampling.device, cfg.sampling.dtype)
    trainer = Trainer(built.potential, train, trainer_cfg, baseline=built.baseline, cache_manager=cm,
                      replay=replay_set(train), nuisance=built.nuisance, priors=built.priors)
    import pandas as pd

    def evaluate(label):
        frames = []
        for split, tasks in (("train", train), ("val", val)):
            if tasks:
                df = evaluate_tasks(built.potential, tasks, label=label)
                df["split"] = split
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    before = evaluate("before")
    before.to_csv(os.path.join(run_dir, "eval_before.csv"), index=False)
    history = trainer.fit()
    after = evaluate("after")
    after.to_csv(os.path.join(run_dir, "eval_after.csv"), index=False)
    summary = {
        "config": asdict(cfg), "selection": built.selection,
        "random_weights": built.potential.provenance.random_init,
        "n_train_tasks": len(train), "n_val_tasks": len(val),
        "steps": len(history),
        "updates": [h.get("update") for h in history],
        "loss": [h["loss"] for h in history],
        "skipped_first_step": history[0]["skipped"] if history else {},
        "rmse_by_split": {
            split: {lab: (float((d[(d.split == split) & d.valid]["residual"] ** 2).mean() ** 0.5)
                          if ((d.split == split) & d.valid).any() else None)
                    for lab, d in (("before", before), ("after", after))}
            for split in ("train", "val")},
        "nuisance": {k: float(v.detach()) for k, v in built.nuisance.items()},
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps({k: summary[k] for k in summary if k != "config"}, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
