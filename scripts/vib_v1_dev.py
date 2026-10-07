"""Vib-v1 development driver (Dev data only; see experiments/vib_v1/PROTOCOL.md).

  python scripts/vib_v1_dev.py precompute [--only H2O,NH3]   # base-model GVPT2 + S_m per dev molecule
  python scripts/vib_v1_dev.py summary                        # structure of the precomputed scored sets
  python scripts/vib_v1_dev.py cv --folds 0,1,2 --caps 3e-4,1e-3,3e-3 --steps 60 --eval-every 10 --tag grid
      grouped K-fold CV on Dev (PROTOCOL 6): train FT on the other folds, score the held-out fold every N steps
  python scripts/vib_v1_dev.py select --tag grid      # one-standard-error rule; power estimate (no decision)
  python scripts/vib_v1_dev.py final --cap 1e-3 --steps 40
      train FT on ALL Dev with the selected settings (epoch-matched steps); save trainable tensors
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import time
from collections import Counter

import copy

import numpy as np
import torch

from umaexp.losses.objective import FamilyWeight
from umaexp.potential import UMAPotential, select_parameters
from umaexp.training import Trainer, TrainerConfig
from umaexp.vib_v1 import ROOT, base_precompute, load_dev_molecules, make_task, save_precompute

logging.getLogger("fairchem").setLevel(logging.ERROR)
OUT = os.path.join(ROOT, "runs", "vib_v1", "base")


def precompute(only: set[str] | None):
    pot = UMAPotential.from_checkpoint("uma-s-1p1", task="omol", dtype=torch.float64)
    mols, skipped = load_dev_molecules()
    print(f"dev molecules: {len(mols)} (skipped: {skipped})", flush=True)
    for m in mols:
        if only and m.system_id not in only:
            continue
        t = time.time()
        rec = base_precompute(pot, m)
        rec["provenance"] = {"checkpoint": pot._checkpoint_path, "sha256": pot.provenance.checkpoint_sha256,
                             "initial_geometry": m.system.provenance.get("initial_geometry", "rdkit")}
        save_precompute(rec, OUT)
        print(f"{m.system_id:10s} conv={rec['converged']} imag={rec['n_imaginary']} pg_ok={rec['point_group_ok']} "
              f"merged={rec['merged_classes']} scored={rec['n_scored']}/{len(rec['modes'])} "
              f"({time.time() - t:.0f}s)", flush=True)


def summary():
    reasons, rows = Counter(), []
    for f in sorted(glob.glob(os.path.join(OUT, "*.json"))):
        r = json.load(open(f))
        for m in r["modes"]:
            reasons[m["excluded_because"] or "scored"] += 1
        rows.append((r["system_id"], r["n_scored"], len(r["modes"]), r["merged_classes"], r["notes"]))
    for row in rows:
        print(*row)
    print(dict(reasons))


def load_pre() -> dict:
    return {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(OUT, "*.json"))}


def make_folds(mols, k: int = 3, seed: int = 20261006):
    groups = sorted({m.group for m in mols})
    perm = np.random.default_rng(seed).permutation(len(groups))
    fold_of = {groups[g]: i % k for i, g in enumerate(perm)}
    return [[m for m in mols if fold_of[m.group] == f] for f in range(k)]


def base_errors(pre: dict) -> dict:
    """B-VPT2 per-molecule RMS error over S_m (from the precompute; nothing fitted)."""
    out = {}
    for sid, r in pre.items():
        res = [m["omega_base"] + m["delta"] - m["experiment"] for m in r["modes"] if m["scored"]]
        if res:
            out[sid] = float(np.sqrt(np.mean(np.square(res))))
    return out


def model_errors(pot, mols, pre) -> dict:
    """Per-molecule RMS error over S_m for a model; fresh symmetry assignment on its own modes."""
    from umaexp.vib_v1 import evaluate_anchored

    out = {}
    for m in mols:
        p = pre[m.system_id]
        if p["n_scored"] == 0:
            continue
        got, _ = evaluate_anchored(pot, m, p)
        if got is None:
            out[m.system_id] = None
            continue
        res = [got[x["label"]] - x["experiment"] for x in p["modes"] if x["scored"]]
        out[m.system_id] = float(np.sqrt(np.mean(np.square(res))))
    return out


def replay_set(mols, pre, n=6, sd=0.03, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for m in mols:
        x0 = torch.tensor(pre[m.system_id]["geometry"], dtype=torch.float64)
        out.append((m.system, x0 + sd * torch.randn(n, *x0.shape, generator=g, dtype=torch.float64)))
    return out


LR = 1e-3  # Adam; the trust-region cap sets the step (DECISIONS D-18)


def cv(folds_sel, caps, steps, eval_every, batch, tag, save_ft: bool = False):
    base = UMAPotential.from_checkpoint("uma-s-1p1", task="omol", dtype=torch.float64)
    mols, _ = load_dev_molecules()
    pre = load_pre()
    mols = [m for m in mols if m.system_id in pre and pre[m.system_id]["n_scored"] > 0]
    folds = make_folds(mols)
    b_err = base_errors(pre)
    out_dir = os.path.join(ROOT, "runs", "vib_v1", "cv", tag)
    os.makedirs(out_dir, exist_ok=True)
    for f in folds_sel:
        held = folds[f]
        train = [m for k, fo in enumerate(folds) if k != f for m in fo]
        for cap in caps:
            pot = copy.deepcopy(base)
            select_parameters(pot.model, preset="readout")
            tasks = [make_task(m, pre[m.system_id]) for m in train]
            tc = TrainerConfig(steps=steps, lr=LR, batch_size=batch, l2sp=1.0, lambda_baseline=1.0,
                               max_relative_update=cap, seed=20261006, checkpoint_dir=None,
                               family_weights={"vibrational": FamilyWeight(1.0, "n_targets")},  # P5: per molecule
                               log_path=os.path.join(out_dir, f"fold{f}_cap{cap:g}_train.jsonl"))
            trainer = Trainer(pot, tasks, tc, baseline=base, replay=replay_set(train, pre))
            trace = [{"step": 0, "errors": {m.system_id: b_err.get(m.system_id) for m in held}}]
            t0 = time.time()
            for s_ in range(steps):
                rec = trainer.step(s_)
                print(f"fold {f} cap {cap:g} step {s_ + 1} loss {rec['loss']:.4f} update {rec.get('update')} "
                      f"rel_step {rec.get('applied_rel_step', float('nan')):.2e} ({time.time() - t0:.0f}s)", flush=True)
                if (s_ + 1) % eval_every == 0:  # held-out errors go to the trace file only (not printed)
                    trace.append({"step": s_ + 1, "errors": model_errors(pot, held, pre)})
            with open(os.path.join(out_dir, f"fold{f}_cap{cap:g}.json"), "w") as fh:
                json.dump({"fold": f, "cap": cap, "lr": LR, "steps": steps, "batch": batch,
                           "held_out": [m.system_id for m in held],
                           "train": [m.system_id for m in train], "trace": trace,
                           "wall_s": time.time() - t0}, fh, indent=1)
            if save_ft:  # e.g. for the PROTOCOL 6 real-path dress rehearsal (trainable tensors only)
                from umaexp.training.checkpointing import save_training_state

                save_training_state(os.path.join(out_dir, f"fold{f}_cap{cap:g}_ft.pt"), pot, None, steps,
                                    {"fold": f, "cap": cap, "steps": steps})
            del pot, trainer


def select(tag):
    """PROTOCOL 6: CV mean of e_m over all held-out molecules; one-standard-error rule (smallest cap, then
    fewest steps within one SE of the best). Invalid held-out predictions count with the configuration's
    worst valid e_m (as in the primary analysis). Writes experiments/vib_v1/cv_selection.json."""
    files = sorted(glob.glob(os.path.join(ROOT, "runs", "vib_v1", "cv", tag, "fold*_cap*.json")))
    runs = [json.load(open(f)) for f in files if not f.endswith("_train.jsonl")]
    b_err = base_errors(load_pre())
    table = {}
    for r in runs:
        for t in r["trace"][1:]:
            table.setdefault((r["cap"], t["step"]), {}).update(t["errors"])
    keys = sorted({k for errs in table.values() for k in errs})

    def imputed(errs):  # critic report #3 P3: invalid -> max(e_base, worst valid e) (never an improvement)
        valid = [v for v in errs.values() if v is not None]
        worst = max(valid) if valid else float("nan")
        return np.array([errs[k] if errs.get(k) is not None else max(b_err.get(k, worst), worst) for k in keys])

    stats = []
    for (cap, step), errs in sorted(table.items()):
        e = imputed(errs)
        d = e - np.array([b_err[k] for k in keys])
        stats.append({"cap": cap, "step": step, "n": len(e), "mean_e": float(e.mean()), "e": e.tolist(),
                      "mean_delta_vs_base": float(d.mean()), "sd_delta_vs_base": float(d.std(ddof=1)),
                      "n_invalid": int(sum(v is None for v in errs.values()))})
    best = min(stats, key=lambda r: r["mean_e"])
    eb = np.array(best["e"])
    for r in stats:  # one-SE rule on PAIRED differences to the best configuration (critic report #3 P4)
        dd = np.array(r["e"]) - eb
        r["mean_diff_to_best"] = float(dd.mean())
        r["se_diff_to_best"] = float(dd.std(ddof=1) / np.sqrt(len(dd))) if len(dd) > 1 else 0.0
    pick = min([r for r in stats if r["mean_diff_to_best"] <= r["se_diff_to_best"]],
               key=lambda r: (r["cap"], r["step"]))
    n_dev = len({k for r in runs for k in r["held_out"]})
    n_train = float(np.mean([len(r["train"]) for r in runs]))
    out = {"stats": [{k: v for k, v in r.items() if k != "e"} for r in stats],
           "best": {k: v for k, v in best.items() if k != "e"}, "selected": {k: v for k, v in pick.items() if k != "e"},
           "rule": "one-standard-error on paired per-molecule differences to the best configuration",
           "final_steps_epoch_matched": int(round(pick["step"] * n_dev / n_train)), "n_dev": n_dev,
           "mean_train_size": n_train,
           "power_note": "informational only; no decision depends on it (PROTOCOL 6)",
           "power_sd_delta_at_selected": pick["sd_delta_vs_base"]}
    with open(os.path.join(ROOT, "experiments", "vib_v1", "cv_selection.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in ("selected", "final_steps_epoch_matched")}, indent=1))


def final(cap, steps, batch):
    from umaexp.training.checkpointing import save_training_state
    from umaexp.vib_v1 import drift

    base = UMAPotential.from_checkpoint("uma-s-1p1", task="omol", dtype=torch.float64)
    mols, _ = load_dev_molecules()
    pre = load_pre()
    mols = [m for m in mols if m.system_id in pre and pre[m.system_id]["n_scored"] > 0]
    out_dir = os.path.join(ROOT, "runs", "vib_v1", "final")
    os.makedirs(out_dir, exist_ok=True)
    pot = copy.deepcopy(base)
    select_parameters(pot.model, preset="readout")
    tasks = [make_task(m, pre[m.system_id]) for m in mols]
    tc = TrainerConfig(steps=steps, lr=LR, batch_size=batch, l2sp=1.0, lambda_baseline=1.0, max_relative_update=cap,
                       family_weights={"vibrational": FamilyWeight(1.0, "n_targets")}, seed=20261006,
                       checkpoint_dir=None, log_path=os.path.join(out_dir, "train.jsonl"))
    trainer = Trainer(pot, tasks, tc, baseline=base, replay=replay_set(mols, pre))
    for s_ in range(steps):
        rec = trainer.step(s_)
        print(f"final step {s_ + 1} loss {rec['loss']:.4f} update {rec.get('update')}", flush=True)
    ckpt = save_training_state(os.path.join(out_dir, "ft_readout.pt"), pot, None, steps,
                               {"lr": LR, "cap": cap, "steps": steps, "batch": batch,
                                "dev_molecules": [m.system_id for m in mols]})
    summary = {"checkpoint": os.path.relpath(ckpt, ROOT), "lr": LR, "cap": cap, "steps": steps, "batch": batch,
               "n_dev_molecules": len(mols), "dev_drift": [drift(base, pot, m, pre[m.system_id]) for m in mols],
               "in_sample_dev_errors_ft": model_errors(pot, mols, pre), "dev_errors_base": base_errors(pre)}
    with open(os.path.join(ROOT, "experiments", "vib_v1", "final_dev_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "dev_drift"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["precompute", "summary", "cv", "select", "final"])
    ap.add_argument("--cap", type=float, default=1e-3)
    ap.add_argument("--caps", default="3e-4,1e-3,3e-3")
    ap.add_argument("--only", default=None)
    ap.add_argument("--folds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=60)  # PROTOCOL 6
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--save-ft", action="store_true")
    a = ap.parse_args()
    if a.cmd == "precompute":
        precompute(set(a.only.split(",")) if a.only else None)
    elif a.cmd == "summary":
        summary()
    elif a.cmd == "final":
        final(a.cap, a.steps, a.batch)
    elif a.cmd == "select":
        select(a.tag)
    else:
        cv([int(x) for x in a.folds.split(",")], [float(x) for x in a.caps.split(",")], a.steps, a.eval_every,
           a.batch, a.tag, a.save_ft)
