"""Unchanged-UMA baseline: harmonic frequencies at UMA minima vs NIST fundamentals.

For every pilot molecule (all splits): optimize with UMA, compute the harmonic
spectrum, assign modes (degeneracy-sorted), and compare with the curated
NIST/Shimanouchi fundamentals (flagged modes excluded as in training). A single
global scale s (nu ~ s * omega) is fitted on the TRAIN split only and applied
to val. The test split is never scored here (DECISIONS.md D-13).

  python scripts/baseline_vibrations.py [--checkpoint uma-s-1p1] [--out runs/baseline_vib]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
import pandas as pd
import torch

from umaexp.config import load_config
from umaexp.observables.vibrational import assign_modes, harmonic_modes, optimize_geometry
from umaexp.pipeline import Built, vibrational_tasks
from umaexp.potential import UMAPotential

logging.getLogger("fairchem").setLevel(logging.ERROR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="uma-s-1p1")
    ap.add_argument("--config", default="configs/vib_pilot.yaml")
    ap.add_argument("--out", default="runs/baseline_vib")
    ap.add_argument("--fmax", type=float, default=1e-5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = load_config(a.config)
    pot = UMAPotential.from_checkpoint(a.checkpoint, task="omol", dtype=torch.float64)
    built = Built(pot, None, "", {}, {})
    rows, mols = [], []
    for split in ("train", "val"):
        for task in vibrational_tasks(cfg, built, split):
            obs = task.observable
            t0 = time.time()
            x, info = optimize_geometry(pot, obs.system, fmax=a.fmax)
            nm = harmonic_modes(pot, obs.system, x, create_graph=False)
            nu = nm.wavenumbers.detach().numpy()
            matches = assign_modes(nu, obs.exp_modes)
            n_imag = int((nu < 0).sum())
            mols.append({"system": obs.system.system_id, "split": split, "converged": info["converged"],
                         "max_force": info["max_force"], "n_imaginary": n_imag,
                         "all_harmonic_cm": [round(float(v), 1) for v in sorted(nu)], "wall_s": time.time() - t0})
            for m in matches:
                em = obs.exp_modes[m.exp_index]
                rows.append({"system": obs.system.system_id, "split": split, "mode": em.label,
                             "degeneracy": em.degeneracy, "nist_fundamental": em.value,
                             "uma_harmonic": float(np.mean(nu[m.comp_indices])),
                             "ambiguous": m.ambiguous, "note": m.note})
            print(f"{split:5s} {obs.system.system_id:6s} conv={info['converged']} max|F|={info['max_force']:.1e} "
                  f"imag={n_imag} modes={len(matches)} ({time.time() - t0:.1f}s)", flush=True)
    df = pd.DataFrame(rows)
    df["ratio"] = df["nist_fundamental"] / df["uma_harmonic"]
    tr = df[df.split == "train"]
    s = float((tr.nist_fundamental * tr.uma_harmonic).sum() / (tr.uma_harmonic**2).sum())  # LSQ scale
    df["scaled"] = s * df["uma_harmonic"]
    df["residual"] = df["scaled"] - df["nist_fundamental"]
    summary = {"checkpoint": a.checkpoint, "fitted_scale_train": s, "n_modes": int(len(df))}
    for split in ("train", "val"):
        d = df[df.split == split]
        if len(d):
            summary[split] = {
                "n_modes": int(len(d)),
                "rmse_scaled_cm": float(np.sqrt((d.residual**2).mean())),
                "mae_scaled_cm": float(d.residual.abs().mean()),
                "max_abs_scaled_cm": float(d.residual.abs().max()),
                "mean_rel_harmonic_minus_fund": float(((d.uma_harmonic - d.nist_fundamental) / d.nist_fundamental).mean()),
            }
    df.to_csv(os.path.join(a.out, "modes.csv"), index=False)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump({"summary": summary, "molecules": mols}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
