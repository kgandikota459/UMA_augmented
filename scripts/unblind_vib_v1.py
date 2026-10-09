"""One-shot scoring of the sealed Vib-v1 held-out set (experiments/vib_v1/PROTOCOL.md section 7).

  python scripts/unblind_vib_v1.py --rehearse [--ft path]   # identical code on DEVELOPMENT molecules;
                                                            # never reads data/sealed (dry run of the analysis)
  python scripts/unblind_vib_v1.py                          # the single real run (requires FROZEN.json)
  python scripts/unblind_vib_v1.py --bugfix-reason "..."    # re-run only for a bug fix recorded in DEVIATIONS.md

The real run refuses to start unless every digest in FROZEN.json still matches
(protocol, decision log, code tree, data, checkpoint) and no lock file exists.
It is the first time any model output is computed for a held-out molecule.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import glob
import json
import logging
import os
import sys
from collections import defaultdict

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from freeze_vib_v1 import EXP, FROZEN, ROOT, digests  # noqa: E402

from umaexp.data import ExperimentRegistry  # noqa: E402
from umaexp.potential import UMAPotential, select_parameters  # noqa: E402
from umaexp.vib_v1 import (  # noqa: E402
    Molecule,
    _system_for,
    analyse,
    load_dev_molecules,
    lsq_scale,
    safe_precompute,
    score_molecule,
)

logging.getLogger("fairchem").setLevel(logging.ERROR)
SEALED = os.path.join(os.environ.get("UMAEXP_SEALED_DIR", os.path.join(os.path.expanduser("~"), ".umaexp_sealed",
                                                                         "vib_v1")), "build")
LOCK = os.path.join(EXP, "results", "UNBLINDED.lock")
DEV_BASE = os.path.join(ROOT, "runs", "vib_v1", "base")


def load_ft(base, path: str | None):
    ft = copy.deepcopy(base)
    select_parameters(ft.model, preset="readout")
    if path:
        state = torch.load(path, map_location="cpu", weights_only=False)
        named = dict(ft.named_parameters())
        with torch.no_grad():
            for n, v in state["trainable"].items():
                named[n].copy_(v.to(named[n].dtype))
    return ft


def sealed_molecules() -> list[Molecule]:
    by = defaultdict(list)
    for o in ExperimentRegistry.from_parquet(os.path.join(SEALED, "test.parquet")):
        by[o.system_ids[0]].append(o)
    mols = []
    for sid, obs in sorted(by.items()):
        obs = sorted(obs, key=lambda o: o.labels["mode_number"])
        mols.append(Molecule(sid, _system_for(sid, obs[0].labels["smiles"], obs[0].labels.get("g2_name")), obs,
                             obs[0].labels.get("point_group") or "", obs[0].groups.molecule_group))
    return mols


def verify_frozen(bugfix: bool) -> dict:
    """Every frozen digest must match; a documented bug-fix re-run may differ in the code tree only."""
    if not os.path.exists(FROZEN):
        raise SystemExit("not frozen: run scripts/freeze_vib_v1.py first")
    frozen = json.load(open(FROZEN))
    now = digests(os.path.join(ROOT, frozen["checkpoint"]["path"]))
    bad = [k for k in now if now[k] != frozen.get(k) and not (bugfix and k == "code_files")]
    if bad:
        raise SystemExit(f"digest mismatch since freezing: {bad} (re-freezing is not allowed)")
    return frozen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehearse", action="store_true")
    ap.add_argument("--ft", default=None, help="rehearsal only: FT checkpoint (default: FT = base)")
    ap.add_argument("--recompute", action="store_true",
                    help="rehearsal only: recompute the base precompute exactly as unblinding does")
    ap.add_argument("--bugfix-reason", default=None)
    a = ap.parse_args()
    dev_pre = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(DEV_BASE, "*.json"))}
    base = UMAPotential.from_checkpoint("uma-s-1p1", task="omol", dtype=torch.float64)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if a.rehearse:
        out = os.path.join(ROOT, "runs", "vib_v1", "rehearsal", stamp)
        mols = [m for m in load_dev_molecules()[0] if m.system_id in dev_pre]
        ft = load_ft(base, a.ft)
        pre = ({m.system_id: safe_precompute(base, m) for m in mols} if a.recompute
               else {m.system_id: dev_pre[m.system_id] for m in mols})
    else:
        frozen = verify_frozen(bugfix=a.bugfix_reason is not None)
        if os.path.exists(LOCK) and not a.bugfix_reason:
            raise SystemExit("already unblinded; a re-run needs --bugfix-reason recorded in DEVIATIONS.md")
        if a.bugfix_reason:
            dev_path = os.path.join(EXP, "DEVIATIONS.md")
            if not os.path.exists(dev_path) or a.bugfix_reason not in open(dev_path).read():
                raise SystemExit("the bug-fix reason must be recorded verbatim in DEVIATIONS.md first")
        os.environ["UMAEXP_UNBLINDING"] = "1"
        out = os.path.join(EXP, "results", stamp)
        os.makedirs(os.path.dirname(LOCK), exist_ok=True)
        with open(LOCK, "a") as f:  # written BEFORE any held-out computation (critic report #3 P7c)
            f.write(f"{stamp} started {a.bugfix_reason or 'first run'}\n")
        mols = sealed_molecules()
        ft = load_ft(base, os.path.join(ROOT, frozen["checkpoint"]["path"]))
        pre = {m.system_id: safe_precompute(base, m) for m in mols}
    os.makedirs(out, exist_ok=True)
    scale = lsq_scale(dev_pre)
    rows, drifts, excluded = [], [], {}
    for m in mols:  # exception rule: PROTOCOL 5 (only exception types are recorded)
        p = pre[m.system_id]
        if p.get("error"):
            excluded[m.system_id] = f"base precompute exception: {p['error']}"
            continue
        if p["n_scored"] == 0:
            excluded[m.system_id] = "empty scored set"
            continue
        r, d, err = score_molecule(base, ft, m, p, scale)
        if err:
            excluded[m.system_id] = f"base scoring exception: {err}"
            continue
        rows += r
        drifts.append(d)
        print(f"scored molecule {len(drifts)}", flush=True)
    result = analyse(rows, drifts)
    result.update({"excluded_molecules": excluded, "n_excluded": len(excluded),
                   "mode": "rehearsal" if a.rehearse else "unblinded", "b_scale": scale, "n_scored_modes":
                   int(sum(r.get("model") == "FT" and "prediction" in r for r in rows))})
    pd.DataFrame(rows).to_csv(os.path.join(out, "mode_predictions.csv"), index=False)
    pd.DataFrame(drifts).to_csv(os.path.join(out, "drift.csv"), index=False)
    with open(os.path.join(out, "precompute.json"), "w") as f:
        json.dump(pre, f)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    if not a.rehearse:
        with open(LOCK, "a") as f:
            f.write(f"{stamp} finished\n")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
