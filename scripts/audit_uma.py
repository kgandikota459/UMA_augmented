"""Stage-0 capability audit.

  python scripts/audit_uma.py                      # random UMA-S architecture (offline)
  python scripts/audit_uma.py --checkpoint uma-s-1p1 --task omol   # gated checkpoint (HF token)
"""

import argparse
import json
import logging

from umaexp.validation.audit import run_audit

logging.getLogger("fairchem").setLevel(logging.ERROR)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="registry name (e.g. uma-s-1p1) or .pt path")
    ap.add_argument("--task", default="omol")
    ap.add_argument("--out", default="runs/audit")
    ap.add_argument("--no-benchmark", action="store_true")
    a = ap.parse_args()
    rep = run_audit(a.checkpoint, a.task, a.out, benchmark=not a.no_benchmark)
    summary = {k: rep[k] for k in ("random_architecture", "parity", "derivative_checks", "passed", "wall_seconds")}
    summary["presets"] = rep["presets"]
    summary["device_benchmark"] = rep.get("device_benchmark")
    print(json.dumps(summary, indent=2, default=str))
