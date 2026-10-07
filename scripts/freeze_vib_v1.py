"""Staged freezes for Vib-v1 (PROTOCOL.md sections 2.4 and 7).

  python scripts/freeze_vib_v1.py section2 --critic-signoff "..."   # before the network fetch
  python scripts/freeze_vib_v1.py protocol --critic-signoff "..."   # before D_ext is unsealed
  python scripts/freeze_vib_v1.py model --checkpoint runs/vib_v1/final/ft_readout.pt --critic-signoff "..."
                                                                    # before unblinding

Each stage records SHA-256 digests and refuses to overwrite itself. Later stages verify the
earlier ones:
  * S2 covers section 2 of PROTOCOL.md, the builder's full import closure (computed by importing
    it in a fresh interpreter), this script, the pilot list and the reserved list;
  * PROTOCOL covers the whole protocol, per-file digests of src/ scripts/ configs/ and an archive
    of that tree (for diffs); DECISIONS.md is append-only from here;
  * MODEL requires every code file changed since PROTOCOL to be named in DEVIATIONS.md, writes
    their diff, and digests the data, the sealed build, the Dev precompute, the CV selection, the
    final summary, the base UMA checkpoint and the FT checkpoint.
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments", "vib_v1")
FROZEN_S2 = os.path.join(EXP, "FROZEN_S2.json")
FROZEN_PROTOCOL = os.path.join(EXP, "FROZEN_PROTOCOL.json")
FROZEN = os.path.join(EXP, "FROZEN.json")
CODE_ARCHIVE = os.path.join(EXP, "frozen_protocol_code.tar.gz")
SEALED_BUILD = os.path.join(os.environ.get("UMAEXP_SEALED_DIR",
                                           os.path.join(os.path.expanduser("~"), ".umaexp_sealed", "vib_v1")), "build")
S2_EXTRA = ["scripts/build_vib_heldout.py", "scripts/freeze_vib_v1.py", "data/pilot/vibrational_pilot.yaml",
            "experiments/vib_v1/reserved_blocks.json"]


def sha_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def section(md: str, number: str) -> str:
    """Text of '## <number>.' up to the next '## ' heading."""
    m = re.search(rf"^## {re.escape(number)}\..*?(?=^## |\Z)", md, re.S | re.M)
    if not m:
        raise SystemExit(f"section {number} not found in PROTOCOL.md")
    return m.group(0)


def code_files(dirs=("src", "scripts", "configs")) -> dict:
    out = {}
    for d in dirs:
        for base, _, names in os.walk(os.path.join(ROOT, d)):
            if "__pycache__" in base:
                continue
            for n in names:
                if not n.endswith(".pyc"):
                    p = os.path.join(base, n)
                    out[os.path.relpath(p, ROOT)] = sha_file(p)
    return dict(sorted(out.items()))


def s2_import_closure() -> list[str]:
    """Every umaexp module the builder imports (module level and inside its functions)."""
    code = ("import importlib.util, sys; sys.path.insert(0, 'scripts'); "
            "spec = importlib.util.spec_from_file_location('b', 'scripts/build_vib_heldout.py'); "
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
            "import umaexp.data.pointgroup, umaexp.data.curation; "
            "print('\\n'.join(sorted(v.__file__ for k, v in list(sys.modules.items()) "
            "if k.startswith('umaexp') and getattr(v, '__file__', None))))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return sorted({os.path.relpath(f, ROOT) for f in out.split()} | set(S2_EXTRA))


def s2_digests() -> dict:
    md = open(os.path.join(EXP, "PROTOCOL.md"), encoding="utf-8").read()
    return {"protocol_section2": sha_text(section(md, "2")),
            "files": {f: sha_file(os.path.join(ROOT, f)) for f in s2_import_closure()}}


def verify_s2(allow_documented: bool = False) -> list[str]:
    """Section 2 text must be unchanged (always). S2 code files must be unchanged at fetch time (strict);
    at later stages a changed file is accepted only if named in DEVIATIONS.md (the sealed build is
    already made, and its build-time code digests sit in heldout_public.json for audit)."""
    if not os.path.exists(FROZEN_S2):
        raise SystemExit("section 2 is not frozen (scripts/freeze_vib_v1.py section2)")
    frozen = json.load(open(FROZEN_S2))
    now = s2_digests()
    if now["protocol_section2"] != frozen["protocol_section2"]:
        raise SystemExit("PROTOCOL.md section 2 changed since the S2 freeze")
    changed = sorted(f for f in set(now["files"]) | set(frozen["files"]) if now["files"].get(f) != frozen["files"].get(f))
    if changed and not allow_documented:
        raise SystemExit(f"section-2 code changed since freezing: {changed}")
    dev_path = os.path.join(EXP, "DEVIATIONS.md")
    devtext = open(dev_path, encoding="utf-8").read() if os.path.exists(dev_path) else ""
    undocumented = [f for f in changed if f not in devtext]
    if undocumented:
        raise SystemExit(f"section-2 code changed without a DEVIATIONS.md entry: {undocumented}")
    return changed


def verify_protocol() -> tuple[dict, list[str]]:
    """Protocol text unchanged, DECISIONS append-only, every changed code file named in DEVIATIONS.md."""
    if not os.path.exists(FROZEN_PROTOCOL):
        raise SystemExit("protocol is not frozen (scripts/freeze_vib_v1.py protocol)")
    frozen = json.load(open(FROZEN_PROTOCOL))
    if sha_file(os.path.join(EXP, "PROTOCOL.md")) != frozen["protocol"]:
        raise SystemExit("PROTOCOL.md changed after the protocol freeze; record a deviation instead")
    if not open(os.path.join(EXP, "DECISIONS.md"), encoding="utf-8").read().startswith(frozen["decisions_text"]):
        raise SystemExit("DECISIONS.md is append-only after the protocol freeze")
    now = code_files()
    changed = sorted(f for f in set(now) | set(frozen["code_files"]) if now.get(f) != frozen["code_files"].get(f))
    dev_path = os.path.join(EXP, "DEVIATIONS.md")
    devtext = open(dev_path, encoding="utf-8").read() if os.path.exists(dev_path) else ""
    undocumented = [f for f in changed if f not in devtext]
    if undocumented:
        raise SystemExit(f"code changed since the protocol freeze without a DEVIATIONS.md entry: {undocumented}")
    return frozen, changed


def write_code_diff(changed: list[str]) -> str:
    out = os.path.join(EXP, "code_changes_since_protocol.diff")
    old = {}
    with tarfile.open(CODE_ARCHIVE) as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name in changed:
                old[m.name] = tar.extractfile(m).read().decode("utf-8", errors="replace").splitlines(keepends=True)
    with open(out, "w") as f:
        for path in changed:
            new = open(os.path.join(ROOT, path), encoding="utf-8").read().splitlines(keepends=True) \
                if os.path.exists(os.path.join(ROOT, path)) else []
            f.writelines(difflib.unified_diff(old.get(path, []), new, f"protocol/{path}", f"model/{path}"))
    return out


def base_checkpoint() -> str:
    hits = glob.glob(os.path.expanduser("~/.cache/fairchem/models--facebook--UMA/snapshots/*/checkpoints/uma-s-1p1.pt"))
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one cached uma-s-1p1 checkpoint, found {len(hits)}")
    return hits[0]


def digests(checkpoint: str) -> dict:
    """Model-stage digests (verified again by scripts/unblind_vib_v1.py before anything is computed)."""
    data = {os.path.relpath(p, ROOT): sha_file(p) for p in [
        os.path.join(ROOT, "data", "processed", "vib_v1_dev_pilot.parquet"),
        os.path.join(ROOT, "data", "processed", "vib_v1_dev_ext.parquet")]}
    data.update({"sealed/" + n: sha_file(os.path.join(SEALED_BUILD, n))
                 for n in ("test.parquet", "test_molecules.json", "MANIFEST.json")})
    pre = {os.path.basename(p): sha_file(p) for p in sorted(glob.glob(os.path.join(ROOT, "runs", "vib_v1", "base", "*.json")))}
    return {"protocol": sha_file(os.path.join(EXP, "PROTOCOL.md")),
            "final_dev_summary": sha_file(os.path.join(EXP, "final_dev_summary.json")),
            "cv_selection": sha_file(os.path.join(EXP, "cv_selection.json")),
            "code_files": code_files(), "data": data, "dev_precompute": pre,
            "base_checkpoint_sha256": sha_file(base_checkpoint()),
            "checkpoint": {"path": os.path.relpath(checkpoint, ROOT), "sha256": sha_file(checkpoint)}}


def _write(path: str, rec: dict) -> None:
    if os.path.exists(path):
        raise SystemExit(f"{os.path.relpath(path, ROOT)} exists; record changes in DEVIATIONS.md instead")
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    shown = {k: (v if k not in ("decisions_text", "code_files", "files", "dev_precompute") else f"<{len(v)} entries>")
             for k, v in rec.items()}
    print(json.dumps(shown, indent=2))
    print(f"record sha256: {sha_file(path)}  (keep this digest outside the repository)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["section2", "protocol", "model"])
    ap.add_argument("--checkpoint")
    ap.add_argument("--critic-signoff", required=True)
    a = ap.parse_args()
    stamp = {"frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
             "critic_signoff": a.critic_signoff}
    if a.stage == "section2":
        _write(FROZEN_S2, {**stamp, **s2_digests()})
    elif a.stage == "protocol":
        s2_changed = verify_s2(allow_documented=True)
        with tarfile.open(CODE_ARCHIVE, "w:gz") as tar:
            for path in code_files():
                tar.add(os.path.join(ROOT, path), arcname=path)
        _write(FROZEN_PROTOCOL, {**stamp, "s2_files_changed_since_s2": s2_changed,
                                 "protocol": sha_file(os.path.join(EXP, "PROTOCOL.md")),
                                 "code_files": code_files(), "code_archive_sha256": sha_file(CODE_ARCHIVE),
                                 "decisions_text": open(os.path.join(EXP, "DECISIONS.md"), encoding="utf-8").read()})
    else:
        _, changed = verify_protocol()
        if not a.checkpoint:
            raise SystemExit("--checkpoint is required for the model freeze")
        rec = {**stamp, **digests(a.checkpoint), "code_changed_since_protocol": changed}
        if changed:
            rec["code_diff_sha256"] = sha_file(write_code_diff(changed))
        _write(FROZEN, rec)


if __name__ == "__main__":
    main()
