"""Unseal Vib-v1's development extension D_ext after the whole protocol is frozen (PROTOCOL.md 2.4).

Verifies the salted commitments published at build time, copies D_ext into
data/processed/, lists its molecules in data/pilot/vibrational_ext.yaml and
releases its InChIKey blocks from the reserved set. T stays sealed.

  python scripts/unseal_dev_ext.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from freeze_vib_v1 import EXP, ROOT, verify_protocol  # noqa: E402

SEALED = os.path.join(os.environ.get("UMAEXP_SEALED_DIR", os.path.join(os.path.expanduser("~"), ".umaexp_sealed",
                                                                         "vib_v1")), "build")


def main():
    verify_protocol()  # also requires every code change since the protocol freeze to be in DEVIATIONS.md
    public = json.load(open(os.path.join(EXP, "heldout_public.json")))
    salt = open(os.path.join(SEALED, "SALT")).read()
    if hashlib.sha256(salt.encode()).hexdigest() != public["salt_sha256"]:
        raise SystemExit("salt does not match the published digest")
    for name, key in (("dev_ext.parquet", "commit_parquet"), ("dev_ext_molecules.json", "commit_molecules")):
        got = hashlib.sha256((salt + open(os.path.join(SEALED, name), "rb").read().hex()).encode()).hexdigest()
        if got != public["dev_ext"][key]:
            raise SystemExit(f"{name} does not match its commitment")
    dst = os.path.join(ROOT, "data", "processed", "vib_v1_dev_ext.parquet")
    if os.path.exists(dst):
        raise SystemExit("D_ext already unsealed")
    shutil.copy(os.path.join(SEALED, "dev_ext.parquet"), dst)
    mols = json.load(open(os.path.join(SEALED, "dev_ext_molecules.json")))
    with open(os.path.join(ROOT, "data", "pilot", "vibrational_ext.yaml"), "w") as f:
        yaml.safe_dump({"source": "ase.collections.g2 via experiments/vib_v1/PROTOCOL.md 2.4 (unsealed after the "
                                  "protocol freeze)", "molecules": mols}, f, sort_keys=False)
    rpath = os.path.join(EXP, "reserved_blocks.json")
    reserved = json.load(open(rpath))
    released = {m["inchikey"][:14] for m in mols}
    reserved["inchikey_blocks"] = sorted(set(reserved["inchikey_blocks"]) - released)
    reserved["n"] = len(reserved["inchikey_blocks"])
    reserved["released_dev_ext"] = len(released)
    with open(rpath, "w") as f:
        json.dump(reserved, f, indent=1)
    print(f"D_ext unsealed: {len(mols)} molecules; reserved set now {reserved['n']} blocks")


if __name__ == "__main__":
    main()
