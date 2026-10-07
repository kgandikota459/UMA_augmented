"""Reserved-set blocklist for the Vib-v1 held-out experiment (critic reports #2 B6, #3 S2-9).

Every G2 candidate outside the pilot list is reserved: no data family may use it for training or
development until Vib-v1 is unblinded. The list is public (it follows from public, label-free
rules); which candidates are in the sealed test set is not. Loaders call ``assert_not_reserved`` on
the molecules they are about to use. There is no bypass: the unblinding script builds held-out
molecules directly and never goes through these loaders.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESERVED_FILE = os.path.join(ROOT, "experiments", "vib_v1", "reserved_blocks.json")


def reserved_blocks() -> set[str]:
    if not os.path.exists(RESERVED_FILE):
        raise FileNotFoundError(f"{RESERVED_FILE} is missing: refusing to load data without the reserved set")
    return set(json.load(open(RESERVED_FILE))["inchikey_blocks"])


def assert_not_reserved(blocks: Iterable[str | None], context: str) -> None:
    hit = {b for b in blocks if b} & reserved_blocks()
    if hit:
        raise PermissionError(f"{context}: {len(hit)} molecule(s) are reserved for the Vib-v1 held-out experiment "
                              f"(experiments/vib_v1/reserved_blocks.json) and may not be used before unblinding")
