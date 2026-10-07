"""Trainable-parameter selection for UMA.

Parameter names below were read from the UMA-S architecture instantiated with
fairchem-core 2.23.0 (146.6 M parameters, 149 tensors; ~145 M of them are
MoLE expert weights of shape (32, out, in) inside the SO(2) convolutions).
Re-run `inventory()` on the actual checkpoint before relying on a preset.

Presets (all change U_w itself, so all satisfy the "update UMA's weights"
requirement; the additive correction in correction.py does not):

  readout        energy MLP of the output head (~33 k params). In UMA-S the
                 head is shared by all tasks (DatasetSpecificSingleHeadWrapper),
                 so this also moves other tasks' surfaces.
  readout_norm   + final equivariant RMS norm
  last_block     + the last message-passing block (incl. its experts)
  routing        MoLE routing MLP + composition embedding: changes how the
                 32 experts are mixed per (composition, charge, spin, task);
                 every mixture is still an exact UMA-architecture model
  experts        all MoLE expert weights (largest, needs strongest replay)
  radial         edge radial MLPs (distance dependence of messages)
  full           everything that is trainable in the released architecture
                 (dataset embeddings are frozen by construction: requires_grad=False)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from torch import nn

READOUT = [r"^output_heads\..*\.energy_block\."]
FINAL_NORM = [r"^backbone\.norm\."]
ROUTING = [r"^backbone\.routing_mlp\.", r"^backbone\.composition_embedding\."]
EXPERTS = [r"\.fc_m0\.weights$", r"\.so2_m_conv\.\d+\.fc\.weights$"]
RADIAL = [r"\.rad_func\."]
FROZEN_BY_DESIGN = [r"^backbone\.dataset_embedding\."]


def _last_block_patterns(model: nn.Module) -> list[str]:
    n = len(model.backbone.blocks)
    return [rf"^backbone\.blocks\.{n - 1}\."]


PRESETS = {
    "readout": lambda m: READOUT,
    "readout_norm": lambda m: READOUT + FINAL_NORM,
    "last_block": lambda m: READOUT + FINAL_NORM + _last_block_patterns(m),
    "routing": lambda m: ROUTING,
    "experts": lambda m: EXPERTS,
    "radial": lambda m: RADIAL,
    "full": lambda m: [r".*"],
}


@dataclass
class SelectionReport:
    preset: str | None
    n_tensors: int
    n_params: int
    n_total_params: int
    names: list[str]

    def summary(self) -> str:
        frac = self.n_params / max(self.n_total_params, 1)
        return (f"{self.preset or 'custom'}: {self.n_tensors} tensors, "
                f"{self.n_params:,} / {self.n_total_params:,} params ({100 * frac:.3f}%)")


def select_parameters(
    model: nn.Module,
    preset: str | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> SelectionReport:
    """Set requires_grad on the selected UMA parameters and freeze the rest."""
    patterns = list(include or [])
    if preset is not None:
        patterns += PRESETS[preset](model)
    if not patterns:
        raise ValueError("no parameters selected")
    inc = [re.compile(p) for p in patterns]
    exc = [re.compile(p) for p in (list(exclude or []) + FROZEN_BY_DESIGN)]
    names, n_params, total = [], 0, 0
    for name, p in model.named_parameters():
        total += p.numel()
        on = any(r.search(name) for r in inc) and not any(r.search(name) for r in exc)
        p.requires_grad_(on)
        if on:
            names.append(name)
            n_params += p.numel()
    if not names:
        raise ValueError(f"selection {preset}/{include} matched no parameters")
    return SelectionReport(preset, len(names), n_params, total, names)


def inventory(model: nn.Module) -> list[dict]:
    """Parameter table grouped by name pattern (block index collapsed)."""
    groups: dict[str, dict] = {}
    for name, p in model.named_parameters():
        key = re.sub(r"blocks\.\d+", "blocks.*", name)
        g = groups.setdefault(key, {"pattern": key, "count": 0, "numel": 0,
                                    "shape": tuple(p.shape), "requires_grad": p.requires_grad})
        g["count"] += 1
        g["numel"] += p.numel()
    return list(groups.values())
