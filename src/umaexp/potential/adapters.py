"""Low-rank delta reparametrization of existing UMA weight tensors.

This is *not* generic transformer LoRA bolted onto a new layer. The delta is
added to an existing weight tensor (W = W0 + A B, B initialised to zero) and
flows through UMA's own operator, so whatever symmetry that operator
guarantees for arbitrary weight values is inherited:

  * MoLE expert weights (E, out, in) inside eSCN SO(2) convolutions: the SO(2)
    equivariance of those convolutions comes from how the weights act on the
    m-components, not from constraints on the weight values, so W0 + dW is as
    equivariant as W0. Verified numerically by tests/test_uma_potential.py
    (rotation/translation/permutation checks after randomizing adapters).
  * Plain nn.Linear weights acting on scalar (L=0) channels: invariant inputs
    stay invariant.

Do NOT attach deltas to tensors whose values encode symmetry structure
(e.g. Wigner/CG buffers); only nn.Parameters listed by the selectors below.
"""

from __future__ import annotations

import re

import torch
from torch import nn
from torch.nn.utils import parametrize


class LowRankDelta(nn.Module):
    def __init__(self, weight_shape: tuple[int, ...], rank: int, per_expert: bool,
                 dtype, device, init_scale: float = 1e-3):
        super().__init__()
        self.per_expert = per_expert and len(weight_shape) == 3
        out_f, in_f = weight_shape[-2], weight_shape[-1]
        lead = (weight_shape[0],) if self.per_expert else ()
        g = torch.Generator(device="cpu").manual_seed(1234)
        a = torch.randn(*lead, out_f, rank, generator=g, dtype=torch.float64) * init_scale
        self.A = nn.Parameter(a.to(device=device, dtype=dtype))
        self.B = nn.Parameter(torch.zeros(*lead, rank, in_f, dtype=dtype, device=device))

    def forward(self, w0: torch.Tensor) -> torch.Tensor:
        delta = self.A @ self.B
        return w0 + delta  # broadcasts over the expert axis when shared


def add_lowrank_adapters(
    model: nn.Module,
    patterns: list[str],
    rank: int = 4,
    per_expert: bool = False,
) -> list[str]:
    """Attach deltas to parameters whose full name matches any pattern.

    Freezes everything else. Returns the list of adapted parameter names.
    """
    regs = [re.compile(p) for p in patterns]
    targets = []
    for mod_name, module in model.named_modules():
        for p_name, p in list(module.named_parameters(recurse=False)):
            full = f"{mod_name}.{p_name}" if mod_name else p_name
            if p.dim() >= 2 and any(r.search(full) for r in regs):
                targets.append((module, p_name, full, p))
    if not targets:
        raise ValueError(f"no parameters matched {patterns}")
    for p in model.parameters():
        p.requires_grad_(False)
    names = []
    for module, p_name, full, p in targets:
        delta = LowRankDelta(tuple(p.shape), rank, per_expert, p.dtype, p.device)
        parametrize.register_parametrization(module, p_name, delta)
        getattr(module.parametrizations, p_name).original.requires_grad_(False)
        names.append(full)
    return names


def has_adapters(model: nn.Module) -> bool:
    return any(parametrize.is_parametrized(m) for m in model.modules())


def merge_adapters(model: nn.Module) -> int:
    """Bake W0 + A B into plain parameters (for export / fast inference)."""
    n = 0
    for module in list(model.modules()):
        if parametrize.is_parametrized(module):
            for name in list(module.parametrizations.keys()):
                parametrize.remove_parametrizations(module, name, leave_parametrized=True)
                n += 1
    return n
