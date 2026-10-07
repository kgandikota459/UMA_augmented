"""Linear combinations of predictions that may use different gradient modes."""

from __future__ import annotations

import torch

from .base import Prediction


def combine(preds: list[Prediction], coeffs: list[float], const: float = 0.0, meta: dict | None = None) -> Prediction:
    value = sum(c * p.value for c, p in zip(coeffs, preds)) + const
    var = sum((c**2) * (p.variance if p.variance is not None else 0.0) for c, p in zip(coeffs, preds))
    needs_fn = any(p.backward_fn is not None for p in preds)
    backward_fn = None
    if needs_fn:
        leaf_value = value.detach()

        def backward_fn(g):
            for c, p in zip(coeffs, preds):
                p.backward(c * g)

        value = leaf_value
    return Prediction(
        value=value,
        variance=torch.as_tensor(var, dtype=torch.float64),
        valid=all(p.valid for p in preds),
        reason="; ".join(p.reason for p in preds if p.reason),
        diagnostics={f"part{i}": p.diagnostics for i, p in enumerate(preds)},
        meta=dict(meta or {}),
        backward_fn=backward_fn,
    )
