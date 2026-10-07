"""Shared machinery for ensemble-averaged observables.

    y_hat = T(<A>_w)

T is a smooth transform of reweighted frame averages (e.g. rho = M/<V>,
kappa_T = (<V^2> - <V>^2)/(kT <V>)). Works in both reweighting modes:
"graph" keeps the full graph; "surrogate" returns a detached value plus a
`backward_fn` that applies the chain rule dL/d<A> = J_T^T dL/dy and hands it
to the memory-bounded surrogate backward pass.

Sampling uncertainty: frame covariance / n_eff (n_eff combines weight ESS and
time correlation), propagated through T by the delta method.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from ..potential.base import Potential
from ..reweighting.difftre import ReweightedEnsemble
from .base import Prediction


def ensemble_prediction(
    ens: ReweightedEnsemble,
    potential: Potential,
    frame_fn,
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    mode: str = "graph",
    meta: dict | None = None,
    full_covariance_max: int = 400,
) -> Prediction:
    est = ens.estimate(potential, frame_fn, mode)
    T = transform or (lambda x: x)
    backward_fn = None
    if mode == "graph":
        value = T(est.value)
    else:
        leaf = est.value.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            value = T(leaf)

        def backward_fn(g, _est=est):
            # Rebuild T on a fresh leaf: the caller's loss.backward() may already
            # have freed the graph between `leaf` and `value`.
            x = _est.value.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                y = T(x)
                (gx,) = torch.autograd.grad(y, x, grad_outputs=g.to(y.dtype).reshape(y.shape))
            ens.surrogate_backward(potential, frame_fn, _est, gx)

    # delta-method sampling variance
    x0 = est.value.detach()
    a = est.log_weights.exp()
    A = est.frame_values
    n_eff = est.diagnostics["n_eff_combined"]
    J = torch.autograd.functional.jacobian(lambda x: T(x).reshape(-1), x0).reshape(-1, x0.numel())
    if x0.numel() <= full_covariance_max:
        dA = A - x0
        cov = (a[:, None, None] * dA[:, :, None] * dA[:, None, :]).sum(0) / n_eff
        var = torch.einsum("ik,kl,il->i", J, cov, J)
    else:
        var = (J**2) @ (est.sem**2)
    return Prediction(
        value=value,
        variance=var.reshape(value.shape).detach(),
        valid=True,
        diagnostics={
            "ess": est.ess,
            "ess_fraction": est.ess_fraction,
            "statistical_inefficiency": est.g,
            "n_eff": n_eff,
            "max_weight": est.max_weight,
            "beta_du_std": est.diagnostics.get("beta_du_std"),
            "n_frames": est.n_frames,
        },
        meta=dict(meta or {}, estimate=est),
        backward_fn=backward_fn,
    )
