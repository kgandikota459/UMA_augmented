r"""Differentiable trajectory reweighting (DiffTRe-style) in PyTorch.

Samples x_i are drawn (by any sampler) from the reference density
p_s(x) ~ exp(-u_s(x)) with reduced potential

    u(x) = beta [U(R) + bias(R)] + beta p V            (NPT; drop pV for NVT)

and reweighted to the target U_w at target state (beta_t, p_t):

    log a_i(w) = -[u_t(x_i; w) - u_s(x_i)] - logsumexp_j(...)
    <A>_w      = sum_i a_i(w) A_w(x_i)

Measure terms (V^N from scaled coordinates, Jacobians of the barostat
proposal) are identical in numerator and denominator when sampling and
target ensembles share the same coordinates and N, so they cancel; changing
N or the ensemble type is not supported by this estimator. Coordinates are
detached; U_w(x_i) and A_w(x_i) keep their graph to w, so

    d<A>/dw = <dA/dw> - beta_t Cov(A, dU_w/dw)          (canonical, fixed beta)

is reproduced by autograd (checked in tests against the explicit covariance).

Memory: `mode="graph"` keeps every frame's graph (small problems, testing).
`mode="surrogate"` never holds more than one chunk: pass 1 computes values
without a graph; given dL/d<A>, pass 2 back-propagates the exact first-order
gradient through the detached-coefficient surrogate
    S(w) = sum_i c_i U_w(x_i) + sum_i sum_k d_ik A_wk(x_i),
    c_i = -beta_t a_i sum_k g_k (A_ik - <A>_k),  d_ik = a_i g_k,
chunk by chunk (gradient accumulation).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from .. import units
from ..potential.base import Potential
from ..sampling.cache import SampleCache, ThermoState
from ..sampling.diagnostics import statistical_inefficiency
from ..systems.system import System

# Observable on frames: (positions (b,N,3), cells (b,3,3)|None) -> (b, K) tensor.
# Functions with attribute `uses_energy = True` also receive U_w(frames) (b,)
# and are treated as explicitly w-dependent (e.g. enthalpy fluctuations);
# functions with `depends_on_w = True` close over the potential themselves.
FrameObservable = Callable[..., torch.Tensor]


def reduced_potential(U, beta, volume=None, pressure=None, bias=None):
    u = beta * U
    if bias is not None:
        u = u + beta * bias
    if pressure is not None:
        if volume is None:
            raise ValueError("pressure term needs volumes")
        u = u + beta * pressure * volume
    return u


def normalized_log_weights(u_target: torch.Tensor, u_sampled: torch.Tensor) -> torch.Tensor:
    x = -(u_target - u_sampled)
    return x - torch.logsumexp(x, dim=0)


def kish_ess(log_a: torch.Tensor) -> float:
    a = log_a.detach().exp()
    return float(1.0 / (a * a).sum())


@dataclass
class EnsembleEstimate:
    value: torch.Tensor  # (K,) with graph (mode="graph") or detached (surrogate pass 1)
    sem: torch.Tensor  # (K,) correlation-aware standard error estimate (detached)
    ess: float  # Kish ESS of the weights
    ess_fraction: float
    g: float  # statistical inefficiency (max over observables, unweighted series)
    n_frames: int
    max_weight: float
    log_weights: torch.Tensor  # detached
    frame_values: torch.Tensor  # (F, K) detached
    diagnostics: dict = field(default_factory=dict)


class ReweightedEnsemble:
    """Reweighting of one sample cache to the current potential."""

    def __init__(self, cache: SampleCache, system: System, target: ThermoState | None = None,
                 chunk_size: int = 32):
        cache.check_compatible(system.atom_order_hash(), cache.state)
        self.cache = cache
        self.system = system
        self.target = target or cache.state
        if self.target.ensemble != cache.state.ensemble:
            raise ValueError("sampling and target ensembles must match for this estimator")
        self.chunk = chunk_size
        self.pos = cache.positions_tensor()
        self.cells = cache.cells_tensor()
        vols = cache.volumes
        self.vol = None if vols is None else torch.as_tensor(vols, dtype=torch.float64)
        s = cache.state
        self.beta_s = units.beta(s.temperature_k)
        self.beta_t = units.beta(self.target.temperature_k)
        self.p_s = None if s.pressure_bar is None else s.pressure_bar * units.BAR_TO_EV_A3
        self.p_t = None if self.target.pressure_bar is None else self.target.pressure_bar * units.BAR_TO_EV_A3
        bias = None if cache.bias is None else torch.as_tensor(cache.bias, dtype=torch.float64)
        self.u_s = reduced_potential(torch.as_tensor(cache.u_sampling, dtype=torch.float64), self.beta_s,
                                     self.vol if self.p_s is not None else None, self.p_s, bias)

    # ------------------------------------------------------------------
    def _chunks(self):
        F = self.pos.shape[0]
        for i in range(0, F, self.chunk):
            j = min(i + self.chunk, F)
            yield i, j, self.pos[i:j], None if self.cells is None else self.cells[i:j]

    def energies(self, potential: Potential, with_graph: bool) -> torch.Tensor:
        out = []
        ctx = torch.enable_grad() if with_graph else torch.no_grad()
        with ctx:
            for _, _, p, c in self._chunks():
                e = potential.energy(self.system, p, c)
                out.append(e if with_graph else e.detach())
        return torch.cat(out)

    def _u_target(self, U):
        return reduced_potential(U, self.beta_t, self.vol if self.p_t is not None else None, self.p_t)

    def _evaluate(self, potential: Potential, fn: FrameObservable, with_graph: bool):
        """Energies and frame observables in one pass over chunks."""
        Us, As = [], []
        ctx = torch.enable_grad() if with_graph else torch.no_grad()
        with ctx:
            for _, _, p, c in self._chunks():
                U = potential.energy(self.system, p, c)
                v = call_frame_fn(fn, p, c, U)
                v = v.reshape(v.shape[0], -1).to(torch.float64)
                Us.append(U if with_graph else U.detach())
                As.append(v if with_graph else v.detach())
        return torch.cat(Us), torch.cat(As)

    def estimate(self, potential: Potential, fn: FrameObservable, mode: str = "graph") -> EnsembleEstimate:
        """Reweighted <A>; "graph" keeps the full graph, "surrogate" returns detached values."""
        if mode not in ("graph", "surrogate"):
            raise ValueError(mode)
        with_graph = mode == "graph"
        U, A = self._evaluate(potential, fn, with_graph)
        log_a = normalized_log_weights(self._u_target(U), self.u_s)
        a = log_a.exp()
        value = (a[:, None] * A).sum(0)
        est = self._package(value, log_a.detach(), A.detach())
        est.diagnostics["beta_du_std"] = float((self._u_target(U.detach()) - self.u_s).std())
        return est

    def _package(self, value, log_a, A) -> EnsembleEstimate:
        a = log_a.exp()
        F = A.shape[0]
        ess = kish_ess(log_a)
        g = max(statistical_inefficiency(A[:, k].numpy()) for k in range(A.shape[1])) if F > 3 else 1.0
        mean = (a[:, None] * A).sum(0)
        var = (a[:, None] * (A - mean) ** 2).sum(0)
        n_eff = max(min(ess, F / g), 1.0)
        sem = torch.sqrt(var / n_eff)
        return EnsembleEstimate(value, sem, ess, ess / F, g, F, float(a.max()), log_a, A,
                                {"n_eff_combined": n_eff})

    # ------------------------------------------------------------------
    def surrogate_backward(self, potential: Potential, fn: FrameObservable, est: EnsembleEstimate,
                           grad_value: torch.Tensor, fn_depends_on_w: bool = False) -> None:
        """Accumulate dL/dw into .grad given g = dL/d<A> (exact first order).

        Must be called with the potential in the same state used for `est`.
        """
        a = est.log_weights.exp()
        A = est.frame_values
        g = grad_value.detach().to(torch.float64).reshape(-1)
        c = -self.beta_t * a * ((A - est.value.detach()) * g).sum(-1)  # (F,)
        d = a[:, None] * g[None, :]  # (F, K)
        explicit = fn_depends_on_w or getattr(fn, "uses_energy", False) or getattr(fn, "depends_on_w", False)
        for i, j, p, cc in self._chunks():
            with torch.enable_grad():
                U = potential.energy(self.system, p, cc)
                s = (c[i:j].to(U.dtype) * U).sum()
                if explicit:
                    v = call_frame_fn(fn, p, cc, U).reshape(j - i, -1).to(torch.float64)
                    s = s + (d[i:j] * v).sum()
            if s.requires_grad:
                s.backward()


def call_frame_fn(fn, positions, cells, energies):
    """Frame observables may declare `uses_energy = True` to receive U_w per frame."""
    if getattr(fn, "uses_energy", False):
        return fn(positions, cells, energies)
    return fn(positions, cells)


def covariance_gradient(potential: Potential, ens: ReweightedEnsemble, A_frames: torch.Tensor,
                        params: list[torch.Tensor]) -> list[torch.Tensor]:
    """Explicit -beta Cov(A, dU/dw) at the sampling weights (validation helper).

    For observables without explicit w-dependence, evaluated at the
    reference weights (a_i uniform when u_target == u_sampled).
    """
    with torch.no_grad():
        U = ens.energies(potential, with_graph=False)
        log_a = normalized_log_weights(ens._u_target(U), ens.u_s)
        a = log_a.exp()
        mean = (a * A_frames).sum()
    grads = [torch.zeros_like(p) for p in params]
    for i, j, p, c in ens._chunks():
        with torch.enable_grad():
            e = potential.energy(ens.system, p, c)
            coeff = (a[i:j] * (A_frames[i:j] - mean)).to(e.dtype)
            gs = torch.autograd.grad((coeff * e).sum(), params, allow_unused=True)
        for k, gk in enumerate(gs):
            if gk is not None:
                grads[k] += gk
    return [-ens.beta_t * g for g in grads]


def ideal_weight_ess_bound(delta_u_std: float) -> float:
    """Expected ESS fraction for Gaussian-distributed beta*dU with SD s: exp(-s^2)."""
    return math.exp(-(delta_u_std**2))


def summarize_overlap(log_a: torch.Tensor, u_target: torch.Tensor, u_sampled: torch.Tensor) -> dict:
    du = (u_target - u_sampled).detach().numpy()
    return {
        "ess_fraction": kish_ess(log_a) / log_a.numel(),
        "beta_du_std": float(np.std(du)),
        "max_weight": float(log_a.exp().max()),
    }
