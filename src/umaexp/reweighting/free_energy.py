r"""Differentiable free-energy estimators: EXP (Zwanzig), BAR, MBAR, TI.

All reduced potentials u = beta U (dimensionless). Solvers run detached (they
are iterative) and gradients are attached by implicit differentiation with a
detached Newton step on the estimator's defining equation R(f, u) = 0:

    f(w) = f* - J^{-1} R(f*, u(w)),      J = dR/df |_(f*, u*)   (detached)

whose value is f* and whose derivative is df/du = -J^{-1} dR/du, i.e. exactly
the implicit-function-theorem gradient. Using an MBAR *library* for the solve
does not connect it to autograd; this module does that step explicitly, and
tests compare values with pymbar and gradients with finite differences.

Which gradient is a valid *training* gradient
---------------------------------------------
Differentiating BAR/MBAR while holding the samples fixed gives the
sensitivity of the estimator, not d(Delta F)/dw: once w changes, samples
drawn from the old state are no longer samples of the new one. The valid
route (used by the training code) treats the updated potential as an
*unsampled* MBAR target state:

    f_t(w) = -ln sum_n exp(-u_t(x_n; w)) / sum_k N_k exp(f_k - u_k(x_n))

with the sampled states' u_k and f_k fixed. Its gradient is exactly the
reweighted canonical average <du_t/dw>_t (`target_state_free_energy`).
EXP with samples from a fixed reference state is the one-state special case.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.optimize import brentq


def exp_estimator(du_forward: torch.Tensor) -> torch.Tensor:
    """Delta f = f_1 - f_0 = -ln < exp(-du) >_0, du = u_1 - u_0 on state-0 samples."""
    n = du_forward.numel()
    return -(torch.logsumexp(-du_forward, dim=0) - math.log(n))


def _bar_residual(df, w_f, w_r):
    """Bennett residual; w_f = u1-u0 on samples from 0, w_r = u0-u1 on samples from 1."""
    n0, n1 = w_f.numel(), w_r.numel()
    M = math.log(n0 / n1)  # ln(T_F / T_R), Shirts et al. PRL 91, 140601 (2003)
    lhs = torch.sigmoid(-(M + w_f - df)).sum()
    rhs = torch.sigmoid(-(-M + w_r + df)).sum()
    return torch.log(lhs) - torch.log(rhs)


def bar_estimator(w_f: torch.Tensor, w_r: torch.Tensor) -> torch.Tensor:
    """Delta f by Bennett's acceptance ratio.

    The attached gradient is the estimator's sensitivity at FIXED samples
    (implicit differentiation of the BAR equation). It is not a valid
    training gradient when the differentiated potential generated the
    state-1 samples; use `target_state_free_energy` for training.
    """
    wf, wr = w_f.detach(), w_r.detach()

    def r(x):
        return float(_bar_residual(torch.tensor(x, dtype=torch.float64), wf, wr))

    lo = float(torch.min(torch.cat([wf, -wr]))) - 50.0
    hi = float(torch.max(torch.cat([wf, -wr]))) + 50.0
    df_star = brentq(r, lo, hi, xtol=1e-14, rtol=1e-14, maxiter=500)
    df0 = torch.tensor(df_star, dtype=torch.float64, requires_grad=True)
    with torch.enable_grad():
        (J,) = torch.autograd.grad(_bar_residual(df0, wf, wr), df0)
    R = _bar_residual(torch.tensor(df_star, dtype=torch.float64), w_f, w_r)
    return df_star - R / J.detach()


def _mbar_residual(f_sub, u_kn, log_N):
    f = torch.cat([torch.zeros(1, dtype=u_kn.dtype), f_sub])
    log_D = torch.logsumexp(log_N[:, None] + f[:, None] - u_kn, dim=0)  # (N,)
    R = f + torch.logsumexp(-u_kn - log_D[None, :], dim=1)
    return R[1:]


def mbar_solve(u_kn: np.ndarray, N_k: np.ndarray, tol: float = 1e-12, max_iter: int = 10000) -> np.ndarray:
    """Self-consistent iteration + Newton polishing; returns f with f_0 = 0."""
    u = torch.as_tensor(u_kn, dtype=torch.float64)
    log_N = torch.log(torch.as_tensor(N_k, dtype=torch.float64))
    K = u.shape[0]
    if K == 1:  # a single sampled state: nothing to solve (gauge f_0 = 0)
        return np.zeros(1)
    f = torch.zeros(K, dtype=torch.float64)
    for it in range(max_iter):
        log_D = torch.logsumexp(log_N[:, None] + f[:, None] - u, dim=0)
        f_new = -torch.logsumexp(-u - log_D[None, :], dim=1)
        f_new = f_new - f_new[0]
        if float((f_new - f).abs().max()) < 1e-8 or it == max_iter - 1:
            f = f_new
            break
        f = f_new
    f_sub = f[1:].clone()
    for _ in range(20):
        x = f_sub.clone().requires_grad_(True)
        with torch.enable_grad():
            R = _mbar_residual(x, u, log_N)
            J = torch.autograd.functional.jacobian(lambda y: _mbar_residual(y, u, log_N), x)
        if float(R.abs().max()) < tol:
            break
        f_sub = f_sub - torch.linalg.solve(J, R.detach())
    return torch.cat([torch.zeros(1, dtype=torch.float64), f_sub.detach()]).numpy()


def mbar_free_energies(u_kn: torch.Tensor, N_k) -> torch.Tensor:
    """Dimensionless free energies f_k (f_0 = 0) with implicit gradients w.r.t. u_kn.

    u_kn: (K, N_total) reduced potential of every pooled sample in every state.
    Gradients are estimator sensitivities at fixed samples (see module doc).
    """
    N_k = np.asarray(N_k, dtype=float)
    if u_kn.shape[0] == 1:
        return torch.zeros(1, dtype=torch.float64)
    f_star = torch.as_tensor(mbar_solve(u_kn.detach().numpy(), N_k), dtype=torch.float64)
    log_N = torch.log(torch.as_tensor(N_k, dtype=torch.float64))
    x = f_star[1:].clone()
    J = torch.autograd.functional.jacobian(lambda y: _mbar_residual(y, u_kn.detach(), log_N), x)
    R = _mbar_residual(f_star[1:], u_kn, log_N)
    f_sub = f_star[1:] - torch.linalg.solve(J.detach(), R)
    return torch.cat([torch.zeros(1, dtype=torch.float64), f_sub])


def target_state_free_energy(u_kn_sampled, N_k, u_target: torch.Tensor,
                             f_sampled: np.ndarray | None = None) -> torch.Tensor:
    """Free energy of an unsampled target state relative to sampled state 0.

    u_kn_sampled: (K, N) reduced potentials of the pooled samples in the K
      *sampled* states (fixed; detached). u_target: (N,) reduced potential of
      the same samples under the target (carries the graph to w).
    Valid training gradient: d f_t / dw = <d u_t / dw>_t.
    """
    u_s = torch.as_tensor(u_kn_sampled, dtype=torch.float64).detach()
    N_k = np.asarray(N_k, dtype=float)
    if f_sampled is None:
        f_sampled = mbar_solve(u_s.numpy(), N_k)
    f_s = torch.as_tensor(f_sampled, dtype=torch.float64)
    log_N = torch.log(torch.as_tensor(N_k, dtype=torch.float64))
    log_D = torch.logsumexp(log_N[:, None] + f_s[:, None] - u_s, dim=0)
    return -torch.logsumexp(-u_target - log_D, dim=0)


def mbar_weights(u_kn: torch.Tensor, N_k, f: torch.Tensor, state: int) -> torch.Tensor:
    """Normalized MBAR weights of all pooled samples for `state` (differentiable)."""
    log_N = torch.log(torch.as_tensor(np.asarray(N_k, float), dtype=torch.float64))
    log_D = torch.logsumexp(log_N[:, None] + f[:, None] - u_kn, dim=0)
    log_w = f[state] - u_kn[state] - log_D
    return torch.exp(log_w - torch.logsumexp(log_w, 0))


def thermodynamic_integration(lambdas, dudl_means: torch.Tensor, rule: str = "trapezoid") -> torch.Tensor:
    """Integrate <dU/dlambda> over lambda (eV); dudl_means may carry graphs."""
    lam = torch.as_tensor(lambdas, dtype=torch.float64)
    if rule == "trapezoid":
        return torch.trapezoid(dudl_means, lam)
    raise ValueError(rule)
