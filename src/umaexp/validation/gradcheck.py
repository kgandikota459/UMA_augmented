"""Finite-difference checks of every derivative order the objectives use.

These guard against the silent failure modes the design warns about: a
calculator wrapper returning numpy, a detached tensor, a no_grad block, or an
optimized kernel without double-backward support would all make autograd
disagree with finite differences (or return zeros/None).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from ..potential.base import Potential
from ..potential.derivatives import hessian
from ..systems.system import System


@dataclass
class CheckResult:
    name: str
    max_abs_err: float
    max_rel_err: float
    scale: float
    passed: bool

    def __str__(self):
        flag = "PASS" if self.passed else "FAIL"
        return (f"[{flag}] {self.name}: max|err|={self.max_abs_err:.3e} "
                f"(rel {self.max_rel_err:.3e}, scale {self.scale:.3e})")


def _result(name, auto, fd, rtol, atol):
    err = (auto - fd).abs()
    scale = float(fd.abs().max()) if fd.numel() else 0.0
    max_abs = float(err.max()) if err.numel() else 0.0
    rel = max_abs / max(scale, 1e-300)
    return CheckResult(name, max_abs, rel, scale, max_abs <= atol + rtol * scale)


def check_forces(pot: Potential, system: System, pos: torch.Tensor, cell=None, h: float = 1e-4,
                 rtol=1e-5, atol=1e-7, n_coords: int | None = None) -> CheckResult:
    """F = -dU/dR versus central differences of U."""
    pos = pos.detach().to(torch.float64)
    _, f = pot.energy_and_forces(system, pos, cell)
    flat = pos.reshape(-1)
    idx = range(flat.numel()) if n_coords is None else range(min(n_coords, flat.numel()))
    fd, au = [], []
    with torch.no_grad():
        for i in idx:
            xp, xm = flat.clone(), flat.clone()
            xp[i] += h
            xm[i] -= h
            ep = pot.energy(system, xp.view_as(pos), cell)
            em = pot.energy(system, xm.view_as(pos), cell)
            fd.append(-(ep - em) / (2 * h))
            au.append(f.reshape(-1)[i])
    return _result("forces", torch.stack(au), torch.stack(fd), rtol, atol)


def check_hessian(pot: Potential, system: System, pos: torch.Tensor, cell=None, h: float = 1e-4,
                  rtol=1e-5, atol=1e-6) -> CheckResult:
    """Autograd Hessian versus central differences of autograd forces."""
    pos = pos.detach().to(torch.float64)
    H = hessian(pot, system, pos, cell, create_graph=False)
    flat = pos.reshape(-1)
    n = flat.numel()
    Hfd = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        xp, xm = flat.clone(), flat.clone()
        xp[i] += h
        xm[i] -= h
        _, fp = pot.energy_and_forces(system, xp.view_as(pos), cell)
        _, fm = pot.energy_and_forces(system, xm.view_as(pos), cell)
        Hfd[:, i] = -(fp - fm).reshape(-1) / (2 * h)
    return _result("hessian", H, 0.5 * (Hfd + Hfd.T), rtol, atol)


def check_cell_derivative(pot: Potential, system: System, pos, cell, h: float = 1e-5,
                          rtol=1e-4, atol=1e-6) -> CheckResult:
    """dU/deps (symmetric strain) versus finite strains."""
    from ..potential.derivatives import strain_derivative

    pos = pos.detach().to(torch.float64)
    cell = cell.detach().to(torch.float64)
    _, de = strain_derivative(pot, system, pos, cell)
    fd = torch.zeros(3, 3, dtype=torch.float64)
    with torch.no_grad():
        for a in range(3):
            for b in range(3):
                eps = torch.zeros(3, 3, dtype=torch.float64)
                eps[a, b] += 0.5 * h
                eps[b, a] += 0.5 * h
                d = torch.eye(3, dtype=torch.float64)
                ep = pot.energy(system, pos @ (d + eps), cell @ (d + eps))
                em = pot.energy(system, pos @ (d - eps), cell @ (d - eps))
                fd[a, b] = (ep - em) / (2 * h)
    return _result("strain_derivative", 0.5 * (de + de.T), fd, rtol, atol)


def check_parameter_gradient(
    loss_fn: Callable[[], torch.Tensor],
    params: list[torch.nn.Parameter],
    n_probe: int = 5,
    h: float = 1e-5,
    rtol: float = 1e-4,
    atol: float = 1e-9,
    seed: int = 0,
    name: str = "parameter_gradient",
) -> CheckResult:
    """Directional derivative check: autograd g.v versus [L(w+hv) - L(w-hv)]/2h.

    Random directions over all given parameters test the full gradient
    without one finite difference per scalar weight (UMA has ~1.5e8).
    Also fails when every autograd gradient is None/zero but L changes.
    """
    gen = torch.Generator().manual_seed(seed)
    loss = loss_fn()
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    au, fd = [], []
    for _ in range(n_probe):
        vs = [torch.randn(p.shape, generator=gen, dtype=torch.float64).to(p.dtype) for p in params]
        norm = torch.sqrt(sum((v.double() ** 2).sum() for v in vs))
        vs = [v / norm for v in vs]
        dir_auto = sum(((g if g is not None else torch.zeros_like(p)).double() * v.double()).sum()
                       for g, p, v in zip(grads, params, vs))
        with torch.no_grad():
            for p, v in zip(params, vs):
                p.add_(h * v)
            lp = float(loss_fn())
            for p, v in zip(params, vs):
                p.add_(-2 * h * v)
            lm = float(loss_fn())
            for p, v in zip(params, vs):
                p.add_(h * v)
        au.append(float(dir_auto))
        fd.append((lp - lm) / (2 * h))
    return _result(name, torch.tensor(au, dtype=torch.float64), torch.tensor(fd, dtype=torch.float64), rtol, atol)


def check_no_detached_path(loss: torch.Tensor, params: list[torch.nn.Parameter]) -> dict:
    """Report parameters with None/zero/non-finite gradient for a loss."""
    grads = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True)
    none = [i for i, g in enumerate(grads) if g is None]
    zero = [i for i, g in enumerate(grads) if g is not None and float(g.abs().sum()) == 0.0]
    bad = [i for i, g in enumerate(grads) if g is not None and not bool(torch.isfinite(g).all())]
    return {"n_params": len(params), "none": none, "zero": zero, "nonfinite": bad,
            "ok": not bad and len(none) + len(zero) < len(params)}
