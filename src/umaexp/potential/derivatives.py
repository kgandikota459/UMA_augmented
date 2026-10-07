"""Derivative engine: energy -> forces, Hessians, strain derivatives.

Every function keeps the autograd graph to the potential's parameters when
`create_graph=True`, which is what makes losses on forces, frequencies or
pressures differentiable w.r.t. UMA's weights (mixed coordinate/parameter
derivatives such as d/dw d^2U/dR^2).

Verified on the UMA-S architecture (fairchem-core 2.23.0, general execution
backend): autograd Hessians match central finite differences of forces and
third-order parameter derivatives are non-zero for all coordinate-dependent
parameter tensors (see validation/audit.py).
"""

from __future__ import annotations

import torch

from ..systems.system import System
from .base import Potential


def forces(potential: Potential, system: System, positions, cell=None, create_graph=False):
    return potential.energy_and_forces(system, positions, cell, create_graph=create_graph)[1]


def hessian(
    potential: Potential,
    system: System,
    positions: torch.Tensor,
    cell: torch.Tensor | None = None,
    create_graph: bool = True,
    method: str = "rows",
) -> torch.Tensor:
    """Cartesian Hessian d^2U/dR^2 of shape (3N, 3N), eV/A^2.

    `positions` may itself depend on parameters (e.g. an implicitly
    differentiated optimized geometry); that dependence is kept.

    method="rows": one backward per row (lowest memory, always supported).
    method="vmap": batched backward via torch.vmap (faster; requires every
    op in the potential to have batching rules for double backward).
    """
    pos = positions if positions.requires_grad else positions.detach().requires_grad_(True)
    with torch.enable_grad():
        e = potential.energy(system, pos, cell)
        (g,) = torch.autograd.grad(e, pos, create_graph=True)
        g = g.reshape(-1)
        n = g.numel()
        if method == "vmap":
            eye = torch.eye(n, dtype=g.dtype, device=g.device)

            def row(v):
                return torch.autograd.grad(g, pos, grad_outputs=v, retain_graph=True,
                                           create_graph=create_graph)[0].reshape(-1)

            H = torch.vmap(row)(eye)
        elif method == "rows":
            rows = [
                torch.autograd.grad(g[i], pos, retain_graph=True, create_graph=create_graph)[0].reshape(-1)
                for i in range(n)
            ]
            H = torch.stack(rows)
        else:
            raise ValueError(method)
    # Symmetrize: exact Hessians are symmetric; floating-point asymmetry would
    # otherwise leak into eigenvalue derivatives.
    return 0.5 * (H + H.T)


def strain_derivative(
    potential: Potential,
    system: System,
    positions: torch.Tensor,
    cell: torch.Tensor,
    create_graph: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """dU/d(epsilon) for a homogeneous strain R' = R(I+eps), h' = h(I+eps).

    Returns (energy, dU/deps) with dU/deps of shape (3, 3) or (B, 3, 3).
    The (atomic) virial stress is sigma = (1/V) dU/deps; the instantaneous
    virial pressure is P = N k_B T / V - tr(dU/deps) / (3V).
    """
    pos_b = positions.unsqueeze(0) if positions.dim() == 2 else positions
    cell_b = cell.unsqueeze(0) if cell.dim() == 2 else cell
    B = pos_b.shape[0]
    eps = torch.zeros(B, 3, 3, dtype=pos_b.dtype, device=pos_b.device, requires_grad=True)
    with torch.enable_grad():
        deform = torch.eye(3, dtype=pos_b.dtype, device=pos_b.device) + 0.5 * (eps + eps.transpose(1, 2))
        pos_s = torch.einsum("bni,bij->bnj", pos_b, deform)
        cell_s = torch.einsum("bki,bij->bkj", cell_b, deform)
        e = potential.energy(system, pos_s if positions.dim() == 3 else pos_s[0],
                             cell_s if cell.dim() == 3 else cell_s[0])
        (de,) = torch.autograd.grad(e.sum(), eps, create_graph=create_graph)
    if positions.dim() == 2:
        de = de[0]
    return (e if create_graph else e.detach()), de


def virial_pressure(
    potential: Potential,
    system: System,
    positions: torch.Tensor,
    cell: torch.Tensor,
    temperature_k: float,
    create_graph: bool = False,
) -> torch.Tensor:
    """Instantaneous pressure (eV/A^3) with the ideal-gas kinetic term at T.

    P = N k_B T / V - tr(dU/deps) / (3 V). Uses the equipartition kinetic
    term (appropriate for canonical averages over configurations); a
    molecular virial would use N_molecules instead with rigid-body forces.
    """
    from ..units import KB_EV_K

    _, de = strain_derivative(potential, system, positions, cell, create_graph=create_graph)
    vol = torch.linalg.det(cell).abs()
    tr = de.diagonal(dim1=-2, dim2=-1).sum(-1)
    return system.n_atoms * KB_EV_K * temperature_k / vol - tr / (3.0 * vol)


def hessian_quadratic_form_backward(
    potential: Potential,
    system: System,
    positions: torch.Tensor,
    directions: torch.Tensor,
    coefficients: torch.Tensor,
    cell: torch.Tensor | None = None,
) -> None:
    """Accumulate d/dw [ sum_k c_k u_k^T H(w) u_k ] into parameter .grad.

    Memory-bounded route for Hessian-based losses: if L depends on w only
    through H, then dL/dw = sum_ij G_ij dH_ij/dw with G = dL/dH (detached).
    Writing G = sum_k c_k u_k u_k^T, each term is a second directional
    derivative u^T H u computed with one double-backward graph at a time,
    instead of keeping 3N row graphs alive. `positions` may carry a graph to
    w (implicit geometry); it is retained across directions.

    directions: (K, N, 3) Cartesian; coefficients: (K,).
    """
    keep = positions.requires_grad and positions.grad_fn is not None
    pos = positions if positions.requires_grad else positions.detach().requires_grad_(True)
    K = directions.shape[0]
    for k in range(K):
        c = float(coefficients[k])
        if c == 0.0:
            continue
        u = directions[k].to(pos.dtype)
        with torch.enable_grad():
            e = potential.energy(system, pos, cell)
            (g,) = torch.autograd.grad(e, pos, create_graph=True)
            (hu,) = torch.autograd.grad((g * u).sum(), pos, create_graph=True)
            q = (hu * u).sum()
        (c * q).backward(retain_graph=keep)
