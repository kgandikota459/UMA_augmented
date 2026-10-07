"""Potential interface: (system, positions, cell) -> differentiable energy.

    U_w(Z, R, Q, S, h, b, d)

w are trainable weights; Z, Q, S, b and the UMA task d come from the static
`System`; R (positions) and h (cell) are tensor inputs so autograd can reach
coordinate, cell and parameter derivatives from the same graph.

Contract for every implementation:
  * `energy` returns float64 eV of shape () for positions (N, 3) or (B,) for
    a batch of configurations (B, N, 3) of the same system;
  * the result is differentiable w.r.t. positions, cell and all trainable
    parameters (no detach, no numpy round trip, no no_grad);
  * forces are *derived* from it (conservative), never predicted separately;
  * the map is deterministic (dropout disabled) so the same weights define one
    energy surface for sampling, reweighting and training.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import torch
from torch import nn

from ..systems.system import System


class Potential(nn.Module, ABC):
    #: True only for potentials whose trainable weights are UMA's own weights.
    updates_uma_weights: bool = False

    @abstractmethod
    def energy(
        self,
        system: System,
        positions: torch.Tensor,
        cell: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Total potential energy in eV (float64)."""

    # ------------------------------------------------------------------
    def energy_and_forces(
        self,
        system: System,
        positions: torch.Tensor,
        cell: torch.Tensor | None = None,
        create_graph: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Energy and conservative forces F = -dU/dR.

        With create_graph=False the forces are detached from parameter
        gradients (sampling use). With create_graph=True they stay attached so
        force-based losses or Hessians can be differentiated further.
        """
        pos = positions if positions.requires_grad else positions.detach().requires_grad_(True)
        with torch.enable_grad():
            e = self.energy(system, pos, cell)
            (g,) = torch.autograd.grad(e.sum(), pos, create_graph=create_graph)
        if not create_graph:
            e = e.detach()
        return e, -g

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [(n, p) for n, p in self.named_parameters() if p.requires_grad]

    def parameter_fingerprint(self, trainable_only: bool = False) -> str:
        """SHA-256 over parameter values; identifies the energy surface.

        Sample caches record the fingerprint of the potential that generated
        them. `trainable_only=True` is cheaper and sufficient when the frozen
        part is identified by `base_fingerprint`.
        """
        h = hashlib.sha256()
        h.update(getattr(self, "base_fingerprint", "").encode())
        items = self.named_parameters()
        for name, p in sorted(items, key=lambda kv: kv[0]):
            if trainable_only and not p.requires_grad:
                continue
            h.update(name.encode())
            h.update(p.detach().to("cpu").to(torch.float64).contiguous().numpy().tobytes())
        for name, b in sorted(self.named_buffers(), key=lambda kv: kv[0]):
            if b.is_floating_point() and not trainable_only:
                h.update(name.encode())
                h.update(b.detach().to("cpu").to(torch.float64).contiguous().numpy().tobytes())
        return h.hexdigest()[:20]


def as_batch(positions: torch.Tensor, cell: torch.Tensor | None):
    """Normalize (N,3)|(B,N,3) and (3,3)|(B,3,3)|None to batched tensors."""
    single = positions.dim() == 2
    pos = positions.unsqueeze(0) if single else positions
    if cell is not None:
        c = cell.unsqueeze(0) if cell.dim() == 2 else cell
        if c.shape[0] == 1 and pos.shape[0] > 1:
            c = c.expand(pos.shape[0], 3, 3)
    else:
        c = None
    return pos, c, single


def minimum_image(dr: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """Minimum-image displacement for (..., 3) vectors in a (3, 3) cell.

    Integer image shifts are detached; the result stays differentiable
    w.r.t. both the displacements and the cell.
    """
    inv = torch.linalg.inv(cell)
    frac = dr @ inv
    shift = torch.round(frac).detach()
    return dr - shift @ cell


def wrap_positions(pos: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """Wrap (B, N, 3) positions into (B, 3, 3) cells; differentiable (floor detached)."""
    inv = torch.linalg.inv(cell)
    frac = torch.einsum("bni,bij->bnj", pos, inv)
    n = torch.floor(frac).detach()
    return pos - torch.einsum("bni,bij->bnj", n, cell)
