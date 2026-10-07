r"""IR intensities (double-harmonic) with an auxiliary dipole model.

Energy-only UMA provides no dipole moments. IR intensities need dmu/dR from
a separate dipole model; Raman activities need polarizability derivatives;
dielectric constants need <M^2> fluctuations of a dipole model over an
ensemble. None of these is determined by the ground-state energy surface.

    I_k = (N_A pi / (3 c^2)) |d mu / d Q_k|^2,   d mu/d Q_k = sum_i (d mu / d x_i) L_ik / sqrt(m_i)

in km/mol when d mu/dQ is in e / sqrt(u) (conversion 974.88 km/mol per
e^2/u, i.e. 42.2561 km/mol per (D/A)^2/u).

Gradient to UMA: through the normal-mode vectors L and the geometry. Eigenvector
derivatives are undefined inside degenerate sets, so intensities of
degenerate sets are returned summed (basis-invariant) and marked
`gradient_supported=False`; non-degenerate, well-separated modes support
gradients. A frozen dipole model still transmits information to UMA through
the modes; jointly training a dipole model needs independent dipole labels,
otherwise it can absorb errors that belong to the potential.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from ..potential.base import Potential
from ..systems.system import System
from .base import Prediction
from .vibrational import degenerate_groups, external_projector, is_linear

# (N_A pi / 3 c^2) * e^2/u  ->  km/mol
IR_INTENSITY_KM_MOL_PER_E2_U = 974.8801


class FixedChargeDipole(torch.nn.Module):
    """mu = sum_i q_i r_i with fixed partial charges (e). A crude auxiliary model:
    it ignores charge flux and polarization, which dominate many intensities."""

    def __init__(self, charges):
        super().__init__()
        self.register_buffer("q", torch.as_tensor(charges, dtype=torch.float64))

    def forward(self, pos):
        return (self.q[:, None] * pos).sum(0)


def ir_intensities(potential: Potential, system: System, positions: torch.Tensor,
                   dipole_model: Callable[[torch.Tensor], torch.Tensor],
                   degeneracy_tol: float = 1.0) -> Prediction:
    from ..potential.derivatives import hessian
    from .. import units

    masses = system.mass_tensor()
    x = positions if positions.requires_grad else positions.detach().requires_grad_(True)
    H = hessian(potential, system, x, create_graph=True)
    n = x.shape[0]
    inv_sqrt_m = masses.rsqrt().repeat_interleave(3)
    D = inv_sqrt_m[:, None] * H * inv_sqrt_m[None, :]
    P_ext, k = external_projector(x, masses, True, is_linear(x.detach(), masses))
    P = torch.eye(3 * n, dtype=H.dtype) - P_ext
    c = float(torch.linalg.matrix_norm(D.detach())) * 4 + 10
    Dp = P @ D @ P + c * P_ext
    Dp = 0.5 * (Dp + Dp.T)
    # Detached eigendecomposition + first-order perturbative re-attachment:
    # value exact, derivative = standard eigenvector perturbation formula,
    # which only involves gaps to the (non-degenerate) mode itself. Avoids
    # eigh-backward NaNs from degenerate pairs elsewhere in the spectrum.
    evals0, evecs0 = torch.linalg.eigh(Dp.detach())
    dD = Dp - Dp.detach()
    nv = 3 * n - k
    nu0 = torch.sign(evals0[:nv]) * torch.sqrt(evals0[:nv].abs()) * units.EIGVAL_TO_WAVENUMBER
    groups = degenerate_groups(nu0.numpy(), abs_tol=degeneracy_tol)
    with torch.enable_grad():
        # dmu/dR at the (possibly implicitly differentiated) geometry, keeping
        # its geometry dependence for position-dependent dipole models
        mu = dipole_model(x)
        dmu = torch.stack([torch.autograd.grad(mu[a], x, retain_graph=True, create_graph=True)[0].reshape(-1)
                           for a in range(3)])
    vals, nus, grad_ok = [], [], []
    for g in groups:
        if len(g) == 1:
            j = g[0]
            gaps = evals0[j] - evals0
            gaps[j] = float("inf")
            coeff = (evecs0.T @ (dD @ evecs0[:, j])) / gaps
            v = evecs0[:, j] + evecs0 @ coeff
            dmu_dq = dmu @ (v * inv_sqrt_m)
            vals.append((dmu_dq**2).sum() * IR_INTENSITY_KM_MOL_PER_E2_U)
        else:
            Lg = evecs0[:, g] * inv_sqrt_m[:, None]
            vals.append(((dmu @ Lg) ** 2).sum() * IR_INTENSITY_KM_MOL_PER_E2_U)  # basis-invariant; no gradient
        nus.append(nu0[g].mean())
        grad_ok.append(len(g) == 1)
    return Prediction(torch.stack(vals), None, valid=True,
                      diagnostics={"wavenumbers": torch.stack(nus).tolist(), "gradient_supported": grad_ok},
                      meta={"units": "km/mol"})
