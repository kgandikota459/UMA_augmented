r"""Scattering forward models averaged over configurations.

SAXS/WAXS (Debye):   I(q) = sum_ij f_i(q) f_j(q) sin(q r_ij)/(q r_ij)
Neutron / X-ray total structure factor of liquids from partial RDFs:
                     S(q) - 1 = sum_ab c_a c_b w_ab(q) 4 pi rho int r^2 [g_ab(r) - 1] sinc(qr) L(r) dr

Status: the Debye sum and structure-factor transforms are implemented and
differentiable through the ensemble weights. Quantitative solution SAXS
additionally needs an excluded-volume and hydration-shell model (CRYSOL,
FoXS, WAXSiS-class) and buffer subtraction consistent with the experiment;
that is an auxiliary forward model supplied by the caller (`solvent_model`).
Scale and constant background are nuisance parameters, profiled out by
weighted least squares (closed form, differentiable) unless given priors.
Measured curves have correlated errors across q; supply a covariance when
the deposition provides one.
"""

from __future__ import annotations

import math

import torch

from ..potential.base import Potential
from ..reweighting.difftre import ReweightedEnsemble
from ..systems.system import System
from .base import Prediction
from .ensemble import ensemble_prediction

# Cromer-Mann coefficients (International Tables for Crystallography Vol. C,
# Table 6.1.1.4): f(s) = sum_i a_i exp(-b_i s^2) + c, s = q / (4 pi) in 1/A.
CROMER_MANN = {
    1: ([0.489918, 0.262003, 0.196767, 0.049879], [20.6593, 7.74039, 49.5519, 2.20159], 0.001305),
    6: ([2.31000, 1.02000, 1.58860, 0.865000], [20.8439, 10.2075, 0.568700, 51.6512], 0.215600),
    7: ([12.2126, 3.13220, 2.01250, 1.16630], [0.005700, 9.89330, 28.9975, 0.582600], -11.5290),
    8: ([3.04850, 2.28680, 1.54630, 0.867000], [13.2771, 5.70110, 0.323900, 32.9089], 0.250800),
    15: ([6.43450, 4.17910, 1.78000, 1.49080], [1.90670, 27.1570, 0.526000, 68.1645], 1.11490),
    16: ([6.90530, 5.20340, 1.43790, 1.58630], [1.46790, 22.2151, 0.253600, 56.1720], 0.866900),
}

# Bound coherent neutron scattering lengths (fm), Sears, Neutron News 3, 26 (1992).
NEUTRON_B_FM = {(1, 1): -3.7390, (1, 2): 6.671, (6, 12): 6.6460, (7, 14): 9.36, (8, 16): 5.803,
                (15, 31): 5.13, (16, 32): 2.847}


def xray_form_factors(atomic_numbers, q: torch.Tensor) -> torch.Tensor:
    """(n_q, N) vacuum atomic form factors (electrons)."""
    s2 = (q / (4 * math.pi)) ** 2
    cols = []
    for z in atomic_numbers:
        if z not in CROMER_MANN:
            raise KeyError(f"no Cromer-Mann coefficients for Z={z}")
        a, b, c = CROMER_MANN[z]
        f = torch.full_like(q, c)
        for ai, bi in zip(a, b):
            f = f + ai * torch.exp(-bi * s2)
        cols.append(f)
    return torch.stack(cols, -1)


class DebyeSAXS:
    def __init__(self, system: System, q: torch.Tensor, atom_indices=None, solvent_model=None):
        self.q = torch.as_tensor(q, dtype=torch.float64)
        idx = list(range(system.n_atoms)) if atom_indices is None else list(atom_indices)
        self.idx = torch.as_tensor(idx)
        z = [system.atomic_numbers[i] for i in idx]
        self.f = xray_form_factors(z, self.q)  # (n_q, n)
        self.solvent_model = solvent_model

    def frames(self, pos, cells):
        x = pos[:, self.idx]
        r = torch.cdist(x, x)  # (b, n, n)
        qr = self.q[None, :, None, None] * r[:, None]
        sinc = torch.where(qr > 1e-12, torch.sin(qr) / qr.clamp_min(1e-12), torch.ones_like(qr))
        f = self.f
        if self.solvent_model is not None:
            f = self.solvent_model(self.q, x)  # must return (n_q, n) effective amplitudes
        return torch.einsum("qi,bqij,qj->bq", f, sinc, f)

    def predict(self, potential: Potential, ens: ReweightedEnsemble, mode="graph") -> Prediction:
        return ensemble_prediction(ens, potential, self.frames, None, mode, meta={"q": self.q})


def profile_scale_offset(model: torch.Tensor, data: torch.Tensor, sigma: torch.Tensor, fit_offset: bool = True):
    """Weighted LSQ for y ~ c*I + b; returns (c*I + b, c, b), differentiable in I."""
    w = 1.0 / sigma**2
    if not fit_offset:
        c = (w * model * data).sum() / (w * model * model).sum()
        return c * model, c, torch.zeros((), dtype=model.dtype)
    S = w.sum()
    Sx, Sy = (w * model).sum(), (w * data).sum()
    Sxx, Sxy = (w * model * model).sum(), (w * model * data).sum()
    det = S * Sxx - Sx * Sx
    c = (S * Sxy - Sx * Sy) / det
    b = (Sxx * Sy - Sx * Sxy) / det
    return c * model + b, c, b


def structure_factor_from_rdf(r: torch.Tensor, g: torch.Tensor, rho: float, q: torch.Tensor,
                              lorch: bool = True) -> torch.Tensor:
    """S(q) - 1 = 4 pi rho int r^2 (g - 1) sinc(qr) L(r) dr (single partial)."""
    dr = r[1] - r[0]
    rmax = r[-1] + 0.5 * dr
    L = torch.sinc(r / rmax) if lorch else torch.ones_like(r)  # torch.sinc = sin(pi x)/(pi x)
    qr = q[:, None] * r[None, :]
    k = torch.where(qr > 1e-12, torch.sin(qr) / qr.clamp_min(1e-12), torch.ones_like(qr))
    return 4 * math.pi * rho * ((r**2 * (g - 1.0) * L)[None, :] * k).sum(-1) * dr
