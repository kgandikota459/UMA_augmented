r"""Response functions from NPT fluctuations (classical nuclei).

    kappa_T = <dV^2> / (k_B T <V>)
    alpha_P = <dV dH_c> / (k_B T^2 <V>),          H_c = U + pV
    C_P     = C_P^kin + <dH_c^2> / (k_B T^2) (+ quantum correction)

The kinetic energy is independent of configuration classically, so it drops
out of alpha_P and contributes (3N/2) k_B to C_P. Classical C_P of molecular
liquids is too large because high-frequency intramolecular modes are
thermally excited classically; an optional harmonic quantum correction from
the monomer's normal modes (computed with the same U_w) is provided. It
assumes intramolecular modes are unchanged in the condensed phase; its
uncertainty must be carried in the model variance.

These estimators need long, well-converged NPT trajectories: fluctuation
observables converge far more slowly than means. Treat the statistical
inefficiency reported in diagnostics as a gate.
"""

from __future__ import annotations

import torch

from .. import units
from ..potential.base import Potential
from ..reweighting.difftre import ReweightedEnsemble
from ..systems.system import System
from .base import Prediction
from .ensemble import ensemble_prediction


class IsothermalCompressibility:
    """kappa_T in 1/GPa."""

    def __init__(self, temperature_k: float):
        self.kT = units.KB_EV_K * temperature_k

    def predict(self, potential: Potential, ens: ReweightedEnsemble, mode="graph") -> Prediction:
        def frames(p, c):
            v = torch.linalg.det(c).abs()
            return torch.stack([v, v * v], -1)

        kT = self.kT

        def T(m):
            return ((m[1] - m[0] ** 2) / (kT * m[0]) / units.EV_A3_TO_GPA).reshape(1)

        return ensemble_prediction(ens, potential, frames, T, mode, meta={"units": "1/GPa"})


class ThermalExpansion:
    """alpha_P in 1/K; frame observable uses U_w explicitly."""

    def __init__(self, temperature_k: float, pressure_bar: float):
        self.T = temperature_k
        self.p = pressure_bar * units.BAR_TO_EV_A3

    def predict(self, potential: Potential, ens: ReweightedEnsemble, mode="graph") -> Prediction:
        p_ev = self.p

        def frames(pos, c, U):
            v = torch.linalg.det(c).abs()
            h = U.to(v.dtype) + p_ev * v
            return torch.stack([v, h, v * h], -1)

        frames.uses_energy = True
        kT2 = units.KB_EV_K * self.T**2

        def T(m):
            return ((m[2] - m[0] * m[1]) / (kT2 * m[0])).reshape(1)

        return ensemble_prediction(ens, potential, frames, T, mode, meta={"units": "1/K"})


class HeatCapacityP:
    """Molar isobaric heat capacity C_P,m in J/(mol K)."""

    def __init__(self, system: System, temperature_k: float, pressure_bar: float,
                 quantum_correction_j_mol_k: torch.Tensor | float | None = None):
        self.system = system
        self.T = temperature_k
        self.p = pressure_bar * units.BAR_TO_EV_A3
        self.n_mol = system.n_molecules
        self.qc = quantum_correction_j_mol_k

    def predict(self, potential: Potential, ens: ReweightedEnsemble, mode="graph") -> Prediction:
        p_ev = self.p

        def frames(pos, c, U):
            v = torch.linalg.det(c).abs()
            h = U.to(v.dtype) + p_ev * v
            return torch.stack([h, h * h], -1)

        frames.uses_energy = True
        kT2 = units.KB_EV_K * self.T**2
        to_molar = units.EV_J * units.AVOGADRO / self.n_mol
        kin = 1.5 * self.system.n_atoms * units.KB_EV_K
        qc = self.qc

        def T(m):
            cp = ((m[1] - m[0] ** 2) / kT2 + kin) * to_molar
            if qc is not None:
                cp = cp + qc
            return cp.reshape(1)

        pred = ensemble_prediction(ens, potential, frames, T, mode, meta={"units": "J/(mol K)"})
        pred.meta["quantum_correction"] = None if qc is None else float(torch.as_tensor(qc))
        return pred


def harmonic_cv_quantum_correction(wavenumbers: torch.Tensor, temperature_k: float) -> torch.Tensor:
    """Sum over modes of [C_v^quantum(nu) - k_B] in J/(mol K) per molecule.

    Differentiable in the (UMA-derived) wavenumbers, so the correction moves
    consistently with the potential.
    """
    x = wavenumbers * units.WAVENUMBER_TO_EV / (units.KB_EV_K * temperature_k)
    cq = x**2 * torch.exp(x) / torch.expm1(x) ** 2  # in units of k_B
    return (cq - 1.0).sum() * units.GAS_CONSTANT_J_MOL_K


__all__ = [
    "HeatCapacityP",
    "IsothermalCompressibility",
    "ThermalExpansion",
    "harmonic_cv_quantum_correction",
]
