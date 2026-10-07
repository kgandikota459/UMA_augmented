r"""Density and equation of state from NPT ensembles.

Estimator (default): rho = M / <V>_{NPT}. The macroscopic density of the
measured sample is mass over thermodynamic volume, and <V> is the NPT
estimate of that volume. The alternative <M/V> differs by
O(Var V / <V>^2) = O(k_B T kappa_T / <V>), ~1e-4 relative for a few hundred
water molecules (kappa_T = 4.5e-10 1/Pa at 298 K), far below typical
potential errors, but both are available and the choice is recorded.

A fixed-volume (NVT) simulation cannot respond to the potential through the
volume, so density is only learnable from NPT (or pressure-matching) data.

Periodic molecular liquids: the UMA `omol` task was trained on aperiodic data
(FAIRChem UMA docs). Using it for liquids requires the explicit
acknowledgement stored on the System, and the validity must be established
on held-out liquids before claims are made.
"""

from __future__ import annotations

import torch

from .. import units
from ..potential.base import Potential
from ..potential.derivatives import virial_pressure
from ..reweighting.difftre import ReweightedEnsemble
from ..systems.system import System
from .base import Prediction
from .ensemble import ensemble_prediction


def volume_frames(p, c):
    return torch.linalg.det(c).abs()[:, None]


class DensityObservable:
    def __init__(self, system: System, estimator: str = "mass_over_mean_volume"):
        if not system.is_periodic:
            raise ValueError("density needs a periodic system")
        if estimator not in ("mass_over_mean_volume", "mean_mass_over_volume"):
            raise ValueError(estimator)
        self.system = system
        self.estimator = estimator
        self.mass = system.total_mass  # u

    def predict(self, potential: Potential, ens: ReweightedEnsemble, mode: str = "graph") -> Prediction:
        if ens.target.ensemble != "NPT":
            raise ValueError("density is not a learnable response in NVT; use NPT samples")
        conv = units.U_PER_A3_TO_G_CM3
        if self.estimator == "mass_over_mean_volume":
            return ensemble_prediction(ens, potential, volume_frames, lambda v: self.mass / v * conv, mode,
                                       meta={"estimator": self.estimator, "units": "g/cm^3"})
        return ensemble_prediction(ens, potential, lambda p, c: self.mass / torch.linalg.det(c).abs()[:, None] * conv,
                                   None, mode, meta={"estimator": self.estimator, "units": "g/cm^3"})


class VirialPressureObservable:
    """<P> in an NVT ensemble at fixed volume (pressure matching / EOS points).

    The frame observable depends on w explicitly (the virial is a derivative
    of U_w), so both the explicit and the distribution terms are
    differentiated. Units: bar.
    """

    def __init__(self, system: System, temperature_k: float):
        self.system = system
        self.T = temperature_k

    def predict(self, potential: Potential, ens: ReweightedEnsemble, mode: str = "graph") -> Prediction:
        sys, T = self.system, self.T

        def frame_fn(p, c):
            out = []
            for b in range(p.shape[0]):
                out.append(virial_pressure(potential, sys, p[b], c[b], T, create_graph=torch.is_grad_enabled()))
            return torch.stack(out)[:, None] / units.BAR_TO_EV_A3

        frame_fn.depends_on_w = True
        return ensemble_prediction(ens, potential, frame_fn, None, mode, meta={"units": "bar"})
