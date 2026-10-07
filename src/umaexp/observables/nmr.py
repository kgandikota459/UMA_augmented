r"""NMR observables through explicit measurement models.

Geometry alone does not determine most NMR signals. Each observable here
names its measurement model, its regime assumptions and a model
uncertainty; the model is *frozen* (not fit to the same residuals), so the
experimental information reaches U_w only through the ensemble weights:

    <J> = sum_i a_i(w) J_Karplus(phi_i)        d<J>/dw = -beta Cov(J, dU/dw)

  3J couplings   Karplus relation; parameter sets carry their source and a
                 typical RMS model error (0.4-1 Hz); dynamic averaging assumed
                 fast relative to the coupling timescale.
  NOE distances  <r^-6>^(-1/6) (slow internal motion vs tumbling, no spin
                 diffusion); <r^-3>^2 regime selectable. Full relaxation-matrix
                 analysis, spin diffusion and exchange are out of scope here.
  Chemical shifts require an auxiliary predictor (SPARTA+/ShiftX2-class or a
                 validated ML model) passed as a callable on frames.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from ..potential.base import Potential
from ..reweighting.difftre import ReweightedEnsemble
from .base import Prediction
from .ensemble import ensemble_prediction
from .structure import dihedral, pair_vectors


@dataclass(frozen=True)
class KarplusParameters:
    A: float
    B: float
    C: float
    offset_deg: float
    model_sd_hz: float
    source: str


KARPLUS = {
    # 3J(HN,HA) vs backbone phi: J = A cos^2(phi-60) + B cos(phi-60) + C
    "3J_HN_HA_VuisterBax1993": KarplusParameters(6.51, -1.76, 1.60, -60.0, 0.73,
                                                  "Vuister & Bax, JACS 115, 7772 (1993)"),
}


class JCoupling:
    def __init__(self, dihedral_atoms: tuple[int, int, int, int], params: KarplusParameters):
        self.atoms = dihedral_atoms
        self.params = params

    def frames(self, p, c):
        phi = dihedral(p, *self.atoms) + math.radians(self.params.offset_deg)
        cp = torch.cos(phi)
        return (self.params.A * cp * cp + self.params.B * cp + self.params.C)[:, None]

    def predict(self, potential: Potential, ens: ReweightedEnsemble, mode="graph") -> Prediction:
        pred = ensemble_prediction(ens, potential, self.frames, None, mode,
                                   meta={"model": self.params.source, "units": "Hz"})
        pred.variance = pred.variance + self.params.model_sd_hz**2
        return pred


class NOEDistance:
    """Effective distance for a proton pair or equivalent groups (e.g. methyls)."""

    def __init__(self, group_i: list[int], group_j: list[int], regime: str = "r-6", periodic: bool = False):
        if regime not in ("r-6", "r-3"):
            raise ValueError(regime)
        ii, jj = torch.meshgrid(torch.as_tensor(group_i), torch.as_tensor(group_j), indexing="ij")
        self.i, self.j = ii.flatten(), jj.flatten()
        self.regime = regime
        self.periodic = periodic

    def frames(self, p, c):
        d = pair_vectors(p, c, self.i, self.j, self.periodic).norm(dim=-1)
        if self.regime == "r-6":
            return (d**-6).sum(-1, keepdim=True)  # sum over equivalent pairs (r^-6 summation)
        return (d**-3).sum(-1, keepdim=True)

    def predict(self, potential, ens, mode="graph") -> Prediction:
        if self.regime == "r-6":
            T = lambda m: m ** (-1.0 / 6.0)  # noqa: E731
        else:
            T = lambda m: m ** (-1.0 / 3.0)  # noqa: E731  (<r^-3>^2)^(-1/6) == <r^-3>^(-1/3)
        return ensemble_prediction(ens, potential, self.frames, T, mode, meta={"regime": self.regime, "units": "A"})


class ChemicalShift:
    """Ensemble-averaged shifts from a frozen auxiliary predictor.

    predictor(positions (b,N,3)) -> (b, n_shifts) in ppm. Its systematic
    error (model_sd_ppm) must come from its own validation, not from this fit.
    """

    def __init__(self, predictor: Callable[[torch.Tensor], torch.Tensor], model_sd_ppm: float, name: str):
        self.predictor = predictor
        self.model_sd = model_sd_ppm
        self.name = name

    def predict(self, potential, ens, mode="graph") -> Prediction:
        pred = ensemble_prediction(ens, potential, lambda p, c: self.predictor(p).detach(), None, mode,
                                   meta={"predictor": self.name, "units": "ppm"})
        pred.variance = pred.variance + self.model_sd**2
        return pred
