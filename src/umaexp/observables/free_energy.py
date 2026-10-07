r"""Solvation / hydration / partition free energies via thermodynamic cycles.

UMA has no separable solute-solvent interaction term and ML potentials are
not designed for overlapping (alchemically decoupled) atoms, so the
decoupling legs are run with a reference potential (classical MM) and UMA
enters through endpoint corrections:

    dG_solv(UMA) = dG_solv(ref) + dG_{ref->UMA}(solution) - dG_{ref->UMA}(gas)

    dG_{ref->UMA}(env) = f_t(w)  with target u_t = beta [U_ref - U_ref,region + U_w,region]

computed from reference-potential samples by `target_state_free_energy`
(EXP is the one-state case), whose gradient <du_t/dw>_t is a valid training
gradient. Two variants:
  mlmm  region = solute; UMA replaces only the solute's intramolecular energy
        (solute-solvent interactions stay MM; UMA's intermolecular physics is
        not probed)
  full  region = whole box (requires periodic UMA on liquids; see density.py)

Standard states: FreeSolv hydration free energies use equal concentrations
in both phases (Ben-Naim), which is what the decoupling cycle computes; other
conventions need explicit corrections (`standard_state_shift`).

The reference decoupling calculation (dG_solv(ref)) and its samples are
produced by an external engine (OpenMM + a small-molecule force field);
this package does not bundle one. Overlap between reference and UMA
endpoints is the dominant failure mode; check `ess_fraction` and use
intermediate states (MBAR target) when it is poor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .. import units
from ..potential.base import Potential
from ..reweighting.free_energy import target_state_free_energy
from ..sampling.cache import SampleCache
from ..systems.system import System
from .base import Prediction


def subsystem(system: System, indices: list[int], system_id: str | None = None) -> System:
    idx = list(indices)
    pos = None if system.positions is None else np.asarray(system.positions)[idx]
    return System(
        system_id=system_id or f"{system.system_id}[sub{len(idx)}]",
        atomic_numbers=tuple(system.atomic_numbers[i] for i in idx),
        masses=tuple(system.masses[i] for i in idx),
        charge=system.charge if len(idx) == system.n_atoms else 0,
        spin=system.spin,
        task=system.task,
        pbc=(False, False, False),
        positions=pos,
        mass_convention=system.mass_convention,
        provenance={"parent": system.system_id, "indices": idx},
    )


@dataclass
class EndpointCorrection:
    """dG_{ref->UMA} in one environment from reference-potential samples.

    cache.u_sampling: total reference energy U_ref per frame (eV).
    cache.meta["u_ref_region"]: reference energy of the UMA region alone
      (e.g. MM intramolecular energy of the solute) per frame (eV).
    """

    cache: SampleCache
    region_system: System
    region_indices: list[int]
    temperature_k: float
    chunk_size: int = 64

    def predict(self, potential: Potential) -> Prediction:
        beta = units.beta(self.temperature_k)
        u_ref = torch.as_tensor(self.cache.u_sampling, dtype=torch.float64)
        u_ref_region = torch.as_tensor(np.asarray(self.cache.meta["u_ref_region"]), dtype=torch.float64)
        pos = self.cache.positions_tensor()[:, self.region_indices]
        e_uma = torch.cat([potential.energy(self.region_system, pos[i:i + self.chunk_size])
                           for i in range(0, pos.shape[0], self.chunk_size)])
        u_target = beta * (u_ref - u_ref_region + e_uma)
        u_sampled = (beta * u_ref)[None, :]
        f_t = target_state_free_energy(u_sampled, [u_ref.numel()], u_target)
        du = (u_target - beta * u_ref).detach()
        la = -du - torch.logsumexp(-du, 0)
        ess = float(1.0 / (la.exp() ** 2).sum()) / du.numel()
        return Prediction(f_t.reshape(1) / beta, None, valid=ess > 0.05,
                          reason="" if ess > 0.05 else f"poor overlap (ESS fraction {ess:.3f})",
                          diagnostics={"ess_fraction": ess, "beta_du_std": float(du.std())},
                          meta={"units": "eV"})


@dataclass
class SolvationCycle:
    dg_reference_ev: float
    dg_reference_sd_ev: float
    solution: EndpointCorrection
    gas: EndpointCorrection
    standard_state_shift_ev: float = 0.0

    def predict(self, potential: Potential) -> Prediction:
        ps = self.solution.predict(potential)
        pg = self.gas.predict(potential)
        value = self.dg_reference_ev + ps.value - pg.value + self.standard_state_shift_ev
        return Prediction(value, torch.tensor([self.dg_reference_sd_ev**2], dtype=torch.float64),
                          valid=ps.valid and pg.valid, reason="; ".join(r for r in (ps.reason, pg.reason) if r),
                          diagnostics={"solution": ps.diagnostics, "gas": pg.diagnostics},
                          meta={"units": "eV"})


def standard_state_shift(c_gas_M: float, c_solution_M: float, temperature_k: float) -> float:
    """kT ln(c_gas/c_solution) to convert between concentration conventions (eV)."""
    return units.KB_EV_K * temperature_k * math.log(c_gas_M / c_solution_M)
