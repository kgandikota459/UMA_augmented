r"""Host-guest and protein-ligand binding: conversions, guards and cycle bookkeeping.

Advanced branch, not the starting application. Interaction energies are not
affinities; an affinity requires a free-energy cycle with restraints, a
standard state, protonation/tautomer and receptor states, and adequate
sampling. The UMA-dependent part uses the same endpoint corrections as
solvation (free_energy.EndpointCorrection) for the complex and the free
ligand; decoupling legs run with an external reference engine.

    dG_bind = dG_decouple(solvent) - dG_decouple(complex) - dG_restraint + corrections
    dG deg  = R T ln(K_d / c deg),  c deg = 1 M     (only for true equilibrium K_d)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .. import units

ACCEPTED_DIRECT = {"Kd", "Ka"}
NEEDS_ASSAY_MODEL = {"Ki", "IC50", "EC50", "Kb"}


class AssayModelRequired(ValueError):
    pass


def affinity_to_dg(value_M: float, kind: str, temperature_k: float, standard_M: float = 1.0,
                   assay: dict | None = None) -> float:
    """Binding free energy (eV) from an equilibrium constant with explicit guards.

    Kd/Ka are converted directly. Ki converts only with an assay model stating
    competitive binding against a measured Kd-like quantity; IC50/EC50 need a
    mechanism and assay parameters (e.g. Cheng-Prusoff: Ki = IC50 / (1 + [S]/Km)
    for competitive inhibition), otherwise they are rejected.
    """
    kT = units.KB_EV_K * temperature_k
    if kind == "Kd":
        return kT * math.log(value_M / standard_M)
    if kind == "Ka":
        return -kT * math.log(value_M * standard_M)
    if kind in NEEDS_ASSAY_MODEL:
        if not assay:
            raise AssayModelRequired(f"{kind} cannot be converted to dG without an assay model")
        if kind == "IC50" and assay.get("mechanism") == "competitive" and {"S_M", "Km_M"} <= assay.keys():
            ki = value_M / (1.0 + assay["S_M"] / assay["Km_M"])
            return kT * math.log(ki / standard_M)
        if kind == "Ki" and assay.get("mechanism") == "competitive":
            return kT * math.log(value_M / standard_M)
        raise AssayModelRequired(f"unsupported assay model for {kind}: {assay}")
    raise ValueError(kind)


@dataclass
class BoreschRestraint:
    """Analytical standard-state correction for Boresch orientational restraints.

    Boresch et al., J. Phys. Chem. B 107, 9535 (2003), Eq. 32. Force constants
    in eV/A^2 (distance) and eV/rad^2 (angles, dihedrals); r0 in A.
    """

    r0: float
    theta_a0: float
    theta_b0: float
    k_r: float
    k_theta_a: float
    k_theta_b: float
    k_phi_a: float
    k_phi_b: float
    k_phi_c: float

    def dg_release(self, temperature_k: float) -> float:
        """Free energy of releasing the restraint to the standard state (eV)."""
        kT = units.KB_EV_K * temperature_k
        v0 = 1.0 / units.STANDARD_CONCENTRATION_PER_A3  # A^3 per molecule at 1 M
        num = 8 * math.pi**2 * v0 * math.sqrt(self.k_r * self.k_theta_a * self.k_theta_b
                                              * self.k_phi_a * self.k_phi_b * self.k_phi_c)
        den = self.r0**2 * math.sin(self.theta_a0) * math.sin(self.theta_b0) * (2 * math.pi * kT) ** 3
        return -kT * math.log(num / den)


@dataclass
class BindingCycle:
    dg_decouple_solvent: float
    dg_decouple_complex: float
    restraint: BoreschRestraint | None
    temperature_k: float
    corrections: float = 0.0

    def dg_bind(self) -> float:
        dgr = self.restraint.dg_release(self.temperature_k) if self.restraint else 0.0
        return self.dg_decouple_solvent - self.dg_decouple_complex - dgr + self.corrections
