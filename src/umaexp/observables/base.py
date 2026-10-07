"""Observable-family registry and shared prediction types.

    y_hat_k = g_k[U_w, S, c_k; phi_k]

Each family module implements a forward model g_k with an explicit gradient
route to w. The registry records, per family, what is implemented here, what
needs an auxiliary model or an external engine, and what is out of scope for
a ground-state potential, so no family is silently dropped.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import torch


class Family(str, Enum):
    VIBRATIONAL = "vibrational"
    VDOS = "vibrational_density_of_states"
    DENSITY = "density"
    EQUATION_OF_STATE = "equation_of_state"
    FLUCTUATION = "fluctuation_response"  # compressibility, expansion, Cp, elastic
    THERMOCHEMISTRY = "thermochemistry"
    STRUCTURE = "structure"  # distances, RDFs, populations
    SCATTERING = "scattering"  # SAXS/SANS/diffraction
    NMR = "nmr"
    SOLVATION = "solvation_free_energy"
    BINDING = "binding_free_energy"
    SPECTRAL_RESPONSE = "ir_raman_dielectric"
    KINETICS = "kinetics_transport"
    ELECTRONIC = "electronic_state"  # pKa, redox, IE/EA, excited states


class Status(str, Enum):
    IMPLEMENTED = "implemented"  # forward model + validated gradient route here
    IMPLEMENTED_WITH_AUX = "implemented_requires_auxiliary_model"
    EVAL_ONLY = "implemented_evaluation_only_no_training_gradient"
    INTERFACE_ONLY = "interface_only_requires_external_engine_or_data"
    OUT_OF_SCOPE = "not_recoverable_from_ground_state_potential_alone"


class GradientRoute(str, Enum):
    DIRECT = "direct"  # derivative of U_w at fixed configurations
    IMPLICIT = "implicit"  # through an optimized geometry / equilibrium condition
    REWEIGHTING = "reweighting"  # ensemble average via DiffTRe-style weights
    FREE_ENERGY = "free_energy"  # differentiable FEP/BAR/MBAR/TI
    PATH = "path"  # dynamical estimators (unrolled or path reweighting)
    NONE = "none"


@dataclass
class Prediction:
    """Forward-model output for one observation.

    `value` carries the autograd graph to the potential's parameters.
    `variance` is the forward-model/sampling variance to be added to the
    measurement covariance (same shape as value, or full covariance).
    """

    value: torch.Tensor
    variance: torch.Tensor | None = None
    valid: bool = True
    reason: str = ""
    diagnostics: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    # For memory-bounded estimators whose `value` is detached: accumulates
    # dL/dw into parameter .grad given dL/dvalue. None -> backprop `value`.
    backward_fn: Callable[[torch.Tensor], None] | None = None

    def backward(self, grad_value: torch.Tensor) -> None:
        if self.backward_fn is not None:
            self.backward_fn(grad_value)
        elif self.value.requires_grad:
            self.value.backward(grad_value)


@dataclass(frozen=True)
class FamilySpec:
    family: Family
    status: Status
    route: GradientRoute
    module: str
    shares_hamiltonian: bool  # fully determined by U_w (+ masses/state)?
    requires: tuple[str, ...] = ()
    notes: str = ""


REGISTRY: dict[Family, FamilySpec] = {
    Family.VIBRATIONAL: FamilySpec(
        Family.VIBRATIONAL, Status.IMPLEMENTED, GradientRoute.IMPLICIT,
        "umaexp.observables.vibrational", True,
        ("optimized geometry", "isotope masses", "mode assignment",
         "anharmonic/NQE forward model or harmonic targets"),
        "Harmonic frequencies from the projected mass-weighted Hessian; geometry "
        "response via implicit differentiation (or labeled fixed-geometry).",
    ),
    Family.VDOS: FamilySpec(
        Family.VDOS, Status.EVAL_ONLY, GradientRoute.PATH,
        "umaexp.observables.kinetics", True, ("physical dynamics", "velocity autocorrelation"),
        "VDOS from velocity autocorrelation is a dynamical quantity; the harmonic "
        "VDOS (histogram of Hessian eigenvalues) is available through vibrational.py.",
    ),
    Family.DENSITY: FamilySpec(
        Family.DENSITY, Status.IMPLEMENTED, GradientRoute.REWEIGHTING,
        "umaexp.observables.density", True, ("NPT sampling", "periodic-capable UMA task"),
        "rho = M/<V> from NPT samples; NVT density is not a learnable response.",
    ),
    Family.EQUATION_OF_STATE: FamilySpec(
        Family.EQUATION_OF_STATE, Status.IMPLEMENTED, GradientRoute.REWEIGHTING,
        "umaexp.observables.density", True, ("NPT or NVT+virial sampling",),
    ),
    Family.FLUCTUATION: FamilySpec(
        Family.FLUCTUATION, Status.IMPLEMENTED, GradientRoute.REWEIGHTING,
        "umaexp.observables.fluctuation", True,
        ("long NPT trajectories", "kinetic and quantum corrections for Cp"),
        "kappa_T, alpha_P, Cp from fluctuation formulas; classical nuclei only.",
    ),
    Family.THERMOCHEMISTRY: FamilySpec(
        Family.THERMOCHEMISTRY, Status.IMPLEMENTED, GradientRoute.IMPLICIT,
        "umaexp.observables.thermochemistry", True,
        ("consistent energy reference across species", "RRHO/thermal corrections",
         "phase corrections"),
        "Reaction energies with composition-consistent offsets; ideal-gas RRHO.",
    ),
    Family.STRUCTURE: FamilySpec(
        Family.STRUCTURE, Status.IMPLEMENTED, GradientRoute.REWEIGHTING,
        "umaexp.observables.structure", True, ("basin coverage",),
    ),
    Family.SCATTERING: FamilySpec(
        Family.SCATTERING, Status.IMPLEMENTED_WITH_AUX, GradientRoute.REWEIGHTING,
        "umaexp.observables.scattering", True,
        ("form factors", "solvent/background/contrast model", "nuisance scale/offset"),
    ),
    Family.NMR: FamilySpec(
        Family.NMR, Status.IMPLEMENTED_WITH_AUX, GradientRoute.REWEIGHTING,
        "umaexp.observables.nmr", False,
        ("explicit measurement model (Karplus, r^-6 NOE, shift predictor)",
         "exchange regime"),
        "Geometry-dependent NMR observables through a frozen/validated forward model.",
    ),
    Family.SOLVATION: FamilySpec(
        Family.SOLVATION, Status.IMPLEMENTED_WITH_AUX, GradientRoute.FREE_ENERGY,
        "umaexp.observables.free_energy", True,
        ("thermodynamic cycle", "MM or alchemical reference sampling",
         "standard-state conventions"),
        "Differentiable EXP/BAR/MBAR are implemented; the MM decoupling legs "
        "require an external engine (OpenMM) not bundled here.",
    ),
    Family.BINDING: FamilySpec(
        Family.BINDING, Status.INTERFACE_ONLY, GradientRoute.FREE_ENERGY,
        "umaexp.observables.binding", True,
        ("restraints", "standard state", "protonation/receptor states", "assay model"),
        "Advanced branch; Kd -> dG conversion and cycle bookkeeping implemented.",
    ),
    Family.SPECTRAL_RESPONSE: FamilySpec(
        Family.SPECTRAL_RESPONSE, Status.IMPLEMENTED_WITH_AUX, GradientRoute.IMPLICIT,
        "umaexp.observables.response", False,
        ("dipole / polarizability model",),
        "Energy-only UMA provides no dipoles; intensities need an auxiliary model.",
    ),
    Family.KINETICS: FamilySpec(
        Family.KINETICS, Status.EVAL_ONLY, GradientRoute.PATH,
        "umaexp.observables.kinetics", True,
        ("physical dynamics", "thermostat-free or validated dynamics", "timescale convergence"),
        "Equilibrium reweighting is not a kinetic gradient estimator; short-window "
        "unrolled gradients are provided as a research option.",
    ),
    Family.ELECTRONIC: FamilySpec(
        Family.ELECTRONIC, Status.IMPLEMENTED_WITH_AUX, GradientRoute.IMPLICIT,
        "umaexp.observables.electronic", True,
        ("multiple charge/spin states (omol task)", "solvation model for pKa/redox",
         "excited states out of scope"),
        "Gas-phase adiabatic IE/EA from charge/spin-conditioned UMA; solution "
        "pKa/redox need an explicit solvation cycle.",
    ),
}


class ObservableError(RuntimeError):
    pass


class ExtensionRequired(ObservableError):
    """Raised when a family needs a component this build does not provide."""
