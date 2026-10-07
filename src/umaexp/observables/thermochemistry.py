r"""Thermochemistry with consistent energy references.

Electronic energies are not formation enthalpies, vaporization enthalpies or
Gibbs energies. This module builds those quantities explicitly:

Reference offsets. UMA's total energy carries composition-only constants
(the task's element references, and any per-atom bias of the readout). For a
reaction sum_s nu_s E_s they cancel exactly iff the reaction is balanced in
every element (checked: `check_balance`). Equilibrium data for a fixed
composition cannot identify an additive per-element constant, and nothing
here tries to. Formation enthalpies relative to elemental standard states
(graphite, O2(g), ...) are NOT computed: graphite would need a periodic task
with a different reference theory than `omol`, and mixing task outputs is
not allowed. Use balanced reactions among species of one task instead.

Ideal-gas RRHO (per molecule, eV), differentiable in frequencies and geometry:
  H(T) = E_el + ZPE + sum_i h nu_i /(e^{x_i}-1) + H_rot + H_trans
  S    = S_trans(Sackur-Tetrode, p) + S_rot(I, sigma) + S_vib + k ln(2S+1)
with optional quasi-RRHO entropy for low-frequency modes (Grimme,
Chem. Eur. J. 18, 9955 (2012)).

Energies at optimized geometries use the envelope theorem: dE(R*(w))/dw =
dE/dw at R*, so E_el needs no implicit term; frequencies (ZPE, thermal
terms) do, and are computed at the implicitly differentiated geometry.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import torch

from .. import units
from ..potential.base import Potential
from ..reweighting.difftre import ReweightedEnsemble
from ..systems.system import System
from .base import Prediction
from .ensemble import ensemble_prediction
from .vibrational import attach_implicit_geometry, harmonic_modes, optimize_geometry

H_J_S = units.PLANCK_J_S
KB_J = units.BOLTZMANN_J_K


def check_balance(species: list[tuple[System, float]], tol: float = 1e-9) -> None:
    net: Counter = Counter()
    charge = 0.0
    for sys, nu in species:
        for el, n in sys.composition.items():
            net[el] += nu * n
        charge += nu * sys.charge
    bad = {el: v for el, v in net.items() if abs(v) > tol}
    if bad or abs(charge) > tol:
        raise ValueError(f"reaction not balanced: elements {bad}, charge {charge}")
    tasks = {sys.task for sys, _ in species}
    if len(tasks) != 1:
        raise ValueError(f"reaction mixes UMA tasks {tasks}: energies are not comparable")


def moments_of_inertia(pos: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
    m = masses[:, None]
    x = pos - (m * pos).sum(0) / masses.sum()
    inertia = torch.eye(3, dtype=pos.dtype) * (m * x * x).sum() - (m * x).T @ x
    return torch.linalg.eigvalsh(inertia)  # u A^2


def rrho(nu_cm: torch.Tensor, pos: torch.Tensor, masses: torch.Tensor, temperature_k: float,
         pressure_bar: float = 1.0, symmetry_number: int = 1, spin_multiplicity: int = 1,
         linear: bool = False, qrrho_cutoff_cm: float | None = None) -> dict[str, torch.Tensor]:
    """Ideal-gas thermal corrections (eV, eV/K); nu_cm are real vibrational wavenumbers."""
    T = temperature_k
    kT = units.KB_EV_K * T
    dt = nu_cm.dtype
    theta = nu_cm * units.WAVENUMBER_TO_EV  # h nu in eV
    x = theta / kT
    zpe = 0.5 * theta.sum()
    h_vib = (theta / torch.expm1(x)).sum()
    s_ho = units.KB_EV_K * (x / torch.expm1(x) - torch.log1p(-torch.exp(-x)))
    if qrrho_cutoff_cm:
        # free-rotor entropy with moment mu' = mu B_av / (mu + B_av)
        nu_hz = nu_cm * units.SPEED_OF_LIGHT_M_S * 100.0
        mu = H_J_S / (8 * math.pi**2 * nu_hz)
        b_av = 1e-44
        mu_p = mu * b_av / (mu + b_av)
        s_fr = units.KB_EV_K * (0.5 + torch.log(torch.sqrt(8 * math.pi**3 * mu_p * KB_J * T / H_J_S**2)))
        w = 1.0 / (1.0 + (qrrho_cutoff_cm / nu_cm) ** 4)
        s_vib = (w * s_ho + (1 - w) * s_fr).sum()
    else:
        s_vib = s_ho.sum()
    mass_kg = masses.sum() * units.DALTON_KG
    p_pa = pressure_bar * 1e5
    s_trans = units.KB_EV_K * (torch.log((2 * math.pi * mass_kg * KB_J * T / H_J_S**2) ** 1.5 * KB_J * T / p_pa) + 2.5)
    I = moments_of_inertia(pos, masses) * units.DALTON_KG * units.ANGSTROM_M**2  # kg m^2
    if masses.numel() == 1:  # atom: no rotation
        s_rot = torch.zeros((), dtype=dt)
        h_rot = torch.zeros((), dtype=dt)
    elif linear:
        s_rot = units.KB_EV_K * (torch.log(8 * math.pi**2 * I[-1] * KB_J * T / (symmetry_number * H_J_S**2)) + 1.0)
        h_rot = torch.tensor(kT, dtype=dt)
    else:
        s_rot = units.KB_EV_K * (torch.log(math.sqrt(math.pi) / symmetry_number * torch.sqrt(I.prod())
                                           * (8 * math.pi**2 * KB_J * T / H_J_S**2) ** 1.5) + 1.5)
        h_rot = torch.tensor(1.5 * kT, dtype=dt)
    s_el = torch.tensor(units.KB_EV_K * math.log(spin_multiplicity), dtype=dt)
    h_trans = torch.tensor(2.5 * kT, dtype=dt)
    h_corr = zpe + h_vib + h_rot + h_trans
    s = s_trans + s_rot + s_vib + s_el
    return {"zpe": zpe, "h_corr": h_corr, "s": s, "g_corr": h_corr - T * s}


@dataclass
class SpeciesThermo:
    system: System
    stoich: float
    symmetry_number: int = 1
    # Electronic degeneracy for S_el = k ln(2S+1). For the `omol` task the
    # System's spin input IS the multiplicity; other tasks carry spin = 0
    # ("unspecified"), so the multiplicity must be given explicitly for Gibbs
    # energies rather than guessed.
    spin_multiplicity: int | None = None


class ReactionThermochemistry:
    """Delta E_el, Delta H(T), Delta G(T) of a balanced gas-phase reaction."""

    def __init__(self, species: list[SpeciesThermo], temperature_k: float = 298.15, pressure_bar: float = 1.0,
                 quantity: str = "enthalpy", qrrho_cutoff_cm: float | None = 100.0, fmax: float = 1e-5):
        check_balance([(s.system, s.stoich) for s in species])
        if quantity not in ("electronic", "enthalpy", "gibbs", "zpe_corrected"):
            raise ValueError(quantity)
        self.species = species
        self.T, self.p = temperature_k, pressure_bar
        self.quantity = quantity
        self.qrrho = qrrho_cutoff_cm
        self.fmax = fmax
        self._geoms: dict[str, torch.Tensor] = {}

    def _multiplicity(self, sp: SpeciesThermo) -> int:
        if sp.spin_multiplicity is not None:
            return int(sp.spin_multiplicity)
        if sp.system.task == "omol":
            return int(sp.system.spin)
        if self.quantity == "gibbs":
            raise ValueError(f"{sp.system.system_id}: set SpeciesThermo.spin_multiplicity; task "
                             f"'{sp.system.task}' has no multiplicity input")
        return 1  # electronic entropy does not enter this quantity

    def predict(self, potential: Potential) -> Prediction:
        total = torch.zeros((), dtype=torch.float64)
        diag = {}
        valid = True
        for sp in self.species:
            sys = sp.system
            x_star, info = optimize_geometry(potential, sys, self._geoms.get(sys.system_id), fmax=self.fmax)
            self._geoms[sys.system_id] = x_star
            valid &= info["converged"]
            if self.quantity == "electronic":
                e = potential.energy(sys, x_star)  # envelope theorem: no implicit term needed
                total = total + sp.stoich * e
                continue
            x = attach_implicit_geometry(potential, sys, x_star)
            e = potential.energy(sys, x)
            nm = harmonic_modes(potential, sys, x, create_graph=True)
            n_imag = int((nm.wavenumbers.detach() < 0).sum())
            valid &= n_imag == 0
            th = rrho(nm.wavenumbers.clamp_min(1.0), x, sys.mass_tensor(), self.T, self.p, sp.symmetry_number,
                      self._multiplicity(sp), nm.linear, self.qrrho)
            if self.quantity == "zpe_corrected":
                total = total + sp.stoich * (e + th["zpe"])
            elif self.quantity == "enthalpy":
                total = total + sp.stoich * (e + th["h_corr"])
            else:
                total = total + sp.stoich * (e + th["g_corr"])
            diag[sys.system_id] = {"max_force": info["max_force"], "n_imag": n_imag}
        return Prediction(total.reshape(1), torch.zeros(1, dtype=torch.float64), valid=valid,
                          reason="" if valid else "unconverged geometry or imaginary modes",
                          diagnostics=diag, meta={"units": "eV", "quantity": self.quantity})


class VaporizationEnthalpy:
    """Delta_vap H = <U>_gas - <U>_liq / N_mol + RT  (eV per molecule).

    Assumes an ideal vapour and neglects p V_liq (~1e-5 eV at 1 bar); NQE and
    polarization differences between phases are not corrected here and belong
    in the model variance. Gas: single-molecule NVT ensemble; liquid: NPT.
    """

    def __init__(self, n_molecules: int, temperature_k: float):
        self.n = n_molecules
        self.T = temperature_k

    def predict(self, potential: Potential, gas: ReweightedEnsemble, liquid: ReweightedEnsemble,
                mode: str = "graph") -> Prediction:
        def energy_fn(p, c, U):
            return U[:, None]

        energy_fn.uses_energy = True
        pg = ensemble_prediction(gas, potential, energy_fn, None, mode)
        pl = ensemble_prediction(liquid, potential, energy_fn, None, mode)
        from .ensemble_combine import combine

        return combine([pg, pl], [1.0, -1.0 / self.n], const=units.KB_EV_K * self.T,
                       meta={"units": "eV/molecule"})
