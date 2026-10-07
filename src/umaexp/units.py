"""Physical constants and explicit unit conversions.

Internal (numerical-path) unit system, used by every differentiable module:

    energy       eV
    length       Angstrom
    mass         dalton (unified atomic mass unit, u)
    time         tau = Angstrom * sqrt(u / eV)  (~10.18 fs); convert with FS
    temperature  K
    pressure     eV / Angstrom^3
    wavenumber   cm^-1 (output only)

All conversions are plain float multiplications so they stay differentiable
inside torch expressions. Pint is used only at data-ingestion time
(`to_internal`) to parse the units written in experimental records; it never
touches tensors in the live training path.

Constants: SI 2019 exact values (e, h, k_B, N_A, c) and the CODATA 2018
recommended value of the dalton.
"""

from __future__ import annotations

import math

# --- SI 2019 exact constants ---
ELEMENTARY_CHARGE_C = 1.602176634e-19
PLANCK_J_S = 6.62607015e-34
BOLTZMANN_J_K = 1.380649e-23
AVOGADRO = 6.02214076e23
SPEED_OF_LIGHT_M_S = 299792458.0

# --- CODATA 2018 recommended ---
DALTON_KG = 1.66053906660e-27

# --- derived ---
EV_J = ELEMENTARY_CHARGE_C
HBAR_J_S = PLANCK_J_S / (2.0 * math.pi)
ANGSTROM_M = 1e-10
KB_EV_K = BOLTZMANN_J_K / EV_J  # 8.617333262e-5 eV/K
GAS_CONSTANT_J_MOL_K = BOLTZMANN_J_K * AVOGADRO

EV_TO_KJ_MOL = EV_J * AVOGADRO / 1000.0  # 96.485...
EV_TO_KCAL_MOL = EV_TO_KJ_MOL / 4.184  # thermochemical calorie
KJ_MOL_TO_EV = 1.0 / EV_TO_KJ_MOL
KCAL_MOL_TO_EV = 1.0 / EV_TO_KCAL_MOL
HARTREE_EV = 27.211386245988  # CODATA 2018

# Internal time unit: tau = A * sqrt(u/eV) in seconds, and fs per tau.
TAU_S = ANGSTROM_M * math.sqrt(DALTON_KG / EV_J)
FS = 1e-15 / TAU_S  # multiply femtoseconds by FS to get internal time

# Pressure: 1 bar in eV/A^3, 1 atm in eV/A^3.
BAR_TO_EV_A3 = 1e5 * ANGSTROM_M**3 / EV_J
ATM_TO_EV_A3 = 101325.0 * ANGSTROM_M**3 / EV_J
EV_A3_TO_GPA = EV_J / ANGSTROM_M**3 / 1e9

# Density: u / A^3 -> g / cm^3
U_PER_A3_TO_G_CM3 = DALTON_KG * 1e3 / (ANGSTROM_M * 1e2) ** 3

# Mass-weighted Hessian eigenvalue (eV / (A^2 u)) -> angular frequency (rad/s)
# omega = sqrt(lambda) * OMEGA_UNIT_RAD_S
OMEGA_UNIT_RAD_S = math.sqrt(EV_J / (ANGSTROM_M**2 * DALTON_KG))
# ... -> wavenumber (cm^-1): nu_tilde = omega / (2 pi c), c in cm/s
EIGVAL_TO_WAVENUMBER = OMEGA_UNIT_RAD_S / (2.0 * math.pi * SPEED_OF_LIGHT_M_S * 100.0)
# Photon energy of 1 cm^-1 in eV (h c * 100 / e)
WAVENUMBER_TO_EV = PLANCK_J_S * SPEED_OF_LIGHT_M_S * 100.0 / EV_J

# Standard concentration 1 mol/L expressed as molecules per A^3
STANDARD_CONCENTRATION_PER_A3 = AVOGADRO * 1e3 * ANGSTROM_M**3  # 1 M = 1000 mol/m^3


def beta(temperature_k: float) -> float:
    """Inverse thermal energy 1/(k_B T) in 1/eV."""
    return 1.0 / (KB_EV_K * temperature_k)


# ---------------------------------------------------------------------------
# Ingestion-time parsing with Pint (never used on tensors in training paths)
# ---------------------------------------------------------------------------
_UREG = None


def unit_registry():
    """Lazily constructed Pint registry with spectroscopy aliases."""
    global _UREG
    if _UREG is None:
        import pint

        ureg = pint.UnitRegistry()
        ureg.define("wavenumber = 1 / centimeter = cm_1")
        _UREG = ureg
    return _UREG


# Internal units by physical dimension, for ingestion conversion.
_INTERNAL_UNITS = {
    "energy": "eV",
    "molar_energy": "kJ/mol",  # converted separately (per-molecule eV below)
    "length": "angstrom",
    "temperature": "kelvin",
    "pressure": "bar",
    "density": "g/cm**3",
    "wavenumber": "1/cm",
    "time": "fs",
    "molar_volume": "cm**3/mol",
}


def to_internal(value: float, unit: str, kind: str) -> float:
    """Convert a record value to this package's canonical unit for `kind`.

    `kind` selects the canonical unit (see `_INTERNAL_UNITS`). Molar energies
    are returned per molecule in eV. Raises if the dimensions are
    incompatible, so a mislabeled unit fails loudly at curation time.
    """
    ureg = unit_registry()
    q = ureg.Quantity(value, unit)
    if kind == "molar_energy":
        return q.to("kJ/mol").magnitude * KJ_MOL_TO_EV
    if kind == "temperature":
        # Offset units (degC) need Quantity conversion, not multiplication.
        return q.to("kelvin").magnitude
    return q.to(_INTERNAL_UNITS[kind]).magnitude


def pressure_bar_to_internal(p_bar: float) -> float:
    return p_bar * BAR_TO_EV_A3
