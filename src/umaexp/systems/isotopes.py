"""Isotope masses for spectroscopy.

Vibrational frequencies depend on the masses of the specific isotopologue that
was measured. Standard atomic weights (what `ase.data.atomic_masses` holds)
are abundance averages and must not be used when comparing to spectra of,
e.g., H2(16)O versus D2(16)O.

Values: AME2020 atomic masses (Wang et al., Chinese Phys. C 45, 030003 (2021)),
in daltons. Only isotopes needed by the pilot curation are listed; extend the
table rather than falling back silently.
"""

from __future__ import annotations

from ase.data import atomic_masses, chemical_symbols

# (Z, mass_number) -> mass in u
ISOTOPE_MASS_U: dict[tuple[int, int], float] = {
    (1, 1): 1.00782503223,
    (1, 2): 2.01410177812,
    (1, 3): 3.0160492779,
    (3, 7): 7.0160034366,
    (5, 10): 10.01293695,
    (5, 11): 11.00930536,
    (6, 12): 12.0,
    (6, 13): 13.00335483507,
    (7, 14): 14.00307400443,
    (7, 15): 15.00010889888,
    (8, 16): 15.99491461957,
    (8, 17): 16.99913175650,
    (8, 18): 17.99915961286,
    (9, 19): 18.99840316273,
    (11, 23): 22.9897692820,
    (14, 28): 27.97692653465,
    (15, 31): 30.97376199842,
    (16, 32): 31.9720711744,
    (16, 34): 33.967867004,
    (17, 35): 34.968852682,
    (17, 37): 36.965902602,
    (19, 39): 38.9637064864,
    (35, 79): 78.9183376,
    (35, 81): 80.9162897,
    (53, 127): 126.9044719,
}

# Most abundant isotope per element (mass number), for "main isotopologue"
# spectroscopy defaults.
MOST_ABUNDANT_ISOTOPE: dict[int, int] = {
    1: 1, 3: 7, 5: 11, 6: 12, 7: 14, 8: 16, 9: 19, 11: 23, 14: 28, 15: 31,
    16: 32, 17: 35, 19: 39, 35: 79, 53: 127,
}


class UnknownIsotopeError(KeyError):
    pass


def isotope_mass(z: int, mass_number: int | None = None) -> float:
    """Mass of a specific isotope; `mass_number=None` means most abundant."""
    if mass_number is None:
        if z not in MOST_ABUNDANT_ISOTOPE:
            raise UnknownIsotopeError(
                f"No most-abundant isotope recorded for {chemical_symbols[z]}; "
                "add it to umaexp.systems.isotopes instead of using an average weight."
            )
        mass_number = MOST_ABUNDANT_ISOTOPE[z]
    try:
        return ISOTOPE_MASS_U[(z, mass_number)]
    except KeyError as err:
        raise UnknownIsotopeError(
            f"Isotope {mass_number}{chemical_symbols[z]} is not in the table"
        ) from err


def standard_atomic_weight(z: int) -> float:
    """Abundance-averaged atomic weight (appropriate for classical MD masses)."""
    return float(atomic_masses[z])


def isotopologue_masses(
    atomic_numbers: list[int] | tuple[int, ...],
    overrides: dict[int, int] | None = None,
) -> tuple[float, ...]:
    """Isotope masses for every atom; `overrides` maps atom index -> mass number."""
    overrides = overrides or {}
    return tuple(
        isotope_mass(int(z), overrides.get(i)) for i, z in enumerate(atomic_numbers)
    )
