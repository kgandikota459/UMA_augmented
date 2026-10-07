import math

import pytest

from umaexp import units


def test_wavenumber_conversions():
    # 1 eV = 8065.544 cm^-1
    assert 1.0 / units.WAVENUMBER_TO_EV == pytest.approx(8065.544, rel=1e-6)
    # sqrt(eV/(A^2 u)) -> cm^-1 ; known value 521.47
    assert units.EIGVAL_TO_WAVENUMBER == pytest.approx(521.4709, rel=1e-5)


def test_time_pressure_density():
    assert units.TAU_S * 1e15 == pytest.approx(10.1805, rel=1e-4)
    assert units.BAR_TO_EV_A3 == pytest.approx(6.241509e-7, rel=1e-6)
    assert units.U_PER_A3_TO_G_CM3 == pytest.approx(1.66053907, rel=1e-8)
    assert units.KB_EV_K == pytest.approx(8.617333262e-5, rel=1e-9)
    assert units.EV_TO_KJ_MOL == pytest.approx(96.4853321, rel=1e-8)


def test_ingestion_parsing():
    assert units.to_internal(25.0, "degC", "temperature") == pytest.approx(298.15)
    assert units.to_internal(1.0, "kcal/mol", "molar_energy") == pytest.approx(0.0433641, rel=1e-5)
    assert units.to_internal(1.0, "atm", "pressure") == pytest.approx(1.01325)
    import pint

    with pytest.raises(pint.errors.DimensionalityError):
        units.to_internal(1.0, "kelvin", "pressure")


def test_standard_concentration():
    # 1 M = 6.022e-4 molecules per A^3 (1660.5 A^3 per molecule)
    assert 1.0 / units.STANDARD_CONCENTRATION_PER_A3 == pytest.approx(1660.539, rel=1e-5)
    assert math.isclose(units.beta(300.0), 1 / (units.KB_EV_K * 300.0))
