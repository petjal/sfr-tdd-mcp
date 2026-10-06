import math
import pytest
from sfr_mcp.exceptions import DomainBoundaryError
from sfr_mcp.layer1_sodium_properties import (
    celsius_to_kelvin,
    liquid_sodium_density,
    CELSIUS_TO_KELVIN_OFFSET,
)


class TestCelsiusToKelvinConversion:
    """Universal temperature conversion contract: works for any valid real temperature."""

    @pytest.mark.parametrize(
        "temp_c, expected_k",
        [
            (-100.0, 173.15),
            (0.0, 273.15),
            (50.0, 323.15),  # 50 C is valid in general physics!
            (97.80, 370.95),
            (100.0, 373.15),
            (400.0, 673.15),
            (500, 773.15),  # int coercion
            (883.00, 1156.15),
            (1200.0, 1473.15),
        ],
    )
    def test_nominal_temperature_conversion(self, temp_c, expected_k):
        """Verify accurate mathematical conversion from Celsius to Kelvin across all states."""
        assert celsius_to_kelvin(temp_c) == pytest.approx(expected_k, rel=1e-9)

    def test_celsius_to_kelvin_offset_constant(self):
        """Verify the exact canonical offset constant."""
        assert CELSIUS_TO_KELVIN_OFFSET == 273.15

    @pytest.mark.parametrize(
        "invalid_temp",
        [
            float("nan"),
            float("inf"),
            float("-inf"),
            True,
            False,
            "50",
            None,
            -300.0,  # Below absolute zero
            -273.15,  # Absolute zero boundary
        ],
    )
    def test_reject_unphysical_or_non_numeric(self, invalid_temp):
        """Verify unit conversion rejects sub-absolute zero and non-numeric inputs."""
        with pytest.raises(DomainBoundaryError):
            celsius_to_kelvin(invalid_temp)


class TestLiquidSodiumDensity:
    """
    Thermophysical property test suite for liquid sodium mass density rho(T) [kg/m^3].
    
    Primary Source: ANL/RE-95/2 (Fink & Leibowitz, 1995, Section 1.3.1, Eq. 1, p. 86).
    Hornung Formulation:
        theta = 1 - T_K / 2503.7
        rho = 219.0 + 275.32*theta + 511.58*theta^0.5
    """

    @pytest.mark.parametrize(
        "temp_c, anl_expected_rho",
        [
            (100.0, 925.7),
            (250.0, 890.3),
            (400.0, 854.1),
            (550.0, 817.0),
            (700.0, 778.7),
            (850.0, 738.9),
        ],
    )
    def test_density_anl_benchmark_parity(self, temp_c, anl_expected_rho):
        """Verify liquid sodium density matches ANL/RE-95/2 tabulated benchmarks within 0.5%."""
        computed = liquid_sodium_density(temp_c)
        assert computed == pytest.approx(anl_expected_rho, rel=5e-3)

    def test_density_strictly_monotonically_decreasing(self):
        """Universal physical law: Thermal expansion requires d(rho)/dT < 0 everywhere in liquid phase."""
        temps_c = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 850.0]
        densities = [liquid_sodium_density(t) for t in temps_c]
        
        for i in range(len(densities) - 1):
            assert densities[i] > densities[i + 1], (
                f"Density violation: rho({temps_c[i]}°C)={densities[i]} not > "
                f"rho({temps_c[i+1]}°C)={densities[i+1]}"
            )

    @pytest.mark.parametrize(
        "invalid_state",
        [
            float("nan"),
            float("inf"),
            True,
            False,
            "400",
            None,
            50.0,     # Solid sodium (melting point 97.80°C)
            97.80,    # Freezing boundary
            883.00,   # Boiling boundary at 1 atm
            950.0,    # Vapor state
        ],
    )
    def test_liquid_density_rejects_non_liquid_phases(self, invalid_state):
        """Verify liquid density strictly raises DomainBoundaryError on non-liquid phases."""
        with pytest.raises(DomainBoundaryError):
            liquid_sodium_density(invalid_state)
