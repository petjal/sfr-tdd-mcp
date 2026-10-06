import math
import pytest
from sfr_mcp.exceptions import DomainBoundaryError
from sfr_mcp.layer1_sodium_properties import (
    celsius_to_kelvin,
    CELSIUS_TO_KELVIN_OFFSET,
)


class TestCelsiusToKelvinConversion:
    """Test suite establishing the tracer bullet contract for Layer 1 temperature conversion."""

    @pytest.mark.parametrize(
        "temp_c, expected_k",
        [
            (100.0, 373.15),
            (360.0, 633.15),
            (400.0, 673.15),
            (500, 773.15),  # integer input coercion
            (550.0, 823.15),
            (882.99999, 882.99999 + 273.15),
        ],
    )
    def test_nominal_temperature_conversion(self, temp_c, expected_k):
        """Verify accurate mathematical conversion from Celsius to Kelvin."""
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
            "400",
            None,
            50.0,  # Below melting point (97.80 C)
            97.80,  # Solid freezing boundary
            883.00,  # Atmospheric boiling boundary
            1000.0,  # Vapor state
        ],
    )
    def test_layer0_boundary_inheritance(self, invalid_temp):
        """Verify Layer 1 temperature conversion strictly inherits Layer 0 DomainBoundaryError guards."""
        with pytest.raises(DomainBoundaryError):
            celsius_to_kelvin(invalid_temp)
