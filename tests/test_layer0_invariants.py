import math

import pytest

from sfr_mcp.exceptions import DomainBoundaryError
from sfr_mcp.layer0_invariants import (
    ABSOLUTE_ZERO_TEMP_C,
    HEAT_PIPE_CORRIDOR_MAX_POWER_W,
    HEAT_PIPE_CORRIDOR_MAX_TEMP_C,
    HEAT_PIPE_CORRIDOR_MIN_POWER_W,
    HEAT_PIPE_CORRIDOR_MIN_TEMP_C,
    HEAT_PIPE_MAX_POWER_KW,
    NUMERICAL_BOUNDARY_EPSILON,
    SODIUM_ATMOSPHERIC_BOIL_TEMP_C,
    SODIUM_SOLIDUS_MELT_TEMP_C,
    validate_heat_pipe_channel_inputs,
    validate_heat_pipe_power,
    validate_heat_pipe_temperature,
    # Temperature aliases
    validate_liquid_sodium_temperature,
    validate_real_temperature,
    validate_single_channel_inputs,
    validate_single_channel_power,
    validate_single_channel_temperature,
    validate_temperature,
)


class TestLayer0Constants:
    """Verify single-sourced thermodynamic constants for heat pipe operational envelope."""

    def test_constants_values(self):
        assert ABSOLUTE_ZERO_TEMP_C == -273.15
        assert SODIUM_SOLIDUS_MELT_TEMP_C == 97.80
        assert SODIUM_ATMOSPHERIC_BOIL_TEMP_C == 883.00
        assert HEAT_PIPE_MAX_POWER_KW == 150.0
        assert NUMERICAL_BOUNDARY_EPSILON == 1e-7


class TestNumericalAndTypeImmuneSystem:
    """Rigorous IEEE 754 and type guards across all heat pipe validators."""

    @pytest.mark.parametrize("bad_val", [float("nan")])
    def test_reject_nan(self, bad_val):
        with pytest.raises(DomainBoundaryError, match="must be a finite real number"):
            validate_real_temperature(bad_val)
        with pytest.raises(DomainBoundaryError, match="must be a finite real number"):
            validate_heat_pipe_temperature(bad_val)
        with pytest.raises(DomainBoundaryError, match="must be a finite real number"):
            validate_heat_pipe_power(bad_val)

    @pytest.mark.parametrize("bad_val", [float("inf"), float("-inf")])
    def test_reject_infinities(self, bad_val):
        with pytest.raises(DomainBoundaryError, match="must be a finite real number"):
            validate_real_temperature(bad_val)
        with pytest.raises(DomainBoundaryError, match="must be a finite real number"):
            validate_heat_pipe_temperature(bad_val)
        with pytest.raises(DomainBoundaryError, match="must be a finite real number"):
            validate_heat_pipe_power(bad_val)

    @pytest.mark.parametrize("bad_type", [True, False])
    def test_reject_boolean_types(self, bad_type):
        with pytest.raises(DomainBoundaryError, match="Booleans are not valid numeric inputs"):
            validate_real_temperature(bad_type)
        with pytest.raises(DomainBoundaryError, match="Booleans are not valid numeric inputs"):
            validate_heat_pipe_temperature(bad_type)
        with pytest.raises(DomainBoundaryError, match="Booleans are not valid numeric inputs"):
            validate_heat_pipe_power(bad_type)

    @pytest.mark.parametrize("bad_type", [None, "50.0", [25.0], 2 + 3j])
    def test_reject_non_numeric_types(self, bad_type):
        with pytest.raises(DomainBoundaryError, match="Expected a real numeric"):
            validate_real_temperature(bad_type)
        with pytest.raises(DomainBoundaryError, match="Expected a real numeric"):
            validate_heat_pipe_temperature(bad_type)
        with pytest.raises(DomainBoundaryError, match="Expected a real numeric"):
            validate_heat_pipe_power(bad_type)


class TestUniversalRealTemperature:
    """Universal temperature guard: enforces 3rd Law of Thermodynamics (T > -273.15 C)."""

    @pytest.mark.parametrize("cryo_t", [-300.0, -273.16, -273.15])
    def test_reject_sub_absolute_zero(self, cryo_t):
        with pytest.raises(DomainBoundaryError, match="below absolute zero"):
            validate_real_temperature(cryo_t)

    @pytest.mark.parametrize("valid_t", [-273.14, -100.0, 0.0, 50.0, 97.80, 400.0, 883.00, 1500.0])
    def test_accept_valid_real_temperatures(self, valid_t):
        assert validate_real_temperature(valid_t) == float(valid_t)


class TestHeatPipeTemperatureEnvelope:
    """Heat pipe operational envelope: single-phase liquid sodium between thaw and 1-atm boiling."""

    @pytest.mark.parametrize("frozen_t", [-200.0, 0.0, 50.0, 97.80])
    def test_reject_frozen_solid_sodium(self, frozen_t):
        with pytest.raises(DomainBoundaryError, match="violates sodium freezing threshold"):
            validate_heat_pipe_temperature(frozen_t)

    @pytest.mark.parametrize("boiling_t", [883.00, 883.01, 1000.0])
    def test_reject_atmospheric_boiling_inversion(self, boiling_t):
        with pytest.raises(DomainBoundaryError, match="violates sodium boiling threshold"):
            validate_heat_pipe_temperature(boiling_t)

    @pytest.mark.parametrize("valid_t", [97.80001, 150.0, 400.0, 550.0, 650.0, 882.99999])
    def test_accept_valid_heat_pipe_temperatures(self, valid_t):
        assert math.isclose(validate_heat_pipe_temperature(valid_t), valid_t, rel_tol=1e-9)


class TestHeatPipePowerEnvelope:
    """Single heat pipe thermal power: must be positive and <= 150 kWth."""

    @pytest.mark.parametrize("invalid_p", [0.0, -1e-9, -5.0, -50.0])
    def test_reject_non_positive_power(self, invalid_p):
        with pytest.raises(DomainBoundaryError, match="strictly positive"):
            validate_heat_pipe_power(invalid_p)

    @pytest.mark.parametrize("excessive_p", [150.0001, 200.0, 500.0, 1000.0])
    def test_reject_excessive_overpower(self, excessive_p):
        with pytest.raises(DomainBoundaryError, match="exceeds maximum single heat pipe limit"):
            validate_heat_pipe_power(excessive_p)

    @pytest.mark.parametrize("valid_p", [0.01, 1.0, 25.0, 50.0, 100.0, 150.0])
    def test_accept_valid_heat_pipe_power(self, valid_p):
        assert validate_heat_pipe_power(valid_p) == float(valid_p)


class TestHeatPipeChannelInputsComposite:
    """Composite validator for a single heat pipe channel."""

    def test_accept_nominal_operating_point(self):
        p, t = validate_heat_pipe_channel_inputs(power_kw=50.0, temp_c=550.0)
        assert p == 50.0
        assert t == 550.0

    def test_reject_invalid_power_in_composite(self):
        with pytest.raises(DomainBoundaryError, match="strictly positive"):
            validate_heat_pipe_channel_inputs(0.0, 550.0)
        with pytest.raises(DomainBoundaryError, match="exceeds maximum single heat pipe limit"):
            validate_heat_pipe_channel_inputs(200.0, 550.0)

    def test_reject_invalid_temperature_in_composite(self):
        with pytest.raises(DomainBoundaryError, match="violates sodium freezing threshold"):
            validate_heat_pipe_channel_inputs(50.0, 50.0)
        with pytest.raises(DomainBoundaryError, match="violates sodium boiling threshold"):
            validate_heat_pipe_channel_inputs(50.0, 900.0)


class TestBackwardCompatibilityAliases:
    """Ensure older pool/core aliases remain fully functional."""

    def test_legacy_aliases(self):
        assert validate_liquid_sodium_temperature(550.0) == 550.0
        assert validate_temperature(550.0) == 550.0


class TestSingleChannelCorridor:
    """Rigorous tests for the canonical single-channel test bench operating corridor [625-750 C, 50-750 W]."""

    def test_corridor_constants(self):
        assert HEAT_PIPE_CORRIDOR_MIN_TEMP_C == 625.0
        assert HEAT_PIPE_CORRIDOR_MAX_TEMP_C == 750.0
        assert HEAT_PIPE_CORRIDOR_MIN_POWER_W == 50.0
        assert HEAT_PIPE_CORRIDOR_MAX_POWER_W == 750.0

    @pytest.mark.parametrize(
        "q_w,t_c",
        [
            (50.0, 625.0),   # Corner A: Min power, min temp
            (50.0, 750.0),   # Corner B: Min power, max temp
            (750.0, 625.0),  # Corner C: Max power, min temp
            (750.0, 750.0),  # Corner D: Max power, max temp
            (500.0, 650.0),  # Nominal benchmark design point
        ],
    )
    def test_accept_four_corner_matrix(self, q_w, t_c):
        p, t = validate_single_channel_inputs(q_w, t_c)
        assert math.isclose(p, q_w, rel_tol=1e-9)
        assert math.isclose(t, t_c, rel_tol=1e-9)

    @pytest.mark.parametrize(
        "bad_q",
        [
            49.99,
            49.9999,
            0.0,
            -50.0,
            750.01,
            750.0001,
            1000.0,
        ],
    )
    def test_reject_power_breach(self, bad_q):
        with pytest.raises(DomainBoundaryError, match="power.*corridor"):
            validate_single_channel_power(bad_q)

    @pytest.mark.parametrize(
        "bad_t",
        [
            624.99,
            624.9999,
            500.0,
            100.0,
            750.01,
            750.0001,
            800.0,
        ],
    )
    def test_reject_temperature_breach(self, bad_t):
        with pytest.raises(DomainBoundaryError, match="temperature.*corridor"):
            validate_single_channel_temperature(bad_t)

    @pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf"), True, False, "500", None])
    def test_reject_non_finites_and_types(self, bad_val):
        with pytest.raises(DomainBoundaryError):
            validate_single_channel_power(bad_val)
        with pytest.raises(DomainBoundaryError):
            validate_single_channel_temperature(bad_val)

