import math
import pytest
from sfr_mcp.exceptions import DomainBoundaryError
from sfr_mcp.layer0_invariants import (
    validate_thermal_power,
    validate_mass_flow,
    validate_temperature,
    validate_real_temperature,
    validate_liquid_sodium_temperature,
    validate_core_inputs,
)

class TestTypeAndFinitenessGuards:
    """Numerical robustness tests: assert NaN, Inf, and non-numeric types are strictly trapped."""

    @pytest.mark.parametrize("bad_val", [float("nan")])
    def test_reject_nan_on_all_validators(self, bad_val):
        with pytest.raises(DomainBoundaryError, match="must be a finite real number"):
            validate_thermal_power(bad_val)
        with pytest.raises(DomainBoundaryError, match="must be a finite real number"):
            validate_mass_flow(bad_val)
        with pytest.raises(DomainBoundaryError, match="must be a finite real number"):
            validate_temperature(bad_val)
        with pytest.raises(DomainBoundaryError, match="must be a finite real number"):
            validate_real_temperature(bad_val)
        with pytest.raises(DomainBoundaryError, match="must be a finite real number"):
            validate_liquid_sodium_temperature(bad_val)
        with pytest.raises(DomainBoundaryError, match="must be a finite real number"):
            validate_core_inputs(bad_val, 10.0, 400.0)

    @pytest.mark.parametrize("bad_val", [float("inf"), float("-inf")])
    def test_reject_infinity_on_all_validators(self, bad_val):
        with pytest.raises(DomainBoundaryError, match="must be a finite real number"):
            validate_thermal_power(bad_val)
        with pytest.raises(DomainBoundaryError, match="must be a finite real number"):
            validate_mass_flow(bad_val)
        with pytest.raises(DomainBoundaryError, match="must be a finite real number"):
            validate_temperature(bad_val)
        with pytest.raises(DomainBoundaryError, match="must be a finite real number"):
            validate_real_temperature(bad_val)
        with pytest.raises(DomainBoundaryError, match="must be a finite real number"):
            validate_liquid_sodium_temperature(bad_val)

    @pytest.mark.parametrize("bad_type", [True, False])
    def test_reject_boolean_types(self, bad_type):
        """Python booleans subclass int; must be explicitly rejected."""
        with pytest.raises(DomainBoundaryError, match="Booleans are not valid numeric inputs"):
            validate_thermal_power(bad_type)
        with pytest.raises(DomainBoundaryError, match="Booleans are not valid numeric inputs"):
            validate_mass_flow(bad_type)
        with pytest.raises(DomainBoundaryError, match="Booleans are not valid numeric inputs"):
            validate_temperature(bad_type)
        with pytest.raises(DomainBoundaryError, match="Booleans are not valid numeric inputs"):
            validate_real_temperature(bad_type)
        with pytest.raises(DomainBoundaryError, match="Booleans are not valid numeric inputs"):
            validate_liquid_sodium_temperature(bad_type)

    @pytest.mark.parametrize("bad_type", [None, "500", [10.0], 2 + 3j])
    def test_reject_non_numeric_types(self, bad_type):
        with pytest.raises(DomainBoundaryError, match="Expected a real numeric"):
            validate_thermal_power(bad_type)
        with pytest.raises(DomainBoundaryError, match="Expected a real numeric"):
            validate_mass_flow(bad_type)
        with pytest.raises(DomainBoundaryError, match="Expected a real numeric"):
            validate_temperature(bad_type)
        with pytest.raises(DomainBoundaryError, match="Expected a real numeric"):
            validate_real_temperature(bad_type)
        with pytest.raises(DomainBoundaryError, match="Expected a real numeric"):
            validate_liquid_sodium_temperature(bad_type)


class TestPhysicalPowerGuards:
    """Thermal power must be strictly positive (> 0 MWth) for reactor core heating."""

    @pytest.mark.parametrize("invalid_p", [0.0, -1e-9, -5.0, -100.0])
    def test_reject_zero_and_negative_power(self, invalid_p):
        with pytest.raises(DomainBoundaryError, match="Thermal power must be strictly positive"):
            validate_thermal_power(invalid_p)

    @pytest.mark.parametrize("valid_p", [1e-4, 1.0, 4.0, 100.0])
    def test_accept_valid_power(self, valid_p):
        assert validate_thermal_power(valid_p) == float(valid_p)


class TestPhysicalMassFlowGuards:
    """Mass flow rate must be strictly positive (> 0 kg/s) to prevent starvation / zero-division."""

    @pytest.mark.parametrize("invalid_flow", [0.0, -1e-9, -12.5, -50.0])
    def test_reject_zero_and_negative_flow(self, invalid_flow):
        with pytest.raises(DomainBoundaryError, match="Mass flow rate must be strictly positive"):
            validate_mass_flow(invalid_flow)

    @pytest.mark.parametrize("valid_flow", [1e-3, 5.0, 15.0, 500.0])
    def test_accept_valid_flow(self, valid_flow):
        assert validate_mass_flow(valid_flow) == float(valid_flow)


class TestUniversalRealTemperatureGuards:
    """Universal physical temperature guard: only enforces T > absolute zero."""

    @pytest.mark.parametrize("cryo_t", [-300.0, -273.16, -273.15])
    def test_reject_sub_absolute_zero(self, cryo_t):
        with pytest.raises(DomainBoundaryError, match="below absolute zero"):
            validate_real_temperature(cryo_t)

    @pytest.mark.parametrize("valid_t", [-273.14, -100.0, 0.0, 50.0, 97.80, 400.0, 883.00, 1500.0])
    def test_accept_valid_real_temperatures(self, valid_t):
        assert validate_real_temperature(valid_t) == float(valid_t)


class TestSodiumPhaseBoundaryGuards:
    """Coolant temperature must strictly reside in single-phase liquid sodium envelope."""

    @pytest.mark.parametrize("cryo_t", [-300.0, -273.16, -273.15])
    def test_reject_sub_absolute_zero(self, cryo_t):
        with pytest.raises(DomainBoundaryError, match="below absolute zero"):
            validate_liquid_sodium_temperature(cryo_t)

    @pytest.mark.parametrize("frozen_t", [-200.0, 0.0, 50.0, 97.80])
    def test_reject_solid_freezing_boundary(self, frozen_t):
        with pytest.raises(DomainBoundaryError, match="violates sodium freezing threshold"):
            validate_liquid_sodium_temperature(frozen_t)

    @pytest.mark.parametrize("boiling_t", [883.00, 883.01, 1000.0])
    def test_reject_atmospheric_boiling_boundary(self, boiling_t):
        with pytest.raises(DomainBoundaryError, match="violates sodium boiling threshold"):
            validate_liquid_sodium_temperature(boiling_t)

    @pytest.mark.parametrize("valid_t", [97.80001, 150.0, 360.0, 550.0, 882.99999])
    def test_accept_valid_liquid_temperature_window(self, valid_t):
        assert math.isclose(validate_liquid_sodium_temperature(valid_t), valid_t, rel_tol=1e-9)
        # Verify backward-compatible alias validate_temperature
        assert math.isclose(validate_temperature(valid_t), valid_t, rel_tol=1e-9)


class TestCompositeCoreInputValidation:
    """Composite validation pipeline verifying all parameters simultaneously."""

    def test_composite_accepts_clean_operating_point(self):
        p, m, t = validate_core_inputs(power_mwth=4.0, flow_kg_s=15.0, inlet_temp_c=360.0)
        assert p == 4.0
        assert m == 15.0
        assert t == 360.0

    def test_composite_rejects_any_single_invalid_parameter(self):
        # Bad power
        with pytest.raises(DomainBoundaryError, match="Thermal power must be strictly positive"):
            validate_core_inputs(0.0, 15.0, 360.0)
        # Bad flow
        with pytest.raises(DomainBoundaryError, match="Mass flow rate must be strictly positive"):
            validate_core_inputs(4.0, -1.0, 360.0)
        # Bad temp
        with pytest.raises(DomainBoundaryError, match="violates sodium freezing threshold"):
            validate_core_inputs(4.0, 15.0, 50.0)
