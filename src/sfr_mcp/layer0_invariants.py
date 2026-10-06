from sfr_mcp.exceptions import DomainBoundaryError

# Absolute physical constants and phase thresholds
ABSOLUTE_ZERO_C = -273.15
SODIUM_MELTING_POINT_C = 97.80
SODIUM_BOILING_POINT_1ATM_C = 883.00
BOUNDARY_FLOAT_EPSILON = 1e-7

def validate_thermal_power(power_mwth: float) -> float:
    """Validate core thermal power [MWth] against numerical and physical invariants."""
    raise NotImplementedError("validate_thermal_power not implemented yet.")

def validate_mass_flow(flow_kg_s: float) -> float:
    """Validate coolant mass flow rate [kg/s] against numerical and kinematic invariants."""
    raise NotImplementedError("validate_mass_flow not implemented yet.")

def validate_temperature(temp_c: float) -> float:
    """Validate liquid sodium temperature [°C] against single-phase thermodynamic invariants."""
    raise NotImplementedError("validate_temperature not implemented yet.")

def validate_core_inputs(power_mwth: float, flow_kg_s: float, inlet_temp_c: float) -> tuple[float, float, float]:
    """Validate composite core state parameters against Layer 0 physical invariants."""
    raise NotImplementedError("validate_core_inputs not implemented yet.")
