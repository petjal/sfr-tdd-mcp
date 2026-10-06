import math
from typing import Any
from sfr_mcp.exceptions import DomainBoundaryError

# Absolute physical constants and phase thresholds
ABSOLUTE_ZERO_C = -273.15
SODIUM_MELTING_POINT_C = 97.80
SODIUM_BOILING_POINT_1ATM_C = 883.00
BOUNDARY_FLOAT_EPSILON = 1e-7

def _validate_numeric(val: Any, name: str) -> float:
    """Enforce numerical finiteness and reject non-real/boolean data types."""
    if isinstance(val, bool):
        raise DomainBoundaryError(f"Booleans are not valid numeric inputs for {name}. Got {val}.")
        
    if not isinstance(val, (int, float)):
        raise DomainBoundaryError(
            f"Expected a real numeric input for {name}. Got {type(val).__name__}."
        )
        
    val_float = float(val)
    if not math.isfinite(val_float):
        raise DomainBoundaryError(
            f"Input {name} must be a finite real number. Got {val}."
        )
        
    return val_float


def validate_thermal_power(power_mwth: Any) -> float:
    """
    Validate core thermal power [MWth] against numerical and physical invariants.
    
    Invariant: Thermal power must be strictly positive (> 0 MWth) for reactor heating.
    """
    p = _validate_numeric(power_mwth, "power_mwth")
    if p <= 0.0:
        raise DomainBoundaryError(
            f"Thermal power must be strictly positive (> 0 MWth). Got: {p}"
        )
    return p


def validate_mass_flow(flow_kg_s: Any) -> float:
    """
    Validate coolant mass flow rate [kg/s] against numerical and kinematic invariants.
    
    Invariant: Flow rate must be strictly positive (> 0 kg/s) to prevent starvation / zero-division.
    """
    m = _validate_numeric(flow_kg_s, "flow_kg_s")
    if m <= 0.0:
        raise DomainBoundaryError(
            f"Mass flow rate must be strictly positive (> 0 kg/s). Got: {m}"
        )
    return m


def validate_temperature(temp_c: Any) -> float:
    """
    Validate liquid sodium temperature [°C] against single-phase thermodynamic invariants.
    
    Invariants:
    1. Temperature must be strictly above absolute zero (-273.15°C / 0 K).
    2. Temperature must be strictly above the solid melting point (97.80°C).
    3. Temperature must be strictly below the 1-atm boiling point (883.00°C).
    """
    t = _validate_numeric(temp_c, "temp_c")
    
    if t <= ABSOLUTE_ZERO_C:
        raise DomainBoundaryError(
            f"Temperature {t:.2f}°C is at or below absolute zero (-273.15°C)."
        )
        
    if t <= SODIUM_MELTING_POINT_C + BOUNDARY_FLOAT_EPSILON:
        raise DomainBoundaryError(
            f"Temperature {t:.2f}°C violates sodium freezing threshold (97.8°C)."
        )
        
    if t >= SODIUM_BOILING_POINT_1ATM_C - BOUNDARY_FLOAT_EPSILON:
        raise DomainBoundaryError(
            f"Temperature {t:.2f}°C violates sodium boiling threshold (883.0°C at 1 atm)."
        )
        
    return t


def validate_core_inputs(
    power_mwth: Any, flow_kg_s: Any, inlet_temp_c: Any
) -> tuple[float, float, float]:
    """
    Validate composite core state parameters against Layer 0 physical invariants.
    
    Returns tuple of validated floats: (power_mwth, flow_kg_s, inlet_temp_c).
    """
    p = validate_thermal_power(power_mwth)
    m = validate_mass_flow(flow_kg_s)
    t = validate_temperature(inlet_temp_c)
    return p, m, t
