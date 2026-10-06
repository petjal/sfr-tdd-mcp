"""
Layer 1: Pure Liquid Sodium Thermophysical Medium.

This module provides thermophysical property evaluations for pure liquid sodium
used in Sodium Fast Reactor (SFR) coolant systems.

All thermodynamic temperature evaluations in Layer 1 inherit physical boundary
guards from Layer 0 (single-phase liquid envelope between 97.80°C and 883.00°C).
"""

from typing import Any
from sfr_mcp.layer0_invariants import validate_temperature, ABSOLUTE_ZERO_C

# Canonical thermodynamic offset between Celsius and Kelvin scales [K]
CELSIUS_TO_KELVIN_OFFSET: float = abs(ABSOLUTE_ZERO_C)


def celsius_to_kelvin(temp_c: Any) -> float:
    """
    Convert liquid sodium temperature from Celsius [°C] to thermodynamic Kelvin [K].

    Boundary Invariant:
        Temperature is validated via Layer 0 (`validate_temperature`) ensuring
        the medium remains in the single-phase liquid window (97.80°C < T < 883.00°C)
        and rejecting non-finite, non-numeric, and boolean types.

    Parameters:
        temp_c: Temperature in degrees Celsius.

    Returns:
        float: Absolute temperature in Kelvin.
    """
    t_c = validate_temperature(temp_c)
    return t_c + CELSIUS_TO_KELVIN_OFFSET
