"""
Layer 1: Pure Liquid Sodium Thermophysical Medium.

This module provides thermophysical property evaluations for pure liquid sodium
used in Sodium Fast Reactor (SFR) coolant systems.

All thermodynamic evaluations in Layer 1 inherit physical boundary
guards from Layer 0.
"""

from typing import Any
from sfr_mcp.layer0_invariants import (
    validate_real_temperature,
    validate_liquid_sodium_temperature,
    ABSOLUTE_ZERO_C,
)

# Canonical thermodynamic offset between Celsius and Kelvin scales [K]
CELSIUS_TO_KELVIN_OFFSET: float = abs(ABSOLUTE_ZERO_C)

# ANL/RE-95/2 (Fink & Leibowitz, 1995, Section 1.3.1, p. 86) Hornung Liquid Density Formulation Constants
ANL_CRITICAL_TEMP_K: float = 2503.7
ANL_CRITICAL_DENSITY_KG_M3: float = 219.0
ANL_F_COEFF: float = 275.32
ANL_G_COEFF: float = 511.58
ANL_H_EXPONENT: float = 0.5


def celsius_to_kelvin(temp_c: Any) -> float:
    """
    Convert universal temperature from Celsius [°C] to thermodynamic Kelvin [K].

    Boundary Invariant:
        Temperature is validated via Layer 0 (`validate_real_temperature`) ensuring
        T > -273.15°C (0 K) and rejecting non-finite, non-numeric, and boolean types.

    Parameters:
        temp_c: Temperature in degrees Celsius.

    Returns:
        float: Absolute temperature in Kelvin.
    """
    t_c = validate_real_temperature(temp_c)
    return t_c + CELSIUS_TO_KELVIN_OFFSET


def liquid_sodium_density(temp_c: Any) -> float:
    """
    Evaluate saturated liquid sodium mass density rho(T) in [kg/m^3] via the Hornung formulation.

    Primary Source:
        Argonne National Laboratory report ANL/RE-95/2 (Fink & Leibowitz, 1995),
        Section 1.3.1 "Density", Equation 1, p. 86.

    Equation:
        theta = 1 - T_K / 2503.7
        rho = 219.0 + 275.32 * theta + 511.58 * sqrt(theta)  [kg/m^3]

    Physical Invariants:
        1. Single-phase liquid: strictly enforces 97.80°C < temp_c < 883.00°C.
        2. Thermal expansion: density monotonically decreases with temperature (d(rho)/dT < 0).

    Parameters:
        temp_c: Coolant temperature in degrees Celsius [°C].

    Returns:
        float: Liquid sodium density in kilograms per cubic meter [kg/m^3].
    """
    t_c = validate_liquid_sodium_temperature(temp_c)
    t_k = celsius_to_kelvin(t_c)
    
    theta = 1.0 - (t_k / ANL_CRITICAL_TEMP_K)
    rho = (
        ANL_CRITICAL_DENSITY_KG_M3
        + ANL_F_COEFF * theta
        + ANL_G_COEFF * (theta ** ANL_H_EXPONENT)
    )
    return float(rho)
