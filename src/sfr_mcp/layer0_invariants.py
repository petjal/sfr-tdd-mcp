"""
Layer 0: Universal Domain Invariant Substrate for Fast Reactor Systems.

This module provides physical invariant guards and IEEE 754 numerical defenses
specifically tailored for single-channel sodium heat pipe microreactor modeling
and legacy liquid metal coolant channels.
"""

import math
from typing import Any

from sfr_mcp.exceptions import DomainBoundaryError

# Canonical physical constants and environmental thresholds
ABSOLUTE_ZERO_TEMP_C: float = -273.15
SODIUM_SOLIDUS_MELT_TEMP_C: float = 97.80
SODIUM_ATMOSPHERIC_BOIL_TEMP_C: float = 883.00
HEAT_PIPE_MAX_POWER_KW: float = 150.0
NUMERICAL_BOUNDARY_EPSILON: float = 1e-7

# Single-channel operating corridor (nominal steady-state envelope)
HEAT_PIPE_CORRIDOR_MIN_TEMP_C: float = 625.0
HEAT_PIPE_CORRIDOR_MAX_TEMP_C: float = 750.0
HEAT_PIPE_CORRIDOR_MIN_POWER_W: float = 50.0
HEAT_PIPE_CORRIDOR_MAX_POWER_W: float = 750.0

# Backward-compatibility aliases
ABSOLUTE_ZERO_C: float = ABSOLUTE_ZERO_TEMP_C
SODIUM_MELTING_POINT_C: float = SODIUM_SOLIDUS_MELT_TEMP_C
SODIUM_BOILING_POINT_1ATM_C: float = SODIUM_ATMOSPHERIC_BOIL_TEMP_C
BOUNDARY_FLOAT_EPSILON: float = NUMERICAL_BOUNDARY_EPSILON


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


def validate_real_temperature(temp_c: Any) -> float:
    """
    Validate universal temperature [°C] against the third law of thermodynamics.

    Invariant: Temperature must be strictly above absolute zero (-273.15°C / 0 K).
    """
    t = _validate_numeric(temp_c, "temp_c")
    if t <= ABSOLUTE_ZERO_TEMP_C:
        raise DomainBoundaryError(
            f"Temperature {t:.2f}°C is at or below absolute zero (-273.15°C)."
        )
    return t


def validate_heat_pipe_temperature(temp_c: Any) -> float:
    """
    Validate sodium heat pipe operational temperature [°C] within the single-phase liquid window.

    Physical Regimes:
        - T <= 97.80°C: Solid metal. Sodium is frozen into the wick pores; zero flow.
        - 97.80°C < T < 883.00°C: Operational window where internal pressure is sub-atmospheric
          (0.005 to 0.075 atm at nominal power).
        - T >= 883.00°C: Pressurization inversion cliff. Saturated vapor pressure exceeds 1.0 atm,
          flipping pipe walls from external compression into internal tension (burst risk).
    """
    t = validate_real_temperature(temp_c)

    if t <= SODIUM_SOLIDUS_MELT_TEMP_C + NUMERICAL_BOUNDARY_EPSILON:
        raise DomainBoundaryError(
            f"Temperature {t:.2f}°C violates sodium freezing threshold (97.8°C)."
        )

    if t >= SODIUM_ATMOSPHERIC_BOIL_TEMP_C - NUMERICAL_BOUNDARY_EPSILON:
        raise DomainBoundaryError(
            f"Temperature {t:.2f}°C violates sodium boiling threshold (883.0°C at 1 atm)."
        )

    return t


def validate_heat_pipe_power(power_kw: Any) -> float:
    """
    Validate thermal power [kW] transferred through a single sodium heat pipe.

    Invariants:
        1. Power must be strictly positive (> 0 kW) for active heat removal.
        2. Power must not exceed the single-channel capillary limit ceiling (150 kWth).
    """
    p = _validate_numeric(power_kw, "power_kw")
    if p <= 0.0:
        raise DomainBoundaryError(
            f"Heat pipe thermal power must be strictly positive (> 0 kW). Got: {p}"
        )
    if p > HEAT_PIPE_MAX_POWER_KW:
        raise DomainBoundaryError(
            f"Heat pipe power {p:.2f} kW exceeds maximum single heat pipe limit ({HEAT_PIPE_MAX_POWER_KW} kW)."
        )
    return p


def validate_heat_pipe_channel_inputs(
    power_kw: Any, temp_c: Any
) -> tuple[float, float]:
    """
    Validate coupled operating parameters for a single sodium heat pipe channel.

    Returns:
        tuple[float, float]: (validated_power_kw, validated_temp_c)
    """
    p = validate_heat_pipe_power(power_kw)
    t = validate_heat_pipe_temperature(temp_c)
    return p, t


def validate_single_channel_temperature(temp_c: Any) -> float:
    """
    Validate single-channel saturation temperature [°C] against the national lab corridor.

    Corridor: 625.0°C <= T <= 750.0°C (continuum Navier-Stokes sweet spot).
    """
    t = validate_real_temperature(temp_c)
    if t < HEAT_PIPE_CORRIDOR_MIN_TEMP_C - NUMERICAL_BOUNDARY_EPSILON:
        raise DomainBoundaryError(
            f"Saturation temperature {t:.4f}°C breaches minimum corridor boundary ({HEAT_PIPE_CORRIDOR_MIN_TEMP_C}°C)."
        )
    if t > HEAT_PIPE_CORRIDOR_MAX_TEMP_C + NUMERICAL_BOUNDARY_EPSILON:
        raise DomainBoundaryError(
            f"Saturation temperature {t:.4f}°C breaches maximum corridor boundary ({HEAT_PIPE_CORRIDOR_MAX_TEMP_C}°C)."
        )
    return t


def validate_single_channel_power(power_w: Any) -> float:
    """
    Validate single-channel thermal power [W] against the national lab corridor.

    Corridor: 50.0 W <= Q <= 750.0 W (laminar subsonic sweet spot).
    """
    p = _validate_numeric(power_w, "power_w")
    if p < HEAT_PIPE_CORRIDOR_MIN_POWER_W - NUMERICAL_BOUNDARY_EPSILON:
        raise DomainBoundaryError(
            f"Heat pipe thermal power {p:.4f} W breaches minimum corridor boundary ({HEAT_PIPE_CORRIDOR_MIN_POWER_W} W)."
        )
    if p > HEAT_PIPE_CORRIDOR_MAX_POWER_W + NUMERICAL_BOUNDARY_EPSILON:
        raise DomainBoundaryError(
            f"Heat pipe thermal power {p:.4f} W breaches maximum corridor boundary ({HEAT_PIPE_CORRIDOR_MAX_POWER_W} W)."
        )
    return p


def validate_single_channel_inputs(
    power_w: Any, temp_c: Any
) -> tuple[float, float]:
    """
    Validate coupled operating inputs for the single-channel sodium heat pipe corridor.

    Returns:
        tuple[float, float]: (validated_power_w, validated_temp_c)
    """
    p = validate_single_channel_power(power_w)
    t = validate_single_channel_temperature(temp_c)
    return p, t


# Temperature alias for liquid sodium domain checking
validate_liquid_sodium_temperature = validate_heat_pipe_temperature
validate_temperature = validate_heat_pipe_temperature
