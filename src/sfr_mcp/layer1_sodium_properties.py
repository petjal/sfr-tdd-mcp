"""
Layer 1: Pure Liquid Sodium Thermophysical Medium.

This module provides thermophysical property evaluations for pure liquid sodium
used in Sodium Fast Reactor (SFR) coolant systems.

All thermodynamic evaluations in Layer 1 inherit physical boundary
guards from Layer 0.
"""

import math
from typing import Any
from sfr_mcp.layer0_invariants import (
    validate_real_temperature,
    validate_liquid_sodium_temperature,
    ABSOLUTE_ZERO_C,
)

# Canonical thermodynamic offset between Celsius and Kelvin scales [K]
CELSIUS_TO_KELVIN_OFFSET: float = abs(ABSOLUTE_ZERO_C)

# ANL/RE-95/2 (Fink & Leibowitz, 1995, Section 1.3.1, p. 86) Hornung Liquid Mass Density Formulation Constants
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


def liquid_sodium_mass_density(temp_c: Any) -> float:
    """
    Evaluate saturated liquid sodium mass density rho(T) in [kg/m^3] via the Hornung formulation.

    Distinction:
        Specifically evaluates *mass density* [kg/m^3], as distinguished from atomic/number
        density [atoms/b-cm] or core power density [MW/m^3].

    Primary Source:
        Argonne National Laboratory report ANL/RE-95/2 (Fink & Leibowitz, 1995),
        Section 1.3.1 "Density", Equation 1, p. 86.

    Equation:
        theta = 1 - T_K / 2503.7
        rho = 219.0 + 275.32 * theta + 511.58 * sqrt(theta)  [kg/m^3]

    Physical Invariants:
        1. Single-phase liquid: strictly enforces 97.80°C < temp_c < 883.00°C.
        2. Thermal expansion: mass density monotonically decreases with temperature (d(rho)/dT < 0).

    Parameters:
        temp_c: Coolant temperature in degrees Celsius [°C].

    Returns:
        float: Liquid sodium mass density in kilograms per cubic meter [kg/m^3].
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


# Backward-compatible alias for liquid sodium mass density
liquid_sodium_density = liquid_sodium_mass_density


def liquid_sodium_specific_heat(temp_c: Any) -> float:
    """
    Evaluate saturated liquid sodium specific heat capacity C_p(T) in [J/(kg*K)].

    Primary Source:
        ANL/RE-95/2 (Fink & Leibowitz, 1995), Section 1.1.1, p. 8.
    Equation:
        C_p = 1658.2 - 0.84790*T_K + 4.4541e-4*(T_K^2) - 2.9926e6*(T_K^-2)  [J/(kg*K)]
    """
    t_c = validate_liquid_sodium_temperature(temp_c)
    t_k = celsius_to_kelvin(t_c)
    cp = 1658.2 - 0.84790 * t_k + 4.4541e-4 * (t_k ** 2) - 2.9926e6 * (t_k ** -2)
    return float(cp)


def liquid_sodium_surface_tension(temp_c: Any) -> float:
    """
    Evaluate liquid sodium surface tension sigma(T) in [N/m].

    Primary Source:
        ANL/RE-95/2 (Fink & Leibowitz, 1995), Section 1.3.4, p. 109.
    Equation:
        theta = 1 - T_K / 2503.7
        sigma = 0.2405 * (theta ** 1.126)  [N/m]
    """
    t_c = validate_liquid_sodium_temperature(temp_c)
    t_k = celsius_to_kelvin(t_c)
    theta = 1.0 - (t_k / ANL_CRITICAL_TEMP_K)
    sigma = 0.2405 * (theta ** 1.126)
    return float(sigma)


def liquid_sodium_dynamic_viscosity(temp_c: Any) -> float:
    """
    Evaluate saturated liquid sodium dynamic viscosity mu_l(T) in [Pa*s].

    Primary Source:
        ANL/RE-95/2 (Fink & Leibowitz, 1995), Section 1.3.2, p. 94.
    Equation:
        ln(mu_l) = -6.4406 - 0.3958*ln(T_K) + 556.835/T_K  [Pa*s]
    """
    t_c = validate_liquid_sodium_temperature(temp_c)
    t_k = celsius_to_kelvin(t_c)
    ln_mu = -6.4406 - 0.3958 * math.log(t_k) + 556.835 / t_k
    return float(math.exp(ln_mu))


def liquid_sodium_thermal_conductivity(temp_c: Any) -> float:
    """
    Evaluate liquid sodium thermal conductivity k_l(T) in [W/(m*K)].

    Primary Source:
        ANL/RE-95/2 (Fink & Leibowitz, 1995), Section 1.2.1, p. 43.
    Equation:
        k_l = 124.67 - 0.11381*T_K + 5.5226e-5*(T_K^2) - 1.1842e-8*(T_K^3)  [W/(m*K)]
    """
    t_c = validate_liquid_sodium_temperature(temp_c)
    t_k = celsius_to_kelvin(t_c)
    kl = 124.67 - 0.11381 * t_k + 5.5226e-5 * (t_k ** 2) - 1.1842e-8 * (t_k ** 3)
    return float(kl)

