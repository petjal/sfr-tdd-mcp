"""
Layer 2: Steady-State Thermal Network & Hydrodynamic Capillary Solver.

This module models a single-channel liquid sodium heat pipe on a horizontal
laboratory capillary test-bench geometry (horizontal, steady state).

All calculations inherit Layer 0 numerical and corridor guards and Layer 1
constitutive thermophysical properties.
"""

import math
from typing import Any
from sfr_mcp.layer0_invariants import (
    validate_single_channel_inputs,
    validate_single_channel_temperature,
)
from sfr_mcp.layer1_sodium_properties import (
    liquid_sodium_mass_density,
    liquid_sodium_surface_tension,
    liquid_sodium_dynamic_viscosity,
    liquid_sodium_thermal_conductivity,
    sodium_enthalpy_of_vaporization,
    sodium_saturated_vapor_density,
    sodium_vapor_dynamic_viscosity,
    celsius_to_kelvin,
)

# Frozen test-article geometry (representative of liquid-metal heat pipe test articles;
# not a specific published LANL article)
CHANNEL_OUTER_DIAMETER_M: float = 19.05e-3
CHANNEL_WALL_THICKNESS_M: float = 1.00e-3
CHANNEL_INNER_DIAMETER_M: float = 17.05e-3
CHANNEL_WICK_THICKNESS_M: float = 1.00e-3
CHANNEL_VAPOR_DIAMETER_M: float = 15.05e-3
CHANNEL_EVAPORATOR_LENGTH_M: float = 1.00
CHANNEL_ADIABATIC_LENGTH_M: float = 0.50
CHANNEL_CONDENSER_LENGTH_M: float = 1.00
CHANNEL_TOTAL_LENGTH_M: float = 2.50
CHANNEL_EFFECTIVE_LENGTH_M: float = (
    0.5 * CHANNEL_EVAPORATOR_LENGTH_M
    + CHANNEL_ADIABATIC_LENGTH_M
    + 0.5 * CHANNEL_CONDENSER_LENGTH_M
)
# Wick: sintered stainless powder, single coherent specification.
#   r_eff = 0.21 * d_p                          (Chi 1976, packed/sintered spheres)
#   K     = d_p^2 eps^3 / (150 (1 - eps)^2)     (Blake-Kozeny / Kozeny-Carman)
CHANNEL_WICK_POROSITY: float = 0.65
CHANNEL_WICK_PARTICLE_DIAMETER_M: float = 100.0e-6
CHANNEL_WICK_PORE_RADIUS_M: float = 0.21 * CHANNEL_WICK_PARTICLE_DIAMETER_M
CHANNEL_WICK_PERMEABILITY_M2: float = (
    CHANNEL_WICK_PARTICLE_DIAMETER_M ** 2 * CHANNEL_WICK_POROSITY ** 3
    / (150.0 * (1.0 - CHANNEL_WICK_POROSITY) ** 2)
)
CHANNEL_WALL_CONDUCTIVITY_W_M_K: float = 21.5

# Pre-computed Cross-Sectional Areas [m^2]
WICK_CROSS_SECTIONAL_AREA_M2: float = (math.pi / 4.0) * (
    CHANNEL_INNER_DIAMETER_M ** 2 - CHANNEL_VAPOR_DIAMETER_M ** 2
)
VAPOR_CROSS_SECTIONAL_AREA_M2: float = (math.pi / 4.0) * (
    CHANNEL_VAPOR_DIAMETER_M ** 2
)


def calculate_mass_flow(power_w: Any, temp_c: Any) -> float:
    """
    Calculate circulating sodium mass flow rate m_dot in [kg/s].

    Formula:
        m_dot = Q / h_fg
    """
    q, t = validate_single_channel_inputs(power_w, temp_c)
    h_fg = sodium_enthalpy_of_vaporization(t)
    return q / h_fg


def calculate_capillary_head_max(temp_c: Any) -> float:
    """
    Calculate maximum capillary pumping head Delta P_cap,max in [Pa].

    Young-Laplace Equation (pore radius r_eff = 0.21 * d_p = 21 um):
        Delta P_cap,max = 2 * sigma / r_eff
    """
    t = validate_single_channel_temperature(temp_c)
    sigma = liquid_sodium_surface_tension(t)
    return (2.0 * sigma) / CHANNEL_WICK_PORE_RADIUS_M


def calculate_liquid_darcy_drop(power_w: Any, temp_c: Any) -> float:
    """
    Calculate viscous pressure drop of liquid sodium through the porous wick Delta P_l in [Pa].

    Darcy's Law for Porous Annulus:
        Delta P_l = (mu_l * m_dot * L_eff) / (rho_l * K * A_w)
    """
    q, t = validate_single_channel_inputs(power_w, temp_c)
    m_dot = calculate_mass_flow(q, t)
    rho_l = liquid_sodium_mass_density(t)
    mu_l = liquid_sodium_dynamic_viscosity(t)

    dp_l = (mu_l * m_dot * CHANNEL_EFFECTIVE_LENGTH_M) / (
        rho_l * CHANNEL_WICK_PERMEABILITY_M2 * WICK_CROSS_SECTIONAL_AREA_M2
    )
    return float(dp_l)


def calculate_vapor_pressure_drop(power_w: Any, temp_c: Any) -> float:
    """
    Calculate frictional pressure drop of laminar sodium vapor in open core Delta P_v in [Pa].

    Hagen-Poiseuille Incompressible Laminar Flow:
        Delta P_v = (128 * mu_v * m_dot * L_eff) / (pi * rho_v * D_v^4)
    """
    q, t = validate_single_channel_inputs(power_w, temp_c)
    m_dot = calculate_mass_flow(q, t)
    rho_v = sodium_saturated_vapor_density(t)
    mu_v = sodium_vapor_dynamic_viscosity(t)

    dp_v = (128.0 * mu_v * m_dot * CHANNEL_EFFECTIVE_LENGTH_M) / (
        math.pi * rho_v * (CHANNEL_VAPOR_DIAMETER_M ** 4)
    )
    return float(dp_v)


def calculate_total_pressure_drop(power_w: Any, temp_c: Any) -> float:
    """
    Calculate total hydrodynamic pressure drop Delta P_tot = Delta P_l + Delta P_v in [Pa].
    """
    dp_l = calculate_liquid_darcy_drop(power_w, temp_c)
    dp_v = calculate_vapor_pressure_drop(power_w, temp_c)
    return dp_l + dp_v


def calculate_capillary_margin(power_w: Any, temp_c: Any) -> float:
    """
    Calculate capillary pumping safety margin M_cap = Delta P_cap,max / Delta P_tot [-].

    Invariant:
        M_cap > 1.0 everywhere in the corridor (guarantees zero wick dryout).
    """
    dp_cap_max = calculate_capillary_head_max(temp_c)
    dp_tot = calculate_total_pressure_drop(power_w, temp_c)
    return float(dp_cap_max / dp_tot)


def calculate_vapor_reynolds_number(power_w: Any, temp_c: Any) -> float:
    """
    Calculate axial vapor Reynolds number Re_v [-].

    Formula:
        Re_v = 4 * m_dot / (pi * D_v * mu_v)

    Invariant:
        Re_v < 1000 << 2300 across the corridor (proves strictly laminar flow).
    """
    q, t = validate_single_channel_inputs(power_w, temp_c)
    m_dot = calculate_mass_flow(q, t)
    mu_v = sodium_vapor_dynamic_viscosity(t)

    re_v = (4.0 * m_dot) / (math.pi * CHANNEL_VAPOR_DIAMETER_M * mu_v)
    return float(re_v)


def calculate_effective_wick_conductivity(temp_c: Any) -> float:
    """
    Calculate effective thermal conductivity of the liquid-sodium-saturated porous wick k_eff in [W/(m*K)].

    Primary Source:
        Chi (1976), sintered wick (Maxwell form), solid = stainless powder k_s:
        k_eff = k_s * [2 + k_l/k_s - 2 eps (1 - k_l/k_s)] / [2 + k_l/k_s + eps (1 - k_l/k_s)]
    """
    t = validate_single_channel_temperature(temp_c)
    k_l = liquid_sodium_thermal_conductivity(t)
    k_s = CHANNEL_WALL_CONDUCTIVITY_W_M_K
    eps = CHANNEL_WICK_POROSITY
    r = k_l / k_s
    return float(k_s * (2.0 + r - 2.0 * eps * (1.0 - r)) / (2.0 + r + eps * (1.0 - r)))


def calculate_wall_resistance(length_m: float) -> float:
    """Radial conduction resistance of the stainless steel tube wall in [K/W]."""
    return float(
        math.log(CHANNEL_OUTER_DIAMETER_M / CHANNEL_INNER_DIAMETER_M)
        / (2.0 * math.pi * CHANNEL_WALL_CONDUCTIVITY_W_M_K * length_m)
    )


def calculate_wick_resistance(length_m: float, temp_c: Any) -> float:
    """Radial conduction resistance of the saturated porous wick in [K/W]."""
    k_eff = calculate_effective_wick_conductivity(temp_c)
    return float(
        math.log(CHANNEL_INNER_DIAMETER_M / CHANNEL_VAPOR_DIAMETER_M)
        / (2.0 * math.pi * k_eff * length_m)
    )


def calculate_vapor_resistance(power_w: Any, temp_c: Any) -> float:
    """
    Axial vapor thermal resistance R_vapor in [K/W] derived from Clausius-Clapeyron relation.

    Equation:
        R_vapor = (T_sat_K * Delta P_v) / (rho_v * h_fg * Q)
    """
    q, t = validate_single_channel_inputs(power_w, temp_c)
    t_k = celsius_to_kelvin(t)
    dp_v = calculate_vapor_pressure_drop(q, t)
    rho_v = sodium_saturated_vapor_density(t)
    h_fg = sodium_enthalpy_of_vaporization(t)

    return float((t_k * dp_v) / (rho_v * h_fg * q))


def calculate_thermal_resistance_network(power_w: Any, temp_c: Any) -> dict[str, float]:
    """
    Calculate 5-element thermal resistance network for the heat pipe channel in [K/W].
    """
    q, t = validate_single_channel_inputs(power_w, temp_c)
    r_wall_e = calculate_wall_resistance(CHANNEL_EVAPORATOR_LENGTH_M)
    r_wick_e = calculate_wick_resistance(CHANNEL_EVAPORATOR_LENGTH_M, t)
    r_vapor = calculate_vapor_resistance(q, t)
    r_wick_c = calculate_wick_resistance(CHANNEL_CONDENSER_LENGTH_M, t)
    r_wall_c = calculate_wall_resistance(CHANNEL_CONDENSER_LENGTH_M)

    r_total = r_wall_e + r_wick_e + r_vapor + r_wick_c + r_wall_c
    return {
        "r_wall_e": r_wall_e,
        "r_wick_e": r_wick_e,
        "r_vapor": r_vapor,
        "r_wick_c": r_wick_c,
        "r_wall_c": r_wall_c,
        "r_total": r_total,
    }


def calculate_channel_temperatures(power_w: Any, temp_c: Any) -> tuple[float, float, float]:
    """
    Calculate heat pipe outer wall temperatures and end-to-end temperature drop.

    Returns:
        tuple[float, float, float]: (T_evap_outer, T_cond_outer, delta_t) in [°C]
    """
    q, t = validate_single_channel_inputs(power_w, temp_c)
    res = calculate_thermal_resistance_network(q, t)

    t_evap_outer = t + q * (res["r_wall_e"] + res["r_wick_e"])
    t_cond_outer = t - q * (res["r_vapor"] + res["r_wick_c"] + res["r_wall_c"])
    delta_t = q * res["r_total"]
    return float(t_evap_outer), float(t_cond_outer), float(delta_t)


def calculate_effective_channel_conductivity(power_w: Any, temp_c: Any) -> float:
    """
    Calculate equivalent solid metal bulk thermal conductivity of the overall heat pipe in [W/(m*K)].
    """
    q, t = validate_single_channel_inputs(power_w, temp_c)
    _, _, delta_t = calculate_channel_temperatures(q, t)
    cross_sectional_area = (math.pi / 4.0) * (CHANNEL_OUTER_DIAMETER_M ** 2)
    return float((q * CHANNEL_TOTAL_LENGTH_M) / (cross_sectional_area * delta_t))


def calculate_vapor_mach_number(power_w: Any, temp_c: Any) -> float:
    """
    Axial vapor Mach number at the evaporator exit [-].

    Ma = v / c,  v = m_dot / (rho_v A_v),  c = sqrt(gamma R T / M), gamma = 5/3,
    M = monomer molar mass (conservative: dimers lower c slightly).
    """
    q, t = validate_single_channel_inputs(power_w, temp_c)
    t_k = celsius_to_kelvin(t)
    m_dot = calculate_mass_flow(q, t)
    v = m_dot / (sodium_saturated_vapor_density(t) * VAPOR_CROSS_SECTIONAL_AREA_M2)
    c = math.sqrt((5.0 / 3.0) * 8.314462 * t_k / 22.98977e-3)
    return float(v / c)


def regression_anchor_500w_650c() -> dict[str, Any]:
    """
    Regression anchor at Q = 500 W, T_sat = 650 C.

    Returns the code's computed values for comparison against an independent
    hand calculation in the test suite. This is NOT a LANL/HTPIPE code-to-code
    benchmark and carries no NQA-1 qualification claim.
    """
    ref_q, ref_t = 500.0, 650.0
    _, _, delta_t = calculate_channel_temperatures(ref_q, ref_t)
    return {
        "reference_power_w": ref_q,
        "reference_temperature_c": ref_t,
        "vapor_pressure_drop_pa": calculate_vapor_pressure_drop(ref_q, ref_t),
        "liquid_darcy_drop_pa": calculate_liquid_darcy_drop(ref_q, ref_t),
        "temperature_drop_k": delta_t,
    }
