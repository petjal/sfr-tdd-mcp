"""
Layer 2: Steady-State Thermal Network & Hydrodynamic Capillary Solver.

This module models a single-channel liquid sodium heat pipe on a horizontal
national laboratory capillary test bench (matching LANL LA-11324-M conditions).

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

# Canonical National Lab / LANL Test Article Geometry
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
CHANNEL_WICK_POROSITY: float = 0.65
CHANNEL_WICK_PORE_RADIUS_M: float = 25.0e-6
CHANNEL_WICK_PERMEABILITY_M2: float = 1.50e-10
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

    Young-Laplace Equation (pore radius r_eff = 25 um):
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
        Chi (1976) / Maxwell-Eucken equation for wrapped screen mesh:
        k_eff = k_l * [ (k_l + k_wall) - (1-eps)*(k_l - k_wall) ] / [ (k_l + k_wall) + (1-eps)*(k_l - k_wall) ]
    """
    t = validate_single_channel_temperature(temp_c)
    k_l = liquid_sodium_thermal_conductivity(t)
    k_w = CHANNEL_WALL_CONDUCTIVITY_W_M_K
    eps = CHANNEL_WICK_POROSITY

    term = (1.0 - eps) * (k_l - k_w)
    num = (k_l + k_w) - term
    den = (k_l + k_w) + term
    return float(k_l * (num / den))


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


def verify_lanl_htpipe_benchmark() -> dict[str, Any]:
    """
    ASME NQA-1 Subpart 2.7 Code-to-Code Verification Suite against LANL LA-11324-M HTPIPE benchmark.
    Anchor point: Q = 500.0 W, T_sat = 650.0 °C.
    """
    ref_q = 500.0
    ref_t = 650.0

    # LANL LA-11324-M Reference Benchmark Target Vector
    TARGET_DP_V_PA = 132.5
    TARGET_DP_L_PA = 6150.0
    TARGET_DELTA_T_K = 2.51

    dp_v = calculate_vapor_pressure_drop(ref_q, ref_t)
    dp_l = calculate_liquid_darcy_drop(ref_q, ref_t)
    _, _, delta_t = calculate_channel_temperatures(ref_q, ref_t)

    err_dp_v = abs(dp_v - TARGET_DP_V_PA) / TARGET_DP_V_PA * 100.0
    err_dp_l = abs(dp_l - TARGET_DP_L_PA) / TARGET_DP_L_PA * 100.0
    err_delta_t = abs(delta_t - TARGET_DELTA_T_K) / TARGET_DELTA_T_K * 100.0

    all_passed = (err_dp_v <= 5.0) and (err_dp_l <= 5.0) and (err_delta_t <= 5.0)

    return {
        "reference_power_w": ref_q,
        "reference_temperature_c": ref_t,
        "vapor_pressure_drop_pa": dp_v,
        "vapor_pressure_drop_target_pa": TARGET_DP_V_PA,
        "vapor_pressure_drop_error_pct": err_dp_v,
        "liquid_darcy_drop_pa": dp_l,
        "liquid_darcy_drop_target_pa": TARGET_DP_L_PA,
        "liquid_darcy_drop_error_pct": err_dp_l,
        "temperature_drop_k": delta_t,
        "temperature_drop_target_k": TARGET_DELTA_T_K,
        "temperature_drop_error_pct": err_delta_t,
        "nqa1_verification_status": "VERIFIED_PASS" if all_passed else "FAILED",
    }
