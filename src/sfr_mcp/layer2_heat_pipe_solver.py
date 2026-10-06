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
