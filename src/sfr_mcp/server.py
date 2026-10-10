"""
Layer 3: FastMCP Protocol Service for Single-Channel Sodium Heat Pipe Physics.

Exposes verified Layer 0, Layer 1, and Layer 2 liquid metal heat pipe physics
via the standardized Model Context Protocol (FastMCP).
"""

from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

from sfr_mcp.layer0_invariants import validate_single_channel_inputs
from sfr_mcp.layer2_heat_pipe_solver import (
    calculate_capillary_margin,
    calculate_channel_temperatures,
    calculate_effective_channel_conductivity,
    calculate_mass_flow,
    calculate_vapor_mach_number,
    calculate_vapor_reynolds_number,
)

mcp = FastMCP("sfr-heatpipe-mcp")


class HeatPipeAnalysisResult(BaseModel):
    """Structured thermophysical analysis deliverables for a single-channel sodium heat pipe."""

    model_config = ConfigDict(strict=True, allow_inf_nan=False, extra="forbid")

    power_w: float = Field(
        ...,
        description="Applied thermal heat load in Watts [50.0 W to 750.0 W]",
        ge=50.0,
        le=750.0,
    )
    temp_c: float = Field(
        ...,
        description="Operating saturation temperature in degrees Celsius [625.0 C to 750.0 C]",
        ge=625.0,
        le=750.0,
    )
    mass_flow_g_s: float = Field(
        ...,
        description="Circulating sodium mass flow rate in grams per second [g/s]",
        gt=0.0,
    )
    evaporator_wall_temp_c: float = Field(
        ...,
        description="Outer evaporator wall heat source interface temperature [C]",
    )
    condenser_wall_temp_c: float = Field(
        ...,
        description="Outer condenser wall heat rejection interface temperature [C]",
    )
    delta_t_c: float = Field(
        ...,
        description="End-to-end outer wall temperature drop [C]",
        gt=0.0,
    )
    effective_thermal_conductivity_w_m_k: float = Field(
        ...,
        description="Equivalent bulk conductivity Q*L_total/(A_outer*delta_T) [W/(m*K)]",
        gt=0.0,
    )
    capillary_margin: float = Field(
        ...,
        description="Capillary safety pumping margin M_cap = Delta P_cap,max / Delta P_tot [-]",
        ge=1.0,
    )
    vapor_reynolds_number: float = Field(..., description="Axial vapor Reynolds number [-]", gt=0.0)
    vapor_mach_number: float = Field(..., description="Axial vapor Mach number [-]", gt=0.0)
    flow_regime: str = Field(
        ...,
        description="Computed: LAMINAR if Re_v < 2300, SUBSONIC_INCOMPRESSIBLE if Ma < 0.2",
    )
    operating_status: str = Field(
        default="NOMINAL_STEADY_STATE",
        description="Operational regime status within the verified continuum corridor",
    )


@mcp.tool()
def calculate_heatpipe_heat_transfer(
    power_w: Annotated[
        float,
        Field(
            ge=50.0,
            le=750.0,
            description="Applied thermal heat load in Watts [50.0 W to 750.0 W]",
        ),
    ],
    temp_c: Annotated[
        float,
        Field(
            ge=625.0,
            le=750.0,
            description="Vapor saturation temperature in degrees Celsius [625.0 C to 750.0 C]",
        ),
    ],
) -> HeatPipeAnalysisResult:
    """
    Calculate high-precision steady-state heat transfer and capillary limits for a single sodium heat pipe.

    Evaluates a frozen horizontal test-article geometry (Do=19.05mm, Lt=2.5m, sintered powder wick)
    delivering heat from a fast microreactor core channel to an sCO2 power conversion interface.

    Parameters:
        power_w: Thermal heat load transferred in Watts [50.0 - 750.0 W].
        temp_c: Vapor saturation temperature in degrees Celsius [625.0 - 750.0 C].

    Returns:
        HeatPipeAnalysisResult: Verified mass flow, wall temperatures, capillary margin, and conductance.
    """
    q, t = validate_single_channel_inputs(power_w, temp_c)

    m_dot = calculate_mass_flow(q, t)
    mass_flow_g_s = m_dot * 1000.0

    t_evap_outer, t_cond_outer, delta_t = calculate_channel_temperatures(q, t)
    k_eff_channel = calculate_effective_channel_conductivity(q, t)
    capillary_margin = calculate_capillary_margin(q, t)
    re_v = calculate_vapor_reynolds_number(q, t)
    ma_v = calculate_vapor_mach_number(q, t)
    laminar = "LAMINAR" if re_v < 2300.0 else "TURBULENT"
    compress = "SUBSONIC" if ma_v < 0.2 else "COMPRESSIBLE"

    return HeatPipeAnalysisResult(
        power_w=q,
        temp_c=t,
        mass_flow_g_s=mass_flow_g_s,
        evaporator_wall_temp_c=t_evap_outer,
        condenser_wall_temp_c=t_cond_outer,
        delta_t_c=delta_t,
        effective_thermal_conductivity_w_m_k=k_eff_channel,
        capillary_margin=capillary_margin,
        vapor_reynolds_number=re_v,
        vapor_mach_number=ma_v,
        flow_regime=f"{laminar}_{compress}",
        operating_status="NOMINAL_STEADY_STATE",
    )


if __name__ == "__main__":
    mcp.run()
