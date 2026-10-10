import math

import pytest

from sfr_mcp.exceptions import DomainBoundaryError
from sfr_mcp.server import (
    HeatPipeAnalysisResult,
    calculate_heatpipe_heat_transfer,
    mcp,
)


class TestFastMCPService:
    """Test suite for FastMCP protocol service exposing single-channel heat pipe solver."""

    def test_nominal_tool_execution(self):
        result = calculate_heatpipe_heat_transfer(power_w=500.0, temp_c=650.0)

        assert isinstance(result, HeatPipeAnalysisResult)
        assert result.power_w == 500.0
        assert result.temp_c == 650.0
        assert result.mass_flow_g_s == pytest.approx(0.1222, rel=1e-2)
        assert result.evaporator_wall_temp_c > 650.0
        assert result.condenser_wall_temp_c < 650.0
        assert result.delta_t_c == pytest.approx(2.5445, rel=1e-3)
        # Source: independent hand calc (Opus audit 2026-10-06)
        assert result.vapor_reynolds_number == pytest.approx(501.8, rel=2e-3)
        assert math.isclose(
            result.evaporator_wall_temp_c - result.condenser_wall_temp_c,
            result.delta_t_c,
            rel_tol=1e-7,
        )
        assert result.effective_thermal_conductivity_w_m_k > 1.0e6
        assert result.capillary_margin == pytest.approx(2.2452, rel=1e-3)
        assert result.flow_regime == "LAMINAR_SUBSONIC"
        assert result.operating_status == "NOMINAL_STEADY_STATE"

    @pytest.mark.parametrize(
        "q_w, t_c",
        [
            (50.0, 625.0),   # Corner A
            (50.0, 750.0),   # Corner B
            (750.0, 625.0),  # Corner C
            (750.0, 750.0),  # Corner D
        ],
    )
    def test_four_corner_matrix_tool_execution(self, q_w, t_c):
        res = calculate_heatpipe_heat_transfer(power_w=q_w, temp_c=t_c)
        assert isinstance(res, HeatPipeAnalysisResult)
        assert res.capillary_margin >= 1.20
        assert res.flow_regime == "LAMINAR_SUBSONIC"
        assert res.operating_status == "NOMINAL_STEADY_STATE"

    @pytest.mark.parametrize(
        "bad_q, bad_t",
        [
            (49.99, 650.0),
            (750.01, 650.0),
            (500.0, 624.99),
            (500.0, 750.01),
            (float("nan"), 650.0),
            (500.0, float("inf")),
            (True, 650.0),
            (500.0, False),
        ],
    )
    def test_tool_rejects_boundary_breaches(self, bad_q, bad_t):
        with pytest.raises((DomainBoundaryError, ValueError, TypeError)):
            calculate_heatpipe_heat_transfer(power_w=bad_q, temp_c=bad_t)

    def test_mcp_server_metadata(self):
        assert mcp.name == "sfr-heatpipe-mcp"

    def test_fastmcp_tool_schema_and_boundaries(self):
        """Verify FastMCP generates correct JSON Schema bounds for LLM tool selection."""
        import asyncio
        tools = asyncio.run(mcp.list_tools())
        tool = next((t for t in tools if t.name == "calculate_heatpipe_heat_transfer"), None)
        assert tool is not None
        assert tool.inputSchema is not None
        props = tool.inputSchema["properties"]
        assert "power_w" in props
        assert "temp_c" in props
        assert props["power_w"].get("minimum") == 50.0
        assert props["power_w"].get("maximum") == 750.0
        assert props["temp_c"].get("minimum") == 625.0
        assert props["temp_c"].get("maximum") == 750.0
        assert "50.0 W to 750.0 W" in props["power_w"].get("description", "")
        assert "625.0 C to 750.0 C" in props["temp_c"].get("description", "")
