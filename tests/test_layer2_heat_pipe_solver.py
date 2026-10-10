import math

import pytest

from sfr_mcp.exceptions import DomainBoundaryError
from sfr_mcp.layer2_heat_pipe_solver import (
    CHANNEL_EFFECTIVE_LENGTH_M,
    CHANNEL_INNER_DIAMETER_M,
    CHANNEL_OUTER_DIAMETER_M,
    CHANNEL_VAPOR_DIAMETER_M,
    CHANNEL_WICK_PARTICLE_DIAMETER_M,
    CHANNEL_WICK_PERMEABILITY_M2,
    CHANNEL_WICK_PORE_RADIUS_M,
    calculate_capillary_head_max,
    calculate_capillary_margin,
    calculate_channel_temperatures,
    calculate_effective_channel_conductivity,
    calculate_effective_wick_conductivity,
    calculate_liquid_darcy_drop,
    calculate_mass_flow,
    calculate_thermal_resistance_network,
    calculate_total_pressure_drop,
    calculate_vapor_mach_number,
    calculate_vapor_pressure_drop,
    calculate_vapor_reynolds_number,
    regression_anchor_500w_650c,
)


class TestLayer2GeometryConstants:
    """Verify frozen canonical geometry for national lab capillary test bench."""

    def test_dimensions(self):
        assert CHANNEL_OUTER_DIAMETER_M == 19.05e-3
        assert CHANNEL_INNER_DIAMETER_M == 17.05e-3
        assert CHANNEL_VAPOR_DIAMETER_M == 15.05e-3
        assert CHANNEL_EFFECTIVE_LENGTH_M == 1.50
        # Sintered powder wick, d_p = 100 um (Chi 1976: r_eff = 0.21 d_p; Blake-Kozeny K)
        assert CHANNEL_WICK_PARTICLE_DIAMETER_M == 100.0e-6
        assert CHANNEL_WICK_PORE_RADIUS_M == pytest.approx(21.0e-6, rel=1e-9)
        assert CHANNEL_WICK_PERMEABILITY_M2 == pytest.approx(1.4946e-10, rel=1e-3)


class TestLayer2HydrodynamicsAndCapillaryMargin:
    """Hydrodynamic balance, Darcy losses, and capillary safety margins."""

    def test_nominal_operating_point_500w_650c(self):
        q = 500.0
        t = 650.0

        m_dot = calculate_mass_flow(q, t)
        dp_cap = calculate_capillary_head_max(t)
        dp_l = calculate_liquid_darcy_drop(q, t)
        dp_v = calculate_vapor_pressure_drop(q, t)
        dp_tot = calculate_total_pressure_drop(q, t)
        m_cap = calculate_capillary_margin(q, t)
        re_v = calculate_vapor_reynolds_number(q, t)

        # Mass flow ~ 0.122 g/s
        assert m_dot == pytest.approx(1.222e-4, rel=1e-2)
        # Oracle: independent hand calc (Opus audit 2026-10-06, script in HANDOFF), not code output.
        assert dp_cap == pytest.approx(13645.2, rel=1e-3)
        assert dp_l == pytest.approx(5949.1, rel=1e-3)
        assert dp_v == pytest.approx(128.34, rel=1e-3)
        # Total pressure drop = dp_l + dp_v
        assert dp_tot == pytest.approx(dp_l + dp_v, rel=1e-9)
        assert m_cap == pytest.approx(2.2452, rel=1e-3)
        assert calculate_vapor_mach_number(q, t) < 0.1
        # Strictly laminar vapor Reynolds number (< 1000)
        assert re_v < 1000.0

    @pytest.mark.parametrize(
        "q_w, t_c, min_margin, max_re",
        [
            (50.0, 625.0, 18.0, 100.0),   # Corner A
            (50.0, 750.0, 18.0, 100.0),   # Corner B
            (750.0, 625.0, 1.20, 850.0),  # Corner C
            (750.0, 750.0, 1.20, 850.0),  # Corner D
        ],
    )
    def test_four_corner_corridor_bounds(self, q_w, t_c, min_margin, max_re):
        m_cap = calculate_capillary_margin(q_w, t_c)
        re_v = calculate_vapor_reynolds_number(q_w, t_c)

        # Physical safety invariant: Capillary margin must strictly exceed 1.0 (no dryout)
        assert m_cap >= min_margin
        # Hydrodynamic invariant: Flow must be strictly laminar (Re_v < 2300)
        assert re_v <= max_re
        assert re_v < 2300.0

    @pytest.mark.parametrize(
        "bad_q, bad_t",
        [
            (49.9, 650.0),
            (750.1, 650.0),
            (500.0, 624.9),
            (500.0, 750.1),
            (float("nan"), 650.0),
            (500.0, float("inf")),
            (True, 650.0),
            (500.0, False),
        ],
    )
    def test_hydrodynamics_rejects_boundary_breaches(self, bad_q, bad_t):
        with pytest.raises(DomainBoundaryError):
            calculate_capillary_margin(bad_q, bad_t)


class TestThermalResistanceNetwork:
    """5-element thermal resistance network, temperatures, and equivalent conductance."""

    def test_effective_wick_thermal_conductivity(self):
        # Chi (1976) sintered-powder (Maxwell) form, hand calc 41.009 W/(m*K) at 650 C
        k_eff = calculate_effective_wick_conductivity(650.0)
        assert k_eff == pytest.approx(41.009, rel=1e-3)

    def test_thermal_resistance_network_elements_500w_650c(self):
        res = calculate_thermal_resistance_network(500.0, 650.0)
        assert res["r_wall_e"] == pytest.approx(8.211e-4, rel=1e-2)
        # Source: independent hand calc (Opus audit 2026-10-06)
        assert res["r_wick_e"] == pytest.approx(4.8423e-4, rel=1e-3)
        assert res["r_vapor"] == pytest.approx(2.4785e-3, rel=1e-3)
        assert res["r_wick_c"] == pytest.approx(4.8423e-4, rel=1e-3)
        assert res["r_wall_c"] == pytest.approx(8.211e-4, rel=1e-2)
        assert res["r_total"] == pytest.approx(5.0891e-3, rel=1e-3)

    def test_channel_temperature_drops_and_conductance(self):
        t_evap, t_cond, delta_t = calculate_channel_temperatures(500.0, 650.0)
        # Source: independent hand calc (Opus audit 2026-10-06)
        assert delta_t == pytest.approx(2.5445, rel=1e-3)
        # Interface boundary consistency: T_evap > T_sat > T_cond
        assert t_evap > 650.0
        assert t_cond < 650.0
        assert math.isclose(t_evap - t_cond, delta_t, rel_tol=1e-9)

        # Equivalent bulk thermal conductivity must vastly exceed solid copper (> 100,000 W/m*K)
        k_channel = calculate_effective_channel_conductivity(500.0, 650.0)
        assert k_channel > 1.0e6  # ~1.68 MW/(m*K)


class TestRegressionAnchor:
    """Regression anchor at (500 W, 650 C) against an independent hand calc.
    This is NOT a LANL/HTPIPE benchmark and makes no NQA-1 claim."""

    def test_regression_anchor_matches_hand_calc(self):
        r = regression_anchor_500w_650c()
        assert r["vapor_pressure_drop_pa"] == pytest.approx(128.34, rel=1e-3)
        assert r["liquid_darcy_drop_pa"] == pytest.approx(5949.1, rel=1e-3)
        assert r["temperature_drop_k"] == pytest.approx(2.5445, rel=1e-3)
        assert "nqa1_verification_status" not in r
