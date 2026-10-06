import math
import pytest
from sfr_mcp.exceptions import DomainBoundaryError
from sfr_mcp.layer2_heat_pipe_solver import (
    CHANNEL_OUTER_DIAMETER_M,
    CHANNEL_INNER_DIAMETER_M,
    CHANNEL_VAPOR_DIAMETER_M,
    CHANNEL_EFFECTIVE_LENGTH_M,
    CHANNEL_WICK_PORE_RADIUS_M,
    CHANNEL_WICK_PERMEABILITY_M2,
    calculate_mass_flow,
    calculate_capillary_head_max,
    calculate_liquid_darcy_drop,
    calculate_vapor_pressure_drop,
    calculate_total_pressure_drop,
    calculate_capillary_margin,
    calculate_vapor_reynolds_number,
)


class TestLayer2GeometryConstants:
    """Verify frozen canonical geometry for national lab capillary test bench."""

    def test_dimensions(self):
        assert CHANNEL_OUTER_DIAMETER_M == 19.05e-3
        assert CHANNEL_INNER_DIAMETER_M == 17.05e-3
        assert CHANNEL_VAPOR_DIAMETER_M == 15.05e-3
        assert CHANNEL_EFFECTIVE_LENGTH_M == 1.50
        assert CHANNEL_WICK_PORE_RADIUS_M == 25.0e-6
        assert CHANNEL_WICK_PERMEABILITY_M2 == 1.50e-10


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
        # Capillary head ~ 11.46 kPa
        assert dp_cap == pytest.approx(11462.0, rel=1e-2)
        # Darcy liquid loss ~ 5.93 kPa
        assert dp_l == pytest.approx(5928.0, rel=1e-2)
        # Vapor Hagen-Poiseuille loss ~ 132 Pa
        assert dp_v == pytest.approx(131.8, rel=1e-2)
        # Total pressure drop = dp_l + dp_v
        assert dp_tot == pytest.approx(dp_l + dp_v, rel=1e-9)
        # Capillary margin ~ 1.89x
        assert m_cap == pytest.approx(1.89, rel=2e-2)
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
