import pytest

from sfr_mcp.exceptions import DomainBoundaryError
from sfr_mcp.layer1_sodium_properties import (
    CELSIUS_TO_KELVIN_OFFSET,
    celsius_to_kelvin,
    liquid_sodium_density,
    liquid_sodium_dynamic_viscosity,
    liquid_sodium_mass_density,
    liquid_sodium_surface_tension,
    liquid_sodium_thermal_conductivity,
    sodium_enthalpy_of_vaporization,
    sodium_saturated_vapor_density,
    sodium_saturated_vapor_pressure,
    sodium_vapor_dynamic_viscosity,
)


class TestCelsiusToKelvinConversion:
    """Universal temperature conversion contract: works for any valid real temperature."""

    @pytest.mark.parametrize(
        "temp_c, expected_k",
        [
            (-100.0, 173.15),
            (0.0, 273.15),
            (50.0, 323.15),  # 50 C is valid in general physics!
            (97.80, 370.95),
            (100.0, 373.15),
            (400.0, 673.15),
            (500, 773.15),  # int coercion
            (883.00, 1156.15),
            (1200.0, 1473.15),
        ],
    )
    def test_nominal_temperature_conversion(self, temp_c, expected_k):
        """Verify accurate mathematical conversion from Celsius to Kelvin across all states."""
        assert celsius_to_kelvin(temp_c) == pytest.approx(expected_k, rel=1e-9)

    def test_celsius_to_kelvin_offset_constant(self):
        """Verify the exact canonical offset constant."""
        assert CELSIUS_TO_KELVIN_OFFSET == 273.15

    @pytest.mark.parametrize(
        "invalid_temp",
        [
            float("nan"),
            float("inf"),
            float("-inf"),
            True,
            False,
            "50",
            None,
            -300.0,  # Below absolute zero
            -273.15,  # Absolute zero boundary
        ],
    )
    def test_reject_unphysical_or_non_numeric(self, invalid_temp):
        """Verify unit conversion rejects sub-absolute zero and non-numeric inputs."""
        with pytest.raises(DomainBoundaryError):
            celsius_to_kelvin(invalid_temp)


class TestLiquidSodiumMassDensity:
    """
    Thermophysical property test suite for liquid sodium mass density rho(T) [kg/m^3].
    
    Primary Source: ANL/RE-95/2 (Fink & Leibowitz, 1995, Section 1.3.1, Eq. 1, p. 86).
    Hornung Formulation:
        theta = 1 - T_K / 2503.7
        rho = 219.0 + 275.32*theta + 511.58*theta^0.5
    """

    @pytest.mark.parametrize(
        "temp_c, anl_hornung_expected_rho",
        [
            (100.0, 925.77),
            (250.0, 892.48),
            (400.0, 857.73),
            (550.0, 822.93),
            (700.0, 787.29),
            (850.0, 750.69),
        ],
    )
    def test_density_anl_benchmark_parity(self, temp_c, anl_hornung_expected_rho):
        """Verify liquid sodium mass density matches ANL/RE-95/2 Hornung formulation within 0.1%."""
        computed = liquid_sodium_mass_density(temp_c)
        assert computed == pytest.approx(anl_hornung_expected_rho, rel=1e-3)
        # Verify backward-compatible alias parity
        assert liquid_sodium_density(temp_c) == computed

    def test_density_strictly_monotonically_decreasing(self):
        """Universal physical law: Thermal expansion requires d(rho)/dT < 0 everywhere in liquid phase."""
        temps_c = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 850.0]
        densities = [liquid_sodium_mass_density(t) for t in temps_c]
        
        for i in range(len(densities) - 1):
            assert densities[i] > densities[i + 1], (
                f"Mass density violation: rho({temps_c[i]}°C)={densities[i]} not > "
                f"rho({temps_c[i+1]}°C)={densities[i+1]}"
            )

    @pytest.mark.parametrize(
        "invalid_state",
        [
            float("nan"),
            float("inf"),
            True,
            False,
            "400",
            None,
            50.0,     # Solid sodium (melting point 97.80°C)
            97.80,    # Freezing boundary
            883.00,   # Boiling boundary at 1 atm
            950.0,    # Vapor state
        ],
    )
    def test_liquid_density_rejects_non_liquid_phases(self, invalid_state):
        """Verify liquid mass density strictly raises DomainBoundaryError on non-liquid phases."""
        with pytest.raises(DomainBoundaryError):
            liquid_sodium_mass_density(invalid_state)
        with pytest.raises(DomainBoundaryError):
            liquid_sodium_density(invalid_state)


class TestANLLiquidSodiumThermophysicalParity:
    """
    Tier-1 ANL/RE-95/2 Parity Suite across the 5-point discrete test grid:
    T in {625.0 C, 650.0 C, 675.0 C, 700.0 C, 750.0 C}.
    Asserts parity within 0.05% of ANL benchmark values.
    """

    @pytest.mark.parametrize(
        "temp_c, exp_sigma, exp_mu_l, exp_kl",
        [
            (625.0, 0.14583, 2.0100e-4, 58.421),
            (650.0, 0.14327, 1.9552e-4, 57.354),
            (675.0, 0.14073, 1.9041e-4, 56.315),
            (700.0, 0.13818, 1.8564e-4, 55.302),
            (750.0, 0.13311, 1.7697e-4, 53.354),
        ],
    )
    def test_anl_properties_parity(self, temp_c, exp_sigma, exp_mu_l, exp_kl):
        sigma = liquid_sodium_surface_tension(temp_c)
        mu_l = liquid_sodium_dynamic_viscosity(temp_c)
        kl = liquid_sodium_thermal_conductivity(temp_c)

        assert sigma == pytest.approx(exp_sigma, rel=5e-4)  # <= 0.05%
        assert mu_l == pytest.approx(exp_mu_l, rel=5e-4)  # <= 0.05%
        assert kl == pytest.approx(exp_kl, rel=5e-4)  # <= 0.05%

    def test_liquid_properties_monotonicity(self):
        """Verify fundamental physical monotonicity across corridor [625-750 C]."""
        grid = [625.0, 650.0, 675.0, 700.0, 725.0, 750.0]
        sigmas = [liquid_sodium_surface_tension(t) for t in grid]
        viscosities = [liquid_sodium_dynamic_viscosity(t) for t in grid]
        conductivities = [liquid_sodium_thermal_conductivity(t) for t in grid]

        for i in range(len(grid) - 1):
            assert sigmas[i] > sigmas[i + 1], "Surface tension must decrease with T"
            assert viscosities[i] > viscosities[i + 1], "Viscosity must decrease with T"
            assert conductivities[i] > conductivities[i + 1], "Thermal conductivity must decrease with T"

    @pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf"), True, False, "650", None, 50.0, 950.0])
    def test_liquid_properties_reject_bad_inputs(self, bad_val):
        with pytest.raises(DomainBoundaryError):
            liquid_sodium_surface_tension(bad_val)
        with pytest.raises(DomainBoundaryError):
            liquid_sodium_dynamic_viscosity(bad_val)
        with pytest.raises(DomainBoundaryError):
            liquid_sodium_thermal_conductivity(bad_val)


class TestANLSodiumVaporAndSaturationParity:
    """
    Tier-1 ANL/RE-95/2 Parity Suite for vapor and saturation properties
    across the 5-point discrete test grid:
    T in {625.0 C, 650.0 C, 675.0 C, 700.0 C, 750.0 C}.
    Asserts parity within 0.05% of ANL benchmark values.
    """

    @pytest.mark.parametrize(
        "temp_c, exp_psat, exp_hfg, exp_rhov, exp_muv",
        [
            (625.0, 5005.5, 4113914.5, 0.01655, 2.0180e-05),
            (650.0, 7233.0, 4092269.5, 0.02336, 2.0600e-05),
            (675.0, 10247.3, 4070425.3, 0.03237, 2.1017e-05),
            (700.0, 14255.8, 4048376.4, 0.04407, 2.1431e-05),
            (750.0, 26263.6, 4003641.0, 0.07793, 2.2252e-05),
        ],
    )
    def test_anl_vapor_properties_parity(self, temp_c, exp_psat, exp_hfg, exp_rhov, exp_muv):
        psat = sodium_saturated_vapor_pressure(temp_c)
        hfg = sodium_enthalpy_of_vaporization(temp_c)
        rhov = sodium_saturated_vapor_density(temp_c)
        muv = sodium_vapor_dynamic_viscosity(temp_c)

        assert psat == pytest.approx(exp_psat, rel=5e-4)  # <= 0.05%
        assert hfg == pytest.approx(exp_hfg, rel=5e-4)    # <= 0.05%
        assert rhov == pytest.approx(exp_rhov, rel=5e-4)  # <= 0.05%
        assert muv == pytest.approx(exp_muv, rel=5e-4)    # <= 0.05%

    def test_vapor_properties_monotonicity(self):
        """Verify fundamental physical monotonicity across corridor [625-750 C]."""
        grid = [625.0, 650.0, 675.0, 700.0, 725.0, 750.0]
        pressures = [sodium_saturated_vapor_pressure(t) for t in grid]
        latent_heats = [sodium_enthalpy_of_vaporization(t) for t in grid]
        densities = [sodium_saturated_vapor_density(t) for t in grid]
        viscosities = [sodium_vapor_dynamic_viscosity(t) for t in grid]

        for i in range(len(grid) - 1):
            assert pressures[i] < pressures[i + 1], "Vapor pressure must increase with T"
            assert latent_heats[i] > latent_heats[i + 1], "Latent heat must decrease with T"
            assert densities[i] < densities[i + 1], "Vapor density must increase with T"
            assert viscosities[i] < viscosities[i + 1], "Vapor viscosity must increase with T"

    @pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf"), True, False, "650", None, 50.0, 950.0])
    def test_vapor_properties_reject_bad_inputs(self, bad_val):
        with pytest.raises(DomainBoundaryError):
            sodium_saturated_vapor_pressure(bad_val)
        with pytest.raises(DomainBoundaryError):
            sodium_enthalpy_of_vaporization(bad_val)
        with pytest.raises(DomainBoundaryError):
            sodium_saturated_vapor_density(bad_val)
        with pytest.raises(DomainBoundaryError):
            sodium_vapor_dynamic_viscosity(bad_val)




class TestANLPrintedTables:
    """
    Independent oracles copied VERBATIM from the scanned ANL/RE-95/2 report
    (Fink & Leibowitz 1995), Table 1.3-1 "Sodium Density", printed p. 87
    (PDF p. 108). Table values have 3 significant figures, so tolerances
    reflect table rounding, not correlation error.
    Note: the pre-audit Na/Na2 dimerization rho_v model ran ~2.6% low and
    FAILS the vapor rows below; the ANL thermodynamic relation passes.
    """

    @pytest.mark.parametrize(
        "temp_k, table_rho_l, table_rho_v",
        [
            (900.0, 805.0, 1.70e-2),
            (1000.0, 781.0, 6.03e-2),
        ],
    )
    def test_table_1_3_1_density(self, temp_k, table_rho_l, table_rho_v):
        temp_c = temp_k - 273.15
        assert liquid_sodium_mass_density(temp_c) == pytest.approx(table_rho_l, rel=1e-3)
        assert sodium_saturated_vapor_density(temp_c) == pytest.approx(table_rho_v, rel=4e-3)
