# sfr-tdd-mcp

A small, steady-state calculator for a single horizontal sodium heat pipe, exposed as an MCP tool (`calculate_heatpipe_heat_transfer`).

This is a portfolio demonstration. It is not a qualified design or safety code.

## Scope

- **Corridor:** 625–750 °C saturation temperature, 50–750 W, horizontal (θ = 0°), steady state only. Inputs outside this corridor are rejected.
- **Geometry (frozen):**
  - Tube: Do = 19.05 mm, wall 1.0 mm
  - Wick: 1.0 mm annular, sintered stainless powder (d_p = 100 µm, ε = 0.65)
  - Vapor core: Dv = 15.05 mm
  - Lengths: Le = 1.0 m, La = 0.5 m, Lc = 1.0 m
- **Outputs:** mass flow, wall temperatures, ΔT, capillary margin, vapor Re and Mach number, and a flow regime computed from those numbers.

## Physics and sources

| Item | Model | Source |
|---|---|---|
| Liquid ρ, Cp, σ, μ, k; P_sat; ΔH_v | ANL correlations | ANL/RE-95/2 (Fink & Leibowitz, 1995) |
| Vapor density | Clausius-Clapeyron relation using ANL P_sat and ΔH_v | ANL/RE-95/2 method |
| Vapor viscosity | **PROVISIONAL** power-law fit, not yet sourced | to be replaced (Golden & Tokar, ANL-7323) |
| Wick r_eff, K | r_eff = 0.21 d_p; Blake-Kozeny | Chi (1976) |
| Wick k_eff | sintered (Maxwell) form | Chi (1976) |
| Pressure drops | Darcy (liquid), Hagen-Poiseuille over L_eff (vapor) | standard |
| Thermal network | wall + wick (radial) ×2, Clausius-Clapeyron vapor resistance | standard |

## Why other operating limits are excluded

Capillary pumping is the binding limit. Hand calculations at the worst case (625 °C), using this code's property functions:

| Limit | Q_max |
|---|---|
| Capillary | ~1.1 kW (M_cap ≥ 1.48 everywhere in the corridor) |
| Sonic (Levy) | ~3.2 kW |
| Entrainment | ~5.5 kW |
| Viscous (Busse) | ~7.1 kW |

Across the corridor, the vapor flow is laminar (Re_v < 800) and Ma < 0.09, so the incompressible laminar assumption holds.

## Verification status (plain version)

- **Liquid properties:** formulas match the ANL/RE-95/2 correlations. The tests compare against the correlations evaluated at grid points, not yet against the report's printed tables. Adding those table rows is open work.
- **Layer 2:** the tests check against an independent hand calculation at 500 W / 650 °C, done outside the codebase. This is a regression anchor, **not** a code-to-code benchmark against LANL HTPIPE or experimental data.
- **No NQA-1 qualification is claimed.**

## How it was built

Built AI-paired. I set scope, architecture, and invariants; the model wrote tests first, then code. An independent model audit caught a mislabeled benchmark and an inconsistent wick spec, and both were fixed. See the git log, which keeps the red/green history.

## Run

```
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 -m sfr_mcp.server
```
