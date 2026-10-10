# Architectural Steering & Verification Memorandum: `sfr-tdd-mcp`

**System:** Single-Channel Liquid Sodium Heat Pipe Steady-State Solver & FastMCP Interface  
**Engineering Methodology:** Human-Directed Test-Driven Development (TDD) with Adversarial AI Multi-Model Auditing  
**Author Stance:** Systems Architecture & AI Orchestration (Non-Physicist Human-in-the-Loop)  
**Active Steering Effort:** ~3 Engineering Hours (architecture framing, literature reconciliation, and verification harnesses)  
**Verification Baseline:** 181 passing unit tests in 0.41s (`pytest`), clean static typing (`mypy --strict`), clean linting (`ruff`)

---

## 1. Architectural Paradigm & Steering Model

This repository was developed using a strict **Human-Directed Test-Driven Development (TDD)** workflow paired with LLM code synthesis and adversarial multi-model verification.

### Transparency of Roles: Systems Orchestrator vs. Generative Model
The author's engineering background is in systems architecture, enterprise software reliability, and open-source provenance—not nuclear reactor thermal-hydraulics. The objective of this project was not to manually derive alkali metal physics from first principles, but to demonstrate how an experienced software systems architect can rigorously orchestrate and police generative AI to produce defensible scientific software without falling victim to hallucinated physics or compliance theater.

* **Human Systems Architect (Orchestrator):**
  * Defines architectural layer separation (invariants $\rightarrow$ constitutive media $\rightarrow$ hydrodynamic solver $\rightarrow$ protocol interface).
  * Establishes strict verification invariants: zero ungrounded physics formulas, mandatory citations to primary scanned literature (`ANL/RE-95/2`, Chi 1976), and zero synthetic compliance claims.
  * Formulates test contracts, boundary corridors, and defensive typing gates.
  * Orchestrates adversarial audit loops: directing cold-context models to cross-examine generated equations against primary literature tables, challenging unearned benchmark passes, and enforcing remediation.
* **LLM Synthesis & Audit Pairing:**
  * Procedurally implements mathematical functions satisfying pre-written test contracts.
  * Implements literature correlations under strict static typing (`mypy --strict`).
  * Executes automated cross-checks against scanned national laboratory data tables.
  * Generates FastMCP protocol tool registrations and JSON Schema wrappers.

### Git Cadence & Development Velocity
The git commit history reflects this asymmetric pairing model:
1. **Procedural Code Bursts (Sub-Minute Cadence):** During initial implementation sprints, automated test-and-code commit pairs landed in rapid succession (20–40 seconds apart) as procedural generators satisfied pre-specified red/green test contracts.
2. **Discrete Human Verification Intervals (30–45 Minute Pauses):** The substantial gaps between commit clusters represent offline human engineering: directing models to examine scanned ANL reports, questioning unverified assumptions, auditing generated code for physical consistency, and forcing remediation test suites.

---

## 2. Key Technical Catches & Adversarial Corrections

The primary value of human architectural steering in AI-assisted scientific software is identifying and correcting subtle physical invalidities and hallucinatory compliance artifacts that pass superficial testing. Five key catches illustrate this verification dynamic:

---

### Catch 1: Alkali Vapor Dimerization Truncation & The Clapeyron Relation (Layer 1)
* **The Failure Mode:** The pairing model initially implemented an explicit quasi-chemical equilibrium model for sodium vapor dimerization ($2\text{Na} \rightleftharpoons \text{Na}_2$ via Ewing et al., 1967). While vastly superior to a monomer ideal-gas equation of state (which underpredicts vapor density by $8.92\%$ at 750 °C), this truncated equilibrium model neglected higher-order molecular associations and drifted $2.70\%$ low against ANL/RE-95/2 benchmark tables (yielding $0.0758\text{ kg/m}^3$ vs ANL's $0.0779\text{ kg/m}^3$ at 750 °C, failing Table 1.3-1 parity tests).
* **The Steering Correction:** When the audit flagged the discrepancy against ANL Table 1.3-1, the orchestrator directed the solver to discard the truncated chemical model and implement ANL/RE-95/2 Section 1.3.1's exact thermodynamic Clapeyron relation. This relation implicitly accounts for dimerization and real-gas compressibility through the quasi-chemical enthalpy of vaporization fit (Golden & Tokar):
  $$\rho_v = \frac{1}{\frac{h_{fg}}{T \cdot (dP/dT)_{\text{sat}}} + \frac{1}{\rho_l}}$$
  where the saturation pressure gradient is differentiated analytically from Fink & Leibowitz Eq. 1:
  $$\left(\frac{dP}{dT}\right)_{\text{sat}} = P_{\text{sat}} \cdot \left[ \frac{12633.73}{T^2} - \frac{0.4672}{T} \right]$$
* **Verification Gate:** Calculated vapor density was benchmarked directly against printed values in ANL Table 1.3-1 (p. 87 / PDF p. 108) at 900 K and 1000 K, achieving parity within 0.4% (the limit of printed table rounding).

---

### Catch 2: Wick Microstructure Mechanics & Transport Asymmetry (Layer 2)
* **The Failure Mode:** Early solver iterations combined constants for sintered spherical powder with equations derived for wrapped screen-wire mesh (specifically applying Chi's Maxwell-Eucken screen formulation while defining spherical powder porosity), creating an unphysical composite microstructure.
* **The Steering Correction:** The orchestrator caught the mismatched citations and forced unification of the porous media model to Chi (1976) sintered spherical powder geometry:
  * Particle diameter: $d_p = 100\ \mu\text{m}$, high-porosity matrix: $\epsilon = 0.65$.
  * Effective capillary radius: $r_{\text{eff}} = 0.21 d_p = 21\ \mu\text{m}$ (Chi Table 2.1).
  * Wick permeability: $K = \frac{d_p^2 \epsilon^3}{150 (1-\epsilon)^2} = 1.495 \times 10^{-10}\text{ m}^2$ (Blake-Kozeny equation).
  * Effective thermal conductivity: $k_{\text{eff}} = 41.01\text{ W/(m}\cdot\text{K)}$ (Maxwell relation for continuous solid matrix with spherical liquid inclusions, bounded between series and parallel limits).
* **Physical Transport Asymmetry:** Detailed verification highlighted the defining physical asymmetry of liquid metal heat pipes:
  * **Hydrodynamic drop is governed by the liquid phase:** Liquid Darcy porous drag accounts for $97.9\%$ of total pressure drop ($\Delta P_l \approx 5949\text{ Pa}$ vs $\Delta P_v \approx 128\text{ Pa}$). Capillary pumping margin ($M_{\text{cap}} = 2.245$) is virtually independent of vapor friction.
  * **Thermal resistance is governed by the vapor phase:** Conversely, axial vapor thermal resistance accounts for $48.7\%$ of the total temperature drop ($R_{\text{vapor}} \approx 2.48 \times 10^{-3}\text{ K/W}$ out of $R_{\text{total}} = 5.089 \times 10^{-3}\text{ K/W}$, yielding $\Delta T = 2.54\text{ K}$).

---

### Catch 3: SQA Audit Remediation: Expunging Synthetic Benchmark Claims (Layer 2)
* **The Failure Mode:** The pairing model hallucinated an automated verification function titled `verify_lanl_htpipe_benchmark()` that hardcoded expected outputs (`TARGET_DP_L_PA = 6150.0`, `TARGET_DELTA_T_K = 2.51`) and asserted an `"nqa1_verification_status": "VERIFIED_PASS"`.
* **The SQA Reality:** In safety-critical software quality assurance (ASME NQA-1 Subpart 2.7 / 10 CFR 50 Appendix B), claiming third-party code verification without qualified input decks, documented nodal discretization, and traceable pedigree constitutes unacceptable compliance theater. Examination of LANL report `LA-11324-M` confirmed it contains no single-channel validation vector matching this exact geometry.
* **The Steering Correction:** Recognizing the hallucinated compliance pass, the orchestrator immediately purged all synthetic benchmark assertions, mock NQA-1 pass flags, and hardcoded targets. They were replaced with an honest, independent **hand-calculation regression anchor** documenting exact formula evaluations, accompanied by an explicit disclaimer in `README.md` (*"No ASME NQA-1 or 10 CFR 50 Appendix B qualification is claimed"*).

---

### Catch 4: FastMCP Protocol Schema Projection (Layer 3)
* **The Failure Mode:** Initial tool definitions exposed bare Python type annotations (`t_inlet_c: float`), producing an unconstrained JSON Schema (`{"type": "number"}`) via FastMCP introspection. Autonomous client LLMs calling the tool had no visibility into physical domain boundaries ($500\ ^\circ\text{C} \le T \le 750\ ^\circ\text{C}$) until encountering runtime invariant exceptions.
* **The Steering Correction:** Refactored tool parameters to use Pydantic v2 `Annotated[float, Field(ge=500.0, le=750.0, description="...")]`. Verified via `mcp.list_tools()` that boundary corridors project directly into the published client-facing JSON Schema, enabling caller-side validation before RPC execution.

---

### Catch 5: Defensive Numeric Boundary Invariants (Layer 0)
* **The Failure Mode:** Standard Python numeric type checking treats booleans as integers (`isinstance(True, int) == True`). Passing `True` into geometric parameters (e.g., $L = \text{True}$) bypassed naive type guards, silently evaluating as $1.0\text{ meter}$. Furthermore, IEEE-754 non-finites (`NaN`, `+Inf`, `-Inf`) slipped past standard comparison operators, threatening silent corruption of downstream iterative solvers.
* **The Steering Correction:** Established strict Layer 0 invariant guards:
  * Explicitly rejected boolean types (`type(v) is not bool`).
  * Enforced finiteness (`math.isfinite(v)`).
  * Asserted strict physical corridor boundaries prior to passing parameters to Layer 1 property routines.

---

## 3. Layer-by-Layer Verification Architecture

```
+-----------------------------------------------------------------------------------+
| LAYER 3: Protocol Interface & Schema Exposure (server.py)                         |
| FastMCP JSON-RPC Server | Pydantic v2 Boundary Projection | Stdio Communication   |
+-----------------------------------------+-----------------------------------------+
                                          |
+-----------------------------------------v-----------------------------------------+
| LAYER 2: Computational Physics & Hydrodynamic Solver (layer2_heat_pipe_solver.py) |
| Young-Laplace Capillary Head | Darcy Wick Drag | 5-Element Thermal Resistance Net |
+-----------------------------------------+-----------------------------------------+
                                          |
+-----------------------------------------v-----------------------------------------+
| LAYER 1: ANL Primary Constitutive Media (layer1_sodium_properties.py)             |
| Liquid Properties (Hornung, Shpil'rain) | Vapor Properties (Exact Clapeyron)      |
+-----------------------------------------+-----------------------------------------+
                                          |
+-----------------------------------------v-----------------------------------------+
| LAYER 0: Numerical Invariants & Corridor Gatekeepers (layer0_invariants.py)       |
| Strict Type Filtering (bool guard) | IEEE-754 Finiteness | Corridor Bounds        |
+-----------------------------------------------------------------------------------+
```

### Verification Matrix by Subsystem

| Subsystem | Governing Physics / Standard | Verification Method | Pass Criteria |
| :--- | :--- | :--- | :--- |
| **Layer 0: Invariants** | Defensive type safety, IEEE-754 non-finite rejection | Parameterized adversarial fuzzing (`test_layer0_invariants.py`) | 100% rejection of booleans, strings, `NaN`, `Inf`, out-of-range floats |
| **Layer 1: Liquid Na** | ANL/RE-95/2 Table 1.1-2 (Hornung liquid density, Shpil'rain viscosity) | Parity tests against printed ANL reference tables (`test_layer1_sodium_properties.py`) | Agreement within $\pm 0.1\%$ across $500\ ^\circ\text{C} - 750\ ^\circ\text{C}$ operating corridor |
| **Layer 1: Vapor Na** | ANL/RE-95/2 Table 1.3-1 (Clapeyron vapor density via analytical $dP/dT$) | Unit tests against printed ANL reference tables | Agreement within $\pm 0.4\%$ (matching printed table precision) |
| **Layer 2: Hydrodynamics** | Young-Laplace capillary head vs Blake-Kozeny porous wick drag | Analytical balance test against independent hand-calculations | Capillary margin $M_{\text{cap}} = 2.245 > 1.0$; $Re_v < 800$ laminar check |
| **Layer 2: Thermal Network** | 5-element resistance network (envelope, wick, vapor core) | Temperature drop assertion against hand-calculated nodal sum | Total $\Delta T = 2.54\text{ K}$; $R_{\text{total}} = 5.089 \times 10^{-3}\text{ K/W}$ |
| **Layer 3: Protocol API** | FastMCP JSON-RPC stdio transport, Pydantic v2 schemas | End-to-end tool execution and schema inspection (`test_server.py`) | Valid JSON response; input parameters bound to $[50.0, 750.0]$ |

---

## 4. Scope Limits, Parametric Uncertainty & Engineering Disclaimers

1. **Operating Envelope:** Validated strictly for single-channel, steady-state sodium heat pipes operating horizontally in the laminar vapor regime ($Re_v < 2000$, $Ma < 0.2$) between $500\ ^\circ\text{C}$ and $750\ ^\circ\text{C}$. Not valid for startup from frozen state, gravity-assisted thermosyphons, or sonic/entrainment limit regimes.
2. **Parametric Uncertainties:**
   * Liquid sodium surface tension ($\sigma$): Evaluated via ANL/RE-95/2 Section 1.2. ANL reports experimental scatter of $\pm 11\%$ ($2\sigma$ confidence limit).
   * Sodium vapor dynamic viscosity ($\mu_v$): ANL/RE-95/2 provides no experimental vapor viscosity correlation. The solver implements a provisional power-law relation ($2.06 \times 10^{-5}\text{ Pa}\cdot\text{s}$ at $650\ ^\circ\text{C}$). Because vapor thermal resistance accounts for $48.7\%$ of total $\Delta T$, calculated overall temperature drop carries an estimated physical uncertainty of $\pm 15\%$.
3. **Regulatory / QA Status:** This software is a focused computational demonstration. It has not been audited or qualified under ASME NQA-1-2015, 10 CFR 50 Appendix B, or IEEE 730-2014 standards.
