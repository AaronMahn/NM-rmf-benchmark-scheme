---
name: rmf-toolkit
description: Work with RMF / mean-field EOS tooling, especially agentV1-style shared libraries for model definition, beta equilibrium, EOS generation, and TOV/MR pipelines. Use when designing or extending a reusable RMF toolkit, connecting baseline models and their variants, deciding how to represent changing Lagrangians, or documenting input/output conventions and pitfalls for model-agnostic mean-field model families.
---

# RMF Toolkit

Treat the toolkit as a shared physics infrastructure layer, not as one-off scripts for a single model.

## Keep the architecture split into layers

Separate these concerns clearly:

1. **Model structure layer**
   - Define the Lagrangian, field content, and interaction terms.
   - Decide which fields exist and which couplings or self-interaction terms are active.

2. **Model-instance layer**
   - Define concrete models such as `TM1`, `TMV1`, ..., `TMVn`.
   - Keep each model instance explicit and self-describing.

3. **EOS generation layer**
   - Solve beta equilibrium or related mean-field equations on a density grid.
   - Produce EOS tables such as `rho_b / pressure / epsilon`.

4. **TOV/MR layer**
   - Take EOS tables as inputs.
   - Compute MR curves and derived quantities such as `R1.4`, `R2.0`, `mmax`.

Do not mix these layers in one opaque script unless there is a strong reason.

## Primary input rule: ask for the Lagrangian and its conventions directly

If the Lagrangian itself changes across models, do not force the workflow into a rigid batch-table mindset first. Start from the physics definition.

Request these pieces explicitly:

1. **Lagrangian expression**
   - Prefer LaTeX or a clean typed formula.

2. **Symbol definitions**
   - Clarify what each field and coupling means.
   - Clarify whether couplings are constants or density-dependent.

3. **Model assumptions**
   - Zero temperature or not
   - Uniform matter or not
   - Mean-field approximation or not
   - Matter content / charge neutrality / beta-equilibrium assumptions

4. **Parameter values**
   - Give actual masses and couplings.
   - If a baseline-based description is used, state what is inherited unchanged from the chosen baseline model.

5. **Implementation note**
   - State which terms are unchanged, modified, added, or removed relative to the chosen baseline model, if a baseline-based description is used.
   - State whether the EOS and pressure/energy-density expressions follow an existing reference.

If the user only gives a formula and no parameter values or assumptions, treat that as insufficient for direct execution.

## Use parameter tables only as optional convenience

Parameter tables are helpful only when several model instances share the same Lagrangian structure and only numeric values differ.

Examples:

- one baseline model
- several close variants
- same field content and same interaction terms
- only couplings or masses change

In that case, one row = one model instance is a reasonable convenience. But do not make this the conceptual center of the skill.

## Design shared libraries to avoid model-specific hardcoding

Prefer model-agnostic interfaces.

For beta-equilibrium and EOS tooling, do not hardcode a fixed field count or fixed field ordering if the toolkit is meant to support multiple RMF families.

Use model metadata such as:

- field names
- aliases
- compiled model descriptors

Keep the shared path usable by both TM1-like and more extended models.

## Recommended implementation workflow

1. Read the Lagrangian and identify the active fields and interaction terms.
2. Write down all required parameters and inherited baseline assumptions.
3. Translate the theory into mean-field equations and EOS expressions.
4. Build or update the model object used by the shared toolkit.
5. Run EOS generation on the density grid.
6. Feed the EOS table into TOV/MR tools.
7. Save both raw outputs and compact summaries.

## Computing the four nuclear-matter observables — use the bundled helpers, do not hand-roll

These are evaluated at the GIVEN density (e.g. rho_B = 0.16 fm^-3), which for random-coupling
models is generally NOT the saturation point (pressure P != 0 there). Prefer the helpers in
`NMtoolsV1/nuclear_properties.py` over re-deriving formulas:

- Binding energy: E/A = epsilon/rho - m_N  (symmetric matter, rho_n = rho_p = rho_B/2).
- Pressure: P = rho_n*mu_n + rho_p*mu_p - epsilon  (analytic from the self-consistent fields; exact).
- Compressibility / incompressibility K: use `compressibility_from_epsilon(epsilon_func, rho_B, m_N)`.
  Its definition is the GENERAL one,
      K = 9*( 2*rho*(E/A)' + rho**2*(E/A)'' ) = 9*dP/drho .
  DO NOT use the saturation-only form K = 9*rho**2*(E/A)''. That is valid only where P = 0;
  at an off-saturation density it omits the 18*rho*(E/A)' = 18*P/rho term and can even flip the
  SIGN of K (a common bug: K comes out negative). Sanity check: K should satisfy K ≈ 9*dP/drho
  (finite-difference the accurately-computed pressure); if your K is negative or differs from
  9*dP/drho, you used the wrong formula.
- Symmetry energy: E_sym = (1/2) d^2(E/A)/ddelta^2 at delta=0, asymmetry delta=(rho_n-rho_p)/rho;
  a delta ≈ 0.02 second difference is adequate.

## Field-equation solving — verify convergence and identify the branch

Use `NMtoolsV1/solver_checks.py` for field roots. It checks the solver status, finite fields,
positive finite proton/neutron effective masses, and
`max(abs(EOM)) / max((197.327**3) * rho_B, 1) <= 1e-8`. An `ier=1` flag alone is insufficient.
The helper retries explicit seeds and a scaled least-squares solve, then raises
`FieldConvergenceError` if no candidate passes. Propagate that failure into the output status;
do not turn a failed density into a successful six-vector or silently omit it from the EOS.

Warm-start from the previous verified density when appropriate. Beta equilibrium additionally
requires a final chemical-potential residual no larger than `1e-6 MeV`; a sign change across
a discontinuous field branch does not establish a root. The compiler derives vector chemical
potentials from the supplied symbolic energy, so coupling names need not match a baseline model.

Multiple verified roots can still exist. The helper returns the first verified root under its
documented seed order; it does not prove a global energy minimum or a physically stable phase.
Do not impose a universal effective-mass fraction window on synthetic random-coupling models.
Record branch assumptions, nonmonotonic cores, instability and extrapolation limits separately.


## Practical cautions

- Keep units explicit for every mass and coupling.
- Record what is inherited from the baseline model and what is modified whenever a baseline-based specification is used.
- Do not merge physically different model families into one ambiguous input file without documenting the structural difference.
- Preserve an explicit baseline identifier whenever a baseline-based specification is used.
- Treat EOS generation and MR analysis as separate stages even if one runner executes both.
- Distinguish clearly between quantities used for filtering and quantities used for final plotting or presentation.

## Documentation focus

When documenting the toolkit, explain these first:

- what the theory-level input is
- what assumptions are fixed
- what defines one model instance
- what files are produced
- which quantities are used for filtering versus which are used for plotting or presentation

## Bundled tool files

This skill now bundles a snapshot of the existing `agentV1/tools` code under:

```text
scripts/agentV1-tools/
```

Key bundled modules include:

- `scripts/agentV1-tools/NMtoolsV1/analytic_layer.py`
- `scripts/agentV1-tools/NMtoolsV1/numeric_layer.py`
- `scripts/agentV1-tools/NMtoolsV1/nuclear_properties.py`
- `scripts/agentV1-tools/TOVtools/beta_eos.py`
- `scripts/agentV1-tools/TOVtools/hybrid_eos.py`
- `scripts/agentV1-tools/TOVtools/tov_solver.py`
- `scripts/agentV1-tools/TOVtools/tool_beta_eff.py`

Treat these as the concrete implementation snapshot that goes with the skill guidance. Read them when you need the actual code paths rather than only the architectural notes.

## Released model and scoring scripts

- `scripts/agentV1-tools/model_io.py` loads a JSON model definition and builds the analytic,
  numeric, and beta-equilibrium model in one call. The schema and expression conventions are
  documented in `references/model_format.md`.
- `scripts/run_model.py --model models/tm1_paper.json --out results/tm1` runs nuclear
  properties, beta EOS, crust-core matching, and the M-R curve.
- `scripts/bja_score.py --list-constraints` lists the catalogue; use `--model`,
  `--constraints`, and optional `--out` to score a model. The Mmax >= 2.08 Msun term is a
  one-sided lower-bound penalty with sigma 0.07 Msun. `m_sigma = 600 +/- 200 MeV` is a
  parameter prior, not an observable constraint.
- `scripts/fit_model.py` optimizes explicitly selected couplings with user-supplied bounds.
  Run `--help` for required options; fitting is not run implicitly by the scoring command.
- `scripts/run_paper_model.py` extracts a model from a local PDF or arXiv paper and compares
  the computed outputs with reported values. It requires the optional extraction dependencies
  and `ANTHROPIC_API_KEY` in the environment. It makes a paid Anthropic API call only when the
  user explicitly runs extraction; tests use a mocked client.

`paper/validation_walecka/run.sh` reproduces the Table 2 examples.
`paper/sigma6_tm1/fit_run.py --evaluate-only` recomputes the Table 3 Gaussian results
at the released sigma6 and sigma6+c4 couplings. `paper/referee_validation/README.md`
documents the Appendix D joint NICER fits and posterior sampling. `harness/gt/` contains
the M-R target regeneration workflow. See `docs/REPRODUCTION.md` for the released values.

## Neutron-star M–R: REQUIRED crust handling

The TOV/MR stage needs a complete EOS from the stellar surface to the centre, so you MUST attach a crust below the RMF core. Do NOT improvise a polytropic crust — an invented low-density EOS produces unphysical radii (the TOV integrator then runs out to its `r < 100 km` guard and returns R ≈ 100 km).

Procedure:

1. Build the core EOS table on a β-equilibrium density grid with `beta_eos.build_beta_equilibrium_eos_table` (cover from below 0.08 fm^-3 up to the central density).
2. Attach the bundled BPS crust. A BPS table ships with this skill at `scripts/agentV1-tools/TOVtools/bps.dat`; load it with `hybrid_eos.read_bps_table(path)`.
3. Join crust and core with a FIXED-DENSITY split at `rho_b = 0.08 fm^-3`: BPS crust below 0.08, RMF core at/above 0.08. Use `hybrid_eos.build_hybrid_eos_from_tables(bps, core, rho_min_core=0.08)`, or the one-call convenience `hybrid_eos.build_hybrid_eos_with_bundled_crust(core, rho_min_core=0.08)`. Both return a monotonic `merged_table`.
4. Feed `merged_table` to `tov_solver.build_eos_interpolator` (or `EOSInterpolator`), then `integrate_tov_star(eos, central_pressure_mev_fm3, dr_km=0.005, p_surface_mev_fm3=1e-12)`.

Conventions and sanity checks:

- Crust–core transition is at `rho_b = 0.08 fm^-3` for this toolkit/benchmark.
- Units are handled inside the solver (MeV/fm^3 → km^-2 via `1.3234e-6`; mass in M_sun via `1.4766 km`). Pass pressure and energy density in MeV/fm^3.
- A normal neutron star has R ≈ 10–15 km. If R ≳ 30 km, or the integrator stops at r ≈ 100 km, the low-density EOS/crust is wrong — fix the crust; never raise the `r` guard to mask it.


## Reference files

Read `references/rmf-input-patterns.md` when you need a checklist for what information a user must provide when the Lagrangian changes, or when you need guidance on documenting baseline-vs-variant differences.
