# Archive import and sanitisation notes

Source: read-only `agentV1` archive. Archived tools are not copied over the released
toolkit. The notes below record the intended disposition of every archived `.py` and
`.json` file under `agentV1/tools`, `agentV1/scripts`, `agentV1/input`, and
`agentV1/skills`.

## `agentV1/tools/`

| Archived file | Disposition |
|---|---|
| `NMtoolsV1/analytic_layer.py` | Replaced by the newer released version; do not overwrite. |
| `NMtoolsV1/nuclear_properties.py` | Replaced by the newer released version; do not overwrite. |
| `NMtoolsV1/numeric_layer.py` | Replaced by the identical released version; do not overwrite. |
| `TOVtools/beta_eos.py` | Replaced by the newer released version; do not overwrite. |
| `TOVtools/hybrid_eos.py` | Replaced by the newer released version; do not overwrite. |
| `TOVtools/tool_beta_eff.py` | Replaced by the newer released version; do not overwrite. |
| `TOVtools/tov_solver.py` | Replaced by the identical released version; do not overwrite. |
| `model_io.py` | Port to the released toolkit as the JSON model loader; use repo-relative paths and the released package layout. |
| `model_sanity.py` | Skip; the loader and required reproduction pipeline do not depend on this optional diagnostic. |
| `paper_extractor.py` | Port for T7; API key from the environment only, and test with a mocked client. |

## `agentV1/scripts/`

| Archived file | Disposition |
|---|---|
| `GQHD2/backfill_reference_models.py` | Skip as a GQHD2-specific backfill helper; no generic paper workflow depends on it. |
| `GQHD2/bayesian_score_tool.py` | Generalise into the constraint-based `bja_score.py`; no GQHD2-specific defaults. |
| `GQHD2/benchmark_beta_methods_10pts.py` | Skip as a GQHD2 exploratory comparison, outside the paper reproduction interface. |
| `GQHD2/build_three_result_files.py` | Skip as a one-off GQHD2 results exporter. |
| `GQHD2/check_mu_at_given_rhos.py` | Skip as a GQHD2 diagnostic. |
| `GQHD2/check_tool_beta_eff_symbols.py` | Skip as a GQHD2/tool-symbol diagnostic; released toolkit is authoritative. |
| `GQHD2/compare_beta_methods_rhob_0p2.py` | Skip as a GQHD2-specific method comparison. |
| `GQHD2/compare_mr_with_reference.py` | Skip as a GQHD2-specific comparison helper. |
| `GQHD2/debug_energy_compare_rhob_0p2.py` | Skip as a GQHD2 energy diagnostic. |
| `GQHD2/mr_bayesian_tool.py` | Do not port as a standalone GQHD2 tool; retain only a generic extension point for a joint M–R likelihood. |
| `GQHD2/run_agentv1_gqhd2_all.py` | Skip as a GQHD2 batch runner. |
| `GQHD2/run_gqhd2_legacy_beta_and_mr.py` | Skip as a GQHD2 legacy runner. |
| `GQHD2/run_no_a0.py` | Skip as a GQHD2-specific variant runner. |
| `GQHD2/scan_g_a0_mr.py` | Skip as a GQHD2 coupling scan. |
| `GQHD2/scan_g_a0_nmp.py` | Skip as a GQHD2 coupling scan. |
| `GQHD2/solve_beta_point_rhob_0p2.py` | Skip as a single-point GQHD2 diagnostic. |
| `GQHD2/summarize_reference_models.py` | Skip as a GQHD2-specific summary generator. |
| `fit/fit_model.py` | Generalise into the public `fit_model.py`; use the released toolkit and selectable constraints. |
| `run_model.py` | Port as the public pipeline driver; use the released toolkit and repo-relative paths. |
| `run_paper_model.py` | Port for T7 as the paper-extractor runner; do not make real API calls. |
| `scan/scan_coupling.py` | Skip as an exploratory coupling scan, not required for the specified reproduction. |
| `README.md` | Reference only; do not copy archive-specific instructions or paths. |

## `agentV1/input/`

| Archived file | Disposition |
|---|---|
| `model_definitions/gqhd2.json` | Port as a public model definition; validate with the released loader. |
| `model_definitions/tm1.json` | Port as a public model definition; add the required scalar-isovector bilinear symbol for the released compiler. |

## `agentV1/skills/`

| Archived file | Disposition |
|---|---|
| `rmf-fitting/SKILL.md` | Skip as a legacy agent prompt; public toolkit documentation will describe the supported workflow. |

## Sanitisation edits

- Added `harness/runtime_config.py` as the single source for the `OPENCLAW_HOME`
  environment variable and its per-user default. The runners, collector, clone helper,
  legacy runner, and teardown script now resolve agent configuration through that value.
- Updated `harness/README.md`, `config/SETUP_NOTES.md`, and
  `config/openclaw_config_snapshot.json` to use `$OPENCLAW_HOME` and instruct users to
  adapt it to their platform.
- Removed the fixed Homebrew executable path from active and legacy runner scripts; the
  `openclaw` CLI must be available on `PATH`, as documented.
- Added `config/example.env` for `OPENCLAW_HOME` and `ANTHROPIC_API_KEY`. `.env` was
  already ignored by the repository's `.gitignore` and remains ignored.
- The paper extractor will read `ANTHROPIC_API_KEY` from the process environment only.
- No personal agent state, private logs, or archived result transcripts were imported.
