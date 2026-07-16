# RMF Skill Benchmark

A benchmark measuring how much a domain **skill** (a bundled relativistic mean-field
toolkit, `rmf-toolkit`) helps large language models compute nuclear-matter and
neutron-star observables from a relativistic mean-field (RMF) Lagrangian.

Each model is run **with** the skill and **without** it, on the same tasks, and scored
against a fixed numerical ground truth. The question: does giving the model the toolkit
change whether it gets the physics right?

> Research code released alongside the manuscript. The benchmark harness is agent-driven and
> depends on an `openclaw` deployment plus provider credentials; some scripts require local
> adaptation to reproduce. See
> [`config/SETUP_NOTES.md`](config/SETUP_NOTES.md) and [`harness/README.md`](harness/README.md).

## The task

Each of the 100 tasks (`data/tasks/0.txt … 99.txt`) gives an RMF Lagrangian and a set of
**randomly sampled, fixed** couplings, and asks for 6 numbers:

| # | target | where |
|---|--------|-------|
| 0 | binding energy `E/A` (MeV) | symmetric matter at ρ = 0.16 fm⁻³ |
| 1 | pressure `P` (MeV/fm³) | ρ = 0.16 fm⁻³ |
| 2 | incompressibility `K` (MeV) | ρ = 0.16 fm⁻³ |
| 3 | symmetry energy `E_sym` (MeV) | ρ = 0.16 fm⁻³ |
| 4 | neutron-star mass `M` (M☉) | central density ρ_c = 4.0 fm⁻³, β-equilibrium, BPS crust joined at 0.08 fm⁻³ |
| 5 | neutron-star radius `R` (km) | same configuration |

The couplings are random fixed inputs, **not** fitted to empirical values, so the usual
nuclear-physics regularities (E/A ≈ −16 MeV, K ≈ 240 MeV, …) do **not** apply — the task
must be solved by direct forward computation. The task prompt explicitly forbids
scanning / fitting to empirical numbers.

## The two conditions

- **with skill (`_w`)** — the agent's workspace contains `skills/rmf-toolkit/` (this repo's
  `skill/`): `SKILL.md` guidance plus a real ~1150-line Python toolkit
  (`analytic_layer`, `numeric_layer`, `nuclear_properties`, `beta_eos`, `hybrid_eos`,
  `tov_solver`, `tool_beta_eff`) and the bundled `bps.dat` crust table.
- **without skill (`_wo`)** — identical task, no toolkit; the model must reconstruct
  everything from its own knowledge.

## Ground truth

`data/gt/`:

- `targets_100x6.npy` — the ground truth, shape `(100, 6)`, column order as above.
- `targets_100x6_columns.txt` — column labels.

The neutron-star **M and R** columns are recalibrated to the skill's bundled toolkit pipeline,
so that a faithful skill-follower scores ~0 on M–R; the four matter columns are the original
forward-computed values. The reason: the toolkit's β-EOS solver was revised after the first
ground truth was generated — the crust table (`bps.dat`) is byte-identical, but the solver
revision shifts M and R by a median ~1 % (tail to ~13 %) on the same EOS.

Scoring against the skill's own pipeline measures *adherence*; it is not an independent
correctness check (see the manuscript). The pre-recalibration ground truth, the scripts that
produced the recalibration, and the per-sample shift report are kept in a separate archive
outside this repository.

## Repository layout

```
benchmark_scheme/
├── README.md                  ← this file
├── LICENSE
├── requirements.txt
├── skill/rmf-toolkit/         ← the skill under test (SKILL.md + scripts/ + bps.dat)
├── harness/                   ← runners, collector, scorers (see harness/README.md)
│   └── legacy/                ← earlier sequential/gateway-era harness (superseded)
├── data/
│   ├── tasks/0.txt … 99.txt   ← the 100 task prompts
│   └── gt/                    ← ground truth (targets_100x6.npy + columns)
├── config/                    ← setup notes for the published run configuration
└── results/                   ← latest run (see below)
```

Superseded material (earlier result sets, the recalibration process, the skill-development
history, and the initial config snapshot) is kept in a separate archive outside this
repository, not in the published tree.

## Results in `results/`

`results/` holds the **most recent run**: a paired, simultaneous benchmark of 6 models
(gpt-5.5, claude-opus-4.8, gemini-3.1-pro, qwen3.7-max, deepseek-v4-pro, kimi-k2.6 — gpt-5.4
excluded), 3 random samples each, both conditions run at the same time.

- `SUMMARY_parallel.md` — per-model / per-condition scorecard vs the recalibrated GT.
- `par_all.json` — every (model, condition, sample) row with prediction, GT, validity, time, tokens.
- `par_bundle_<model>/results.json` — per-model bundles.
- `parallel_manifest.json` — the samples drawn per model.

Earlier result sets (the sequential single- and two-set-per-model runs, and their scored
summaries) are kept in the separate archive outside this repository.

### Headline finding

With the skill, the four matter quantities come out **bit-exact** for every model that
engages the toolkit, and M–R collapses to ~0 against the recalibrated GT. Without the skill,
behaviour ranges from correct (gpt-5.5) through high-variance (claude, gemini) to broken
(qwen, deepseek, kimi) — with characteristic, identifiable failure modes, most notably a
**negative incompressibility** from the saturation-only `K` formula that the skill explicitly
warns against. The clearest demonstration is paired and simultaneous: on the same sample at
the same moment, the with-skill run is exact while the without-skill run produces `K < 0`.

Caveats for the latest run are noted in `SUMMARY_parallel.md` (deepseek/kimi native APIs were
slow/rate-limited the day it ran).

## Reproducing

This is agent-driven and depends on the `openclaw` agent framework, provider credentials, and a
machine to host the agents; it is not a one-command pipeline. The outline:

1. Stage `skill/rmf-toolkit/` into each `_w` agent workspace; leave `_wo` workspaces without it.
2. `harness/clone_agents.py` — create per-sample agent clones for true parallelism.
3. `harness/prun.py <model>` — launch the 6 simultaneous runs (3 samples × 2 conditions);
   `harness/prun2.py <model>` for paired 2-way (lower concurrency for rate-limited providers).
   Both call `harness/run_cell_local.py` per cell.
4. `harness/prun_collect.py <model>` — collect predictions + tokens, score against GT.
5. `harness/aggregate_parallel.py` — build `results/SUMMARY_parallel.md`.

Set `P26_ROOT` to point the agent-orchestration scripts at your benchmark root. See
`harness/README.md` for each script and `config/SETUP_NOTES.md` for environment details.
