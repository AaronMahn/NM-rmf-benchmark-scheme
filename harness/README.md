# Harness

Scripts to run the benchmark and score the results. (The one-time ground-truth recalibration
that produced `data/gt/targets_100x6.npy` is kept in a separate archive outside this
repository — it isn't needed to run or score the benchmark.)

Most of these scripts orchestrate agents and therefore assume an `openclaw` deployment plus a
benchmark root referenced by `P26_ROOT`. The scorer (`aggregate_parallel.py`) is self-contained
and runs anywhere from a clone.

## Pipeline (paired, parallel) — per model

1. **`clone_agents.py`** — clones each model's `_w`/`_wo` agent into 6 per-sample agents
   (`<model>_w0..2`, `<model>_wo0..2`), each with its own workspace copy (skill present for
   `_w*`, absent for `_wo*`). This lets the 3 samples in a condition run in parallel without
   filesystem/session collisions.

2. **`prun.py <model>`** — launches all **6 runs simultaneously** (3 samples × {with, without}
   skill) for one model, paired on the same samples. Records the drawn samples in
   `results/parallel_manifest.json`.

   **`prun2.py <model>`** — same, but **paired 2-way**: each sample's `_w`+`_wo` run together,
   sample-pairs sequential. Caps concurrent requests at 2 when lower concurrency is needed for
   slower providers.

   Both call `run_cell_local.py` for each cell.

3. **`run_cell_local.py`** — runs one agent on one sample in local embedded mode. Writes
   `results/cells/<agent>/<N>.npy` (the 6-vector prediction) and `<N>.meta.json` (validity, wall
   time, token usage, exit code). Validity = a finite `(6,)` array was saved.

4. **`prun_collect.py <model>`** — gathers each model's 6 cloned-agent results into
   `results/par_bundle_<model>/results.json` (prediction, GT, validity, secs, tokens) and prints
   a per-sample scorecard using the repo ground truth (`data/gt/targets_100x6.npy`).

5. **`aggregate_parallel.py`** — scores `results/` against the ground truth and writes
   `results/SUMMARY_parallel.md`. Self-contained (repo-relative); run from anywhere:
   `python3 harness/aggregate_parallel.py`.

## `legacy/`

The earlier **sequential / gateway-mode** harness, superseded by the parallel pipeline above but
kept for provenance: `run_cell.py`, `scheme_run.py`, `scheme_collect.py`, `scheme_teardown.sh`,
`orchestrate.py`, `score.py`, `bench_index.py`, `aggregate_results.py`.
