# Setup notes (published summary)

High-level notes on the configuration used for the final paired/parallel benchmark. **No API keys
are stored in this folder.** Detailed host-specific setup snapshots remain outside this repository.

## Environment

- An `openclaw` deployment is required to run the benchmark harness.
- `P26_ROOT` should point at the benchmark working tree used by the orchestration scripts.
- The repository's `data/` and `skill/` directories are the canonical task prompts, ground truth,
  and skill contents used by the published run.

## Model lineup (final run: 6 models)

| key | binding | provider |
|-----|---------|----------|
| gpt55 | `openrouter/openai/gpt-5.5` | OpenRouter |
| claude | `openrouter/anthropic/claude-opus-4.8` | OpenRouter |
| gemini | `google/gemini-3.1-pro-preview` | Google |
| qwen | `dashscope/qwen3.7-max` | DashScope |
| deepseek | `deepseek/deepseek-v4-pro` | DeepSeek |
| kimi | `moonshot/kimi-k2.6` | Moonshot |

`gpt-5.4` was tested earlier but excluded from the final parallel run.

## Agent layout

- Base agents are organized as `<model>_w` and `<model>_wo` for the with-skill and without-skill
  conditions.
- `harness/clone_agents.py` expands each base agent into per-sample clones so paired samples can
  run concurrently without workspace collisions.

## Run mode

- The published run used paired execution: 3 samples for the with-skill condition and the matching
  3 samples for the without-skill condition.
- `harness/prun.py` was used for the general 6-way parallel case, while `harness/prun2.py` was
  used when lower concurrency was needed for provider stability.
- `harness/run_cell_local.py` runs one agent on one sample and records the prediction vector plus
  per-run metadata.

## Reproduction notes

- Keep the task prompt's no-scan instruction intact: couplings are fixed random inputs and must be
  handled by direct forward computation.
- If reproducing on a different host, adapt local orchestration details to your own deployment.
