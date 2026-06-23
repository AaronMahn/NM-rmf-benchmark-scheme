# Setup & operating notes (final parallel pipeline)

How the final benchmark was run. **No API keys are stored in this folder.** Host identifiers are
generic; an earlier (2026-06-17) snapshot of the initial setup is archived under
a separate archive outside this repository.

## Host

- An agent host running the `openclaw` agent framework (`openclaw` CLI on `PATH`).
- Benchmark root `P26_ROOT` (default `~/P26exp`): holds `tasks/`, per-agent workspaces `ws/<id>/`,
  and `results/cells/<agent>/`. Agent config + session stores live under `~/.openclaw/agents/<id>/`.
- Runs are kept alive across disconnects with `screen` + `caffeinate`. The repo's `data/` and
  `skill/` are the canonical task prompts, ground truth, and skill.

## Model lineup (final run: 6 models)

| key | binding | provider | base URL |
|-----|---------|----------|----------|
| gpt55 | `openrouter/openai/gpt-5.5` | OpenRouter | `https://openrouter.ai/api/v1` |
| claude | `openrouter/anthropic/claude-opus-4.8` | OpenRouter | `https://openrouter.ai/api/v1` |
| gemini | `google/gemini-3.1-pro-preview` | Google (native) | `https://generativelanguage.googleapis.com` |
| qwen | `dashscope/qwen3.7-max` | DashScope (native) | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| deepseek | `deepseek/deepseek-v4-pro` | DeepSeek (native) | `https://api.deepseek.com/v1` |
| kimi | `moonshot/kimi-k2.6` | Moonshot (native) | `https://api.moonshot.cn/v1` |

`gpt-5.4` (`openrouter/openai/gpt-5.4`) was tested earlier but **excluded** from the final
parallel run. Each provider's key lives in the agent's `~/.openclaw/agents/<id>/agent/auth-profiles.json`
(and inline in `models.json`); the model binding in `openclaw.json` is the source of truth — the
runner passes no `--model` flag.

## Agent layout (per-sample clones for true parallelism)

Base agents `<model>_w` / `<model>_wo` (workspace `ws/<model>_w` contains `skills/rmf-toolkit/`;
`ws/<model>_wo` has no skill). For the final paired/parallel run, `harness/clone_agents.py`
clones each into 6 per-sample agents `<model>_w0..2` and `<model>_wo0..2`, each with its own
workspace copy, so the 3 samples in a condition can run simultaneously without collisions.

## Run mode

- Agents run in **embedded** mode (`openclaw agent --local`, via `harness/run_cell_local.py`) so 6
  concurrent agents don't overload a single gateway. Token usage is recovered from the session
  `.jsonl` (stdout carries no usage in `--local`).
- Concurrency: **6-way** (3 samples × 2 conditions at once) for the OpenRouter/Google/DashScope
  models; **2-way paired** (`harness/prun2.py`) for the native deepseek/kimi keys, which throttle
  under 6 simultaneous requests.

## Hygiene — before any run

1. `openclaw hooks disable session-memory`.
2. Per sample, a fresh session id (`<agent>_<runtag>_s<N>`); clear stale `sessions/*.lock`.
3. The task message appends the no-scan instruction (couplings are random fixed inputs; compute
   forward; never scan/fit to empirical values). Grep `严禁扫描` to confirm.
4. `cell_dir` / `ws_dir` are absolute; the agent is told the absolute output `.npy` path.

## Gotchas

- **Native APIs throttle under high concurrency** (deepseek, kimi) — drop to 2-way and/or raise the
  per-task timeout if runs hit the 600 s cap.
- **One gateway only** — multiple gateways cause a split-brain that wedges concurrent agents;
  embedded (`--local`) mode sidesteps this.
