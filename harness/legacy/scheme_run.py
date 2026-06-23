#!/usr/bin/env python3
"""
P26 RMF skill-benchmark SCHEME runner  (runs on the remote Mac, ~/P26exp/).

Scheme (per user spec):
  - For EACH requested model, randomly pick 3 INDEPENDENT samples from the 100 tasks.
  - Run BOTH conditions: <model>_w (with skill) and <model>_wo (without skill).
  - Time bound = 600 s (10 min) per task.
  - run_cell.py records per-sample wall time (secs) and token usage (usage.total)
    into results/cells/<cell>/N.meta.json  -> that is how time+tokens are saved.
  - The chosen samples per model are written to results/scheme_manifest.json
    (resume-safe: a model already in the manifest reuses its samples).

This script does NOT score. Scoring/collection is scheme_collect.py.

Usage (run inside a screen + caffeinate so it survives disconnect):
    python3 scheme_run.py deepseek            # one model
    python3 scheme_run.py deepseek,qwen       # several
    python3 scheme_run.py all                 # all 5

Prereqs: working API key for each requested model's provider (see SETUP_NOTES.md).
"""
import sys, os, json, subprocess, random, time, glob

ROOT     = os.path.expanduser("~/P26exp")
TIMEOUT  = "600"          # 10 minutes per task
NSAMPLES = 3
RUNTAG   = "scheme1"
MAXCONC  = 3              # max concurrent cells (3 was safe; 10 starved the Mac)
HOME     = os.path.expanduser("~")

MODELS = {
    "gpt55":    "openrouter/openai/gpt-5.5",
    "claude":   "openrouter/anthropic/claude-opus-4.8",
    "gemini":   "google/gemini-3.1-pro-preview",
    "qwen":     "dashscope/qwen3.7-max",
    "deepseek": "deepseek/deepseek-v4-pro",
    "kimi":     "moonshot/kimi-k2.6",
}

MANIFEST = f"{ROOT}/results/scheme_manifest.json"


def load_manifest():
    return json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}


def pick_samples(model, manifest):
    # resume-safe: reuse already-recorded samples; otherwise a fresh random draw
    if model in manifest and manifest[model].get("samples"):
        return manifest[model]["samples"]
    return sorted(random.sample(range(100), NSAMPLES))


def hygiene(agent):
    sd = f"{HOME}/.openclaw/agents/{agent}/sessions"
    for f in glob.glob(sd + "/*.jsonl") + glob.glob(sd + "/*.lock") + glob.glob(sd + "/sessions.json"):
        try:
            os.remove(f)
        except OSError:
            pass


def launch(agent, ids, model):
    cell = f"{ROOT}/results/cells/{agent}"
    ws   = f"{ROOT}/ws/{agent}"
    ids_csv = ",".join(str(i) for i in ids)
    log  = f"{ROOT}/results/cells/{agent}.{RUNTAG}log"
    hygiene(agent)
    env = dict(os.environ); env["PATH"] = "/opt/homebrew/bin:" + env.get("PATH", "")
    lf = open(log, "w")
    return subprocess.Popen(
        ["python3", f"{ROOT}/run_cell.py", agent, cell, f"{ROOT}/tasks",
         RUNTAG, ids_csv, ws, model, TIMEOUT],
        env=env, stdout=lf, stderr=subprocess.STDOUT)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    models = list(MODELS) if which == "all" else which.split(",")
    for m in models:
        assert m in MODELS, f"unknown model '{m}' (known: {list(MODELS)})"

    env = dict(os.environ); env["PATH"] = "/opt/homebrew/bin:" + env.get("PATH", "")
    # benchmark integrity: session-memory hook OFF
    subprocess.run(["openclaw", "hooks", "disable", "session-memory"], env=env)

    manifest = load_manifest()
    jobs = []
    for m in models:
        samples = pick_samples(m, manifest)
        manifest[m] = {
            "model": MODELS[m],
            "samples": samples,
            "timeout_s": int(TIMEOUT),
            "runtag": RUNTAG,
            "agents": {"with_skill": f"{m}_w", "without_skill": f"{m}_wo"},
        }
        for cond in ("w", "wo"):
            jobs.append((f"{m}_{cond}", samples, MODELS[m]))
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    json.dump(manifest, open(MANIFEST, "w"), indent=2)
    print("manifest written:", MANIFEST)
    for m in models:
        print(f"  {m}: samples {manifest[m]['samples']}")

    running = []
    for agent, ids, model in jobs:
        while len([p for p in running if p.poll() is None]) >= MAXCONC:
            time.sleep(5)
        print(f"launching {agent} samples={ids}", flush=True)
        running.append(launch(agent, ids, model))
    for p in running:
        p.wait()
    open(f"{ROOT}/results/scheme_{RUNTAG}_DONE", "w").write("done " + time.strftime("%F %T") + "\n")
    print("ALL DONE")


if __name__ == "__main__":
    main()
