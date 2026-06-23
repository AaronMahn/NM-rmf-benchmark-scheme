"""
Score the parallel run in results/ against the recalibrated ground truth and write
results/SUMMARY_parallel.md. Reads results/par_all.json (produced by prun_collect.py).

Repo-relative; run from anywhere in a clone:  python3 harness/aggregate_parallel.py
"""
import os, json, statistics as st
from pathlib import Path

REPO    = Path(os.environ.get("P26_REPO", Path(__file__).resolve().parent.parent))
RESULTS = REPO/"results"
ORDER   = ["gpt55", "claude", "gemini", "qwen", "deepseek", "kimi"]
# concurrency used per model in the latest run (documented in the scorecard)
CONC    = {"gpt55":"6-way","claude":"6-way","gemini":"6-way","qwen":"6-way","deepseek":"2-way","kimi":"2-way"}


def relerr(pred, gt):
    return [abs(pred[i]-gt[i])/max(abs(gt[i]), 1e-9)*100 for i in range(6)]


def main():
    rows = json.load(open(RESULTS/"par_all.json"))["rows"]
    models = [m for m in ORDER if any(r["model"] == m for r in rows)]
    L = ["# Parallel run scorecard (vs recalibrated GT)", "",
         "All 3 samples per condition + both conditions run simultaneously (paired on the same "
         "samples). `conc` = concurrency used. Errors are medians over the model's **valid** samples.", "",
         "| model | conc | cond | valid/3 | matter med% | M med% | R med% | mean s |",
         "|---|---|---|---|---|---|---|---|"]
    for m in models:
        for cond in ("with_skill", "without_skill"):
            cr = [r for r in rows if r["model"] == m and r["condition"] == cond]
            val = [r for r in cr if r.get("valid") and r.get("pred")]
            secs = [r["secs"] for r in cr if r.get("secs") is not None]
            ms = round(sum(secs)/len(secs)) if secs else "—"
            if val:
                mat = [max(relerr(r["pred"], r["gt"])[:4]) for r in val]
                mm  = [relerr(r["pred"], r["gt"])[4] for r in val]
                rr  = [relerr(r["pred"], r["gt"])[5] for r in val]
                L.append(f"| {m} | {CONC.get(m,'?')} | {cond} | {len(val)}/3 | "
                         f"{st.median(mat):.2f} | {st.median(mm):.2f} | {st.median(rr):.2f} | {ms} |")
            else:
                L.append(f"| {m} | {CONC.get(m,'?')} | {cond} | 0/3 | — | — | — | {ms} |")
    L += ["", "Columns: `matter med%` = median (over valid samples) of the worst of the four matter "
              "quantities; `M med%`/`R med%` = median M and R relative error. Per-sample detail in "
              "`par_all.json`; per-model bundles in `par_bundle_<model>/`."]
    (RESULTS/"SUMMARY_parallel.md").write_text("\n".join(L) + "\n")
    print("wrote", RESULTS/"SUMMARY_parallel.md")
    print("\n".join(L))


if __name__ == "__main__":
    main()
