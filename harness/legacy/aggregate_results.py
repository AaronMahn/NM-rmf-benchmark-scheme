#!/usr/bin/env python3
"""
Aggregate per-model benchmark results into a combined table + scorecard.

Layout (local project folder):
  benchmark_scheme/results/
    <model>/results.json    <- one folder per model, named by model (deepseek, gpt55, ...)
    all_results.csv         <- (generated) every (model,condition,sample) row
    SUMMARY.md              <- (generated) per (model,condition) rollup

Run this after saving each new model's results under results/<model>/.
No official scoring metric is applied — it just reports raw relative errors,
time and tokens so a metric can be designed later.
"""
import os, json, glob, csv, statistics as st

BASE    = os.path.dirname(os.path.abspath(__file__))          # harness/
RESULTS = os.path.normpath(os.path.join(BASE, "..", "results"))
COLS    = ["binding_energy", "pressure", "compressibility", "symmetry_energy", "M_ns", "R_ns"]


def relerrs(pred, gt):
    if not pred:
        return None
    return [abs(pred[i] - gt[i]) / max(abs(gt[i]), 1e-9) * 100 for i in range(6)]


def main():
    rows = []
    for mdir in sorted(glob.glob(os.path.join(RESULTS, "*"))):
        rj = os.path.join(mdir, "results.json")          # a per-model folder = has results.json
        if os.path.isdir(mdir) and os.path.isfile(rj):
            rows += json.load(open(rj)).get("rows", [])

    # ---- all_results.csv ----
    with open(os.path.join(RESULTS, "all_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "condition", "sample", "valid", "exit", "secs", "tokens",
                    "median_relerr_pct", "max_relerr_pct"]
                   + [f"err_{c}_pct" for c in COLS]
                   + [f"pred_{c}" for c in COLS] + [f"gt_{c}" for c in COLS])
        for r in rows:
            e = relerrs(r.get("pred"), r.get("gt"))
            med = round(st.median(e), 4) if e else ""
            mx = round(max(e), 4) if e else ""
            errc = [round(x, 4) for x in e] if e else [""] * 6
            pred = r.get("pred") or [""] * 6
            w.writerow([r["model"], r["condition"], r["sample"], r.get("valid"), r.get("exit"),
                        r.get("secs"), r.get("tokens"), med, mx] + errc + list(pred) + list(r["gt"]))

    # ---- SUMMARY.md ----
    def agg(cr):
        secs = [r["secs"] for r in cr if r.get("secs") is not None]
        toks = [r["tokens"] for r in cr if r.get("tokens") is not None]
        meds = []
        for r in cr:
            e = relerrs(r.get("pred"), r.get("gt"))
            if r.get("valid") and e is not None:
                meds.append(st.median(e))
        return {
            "n": len(cr),
            "n_valid": len([r for r in cr if r.get("valid")]),
            "med": round(st.median(meds), 3) if meds else None,
            "secs": round(sum(secs) / len(secs)) if secs else None,
            "toks": round(sum(toks) / len(toks)) if toks else None,
        }

    models = sorted(set(r["model"] for r in rows))
    L = ["# Benchmark scorecard (raw — no official metric applied yet)", "",
         "Per-model, per-condition rollup over each model's 3 random samples.",
         "`median_err%` = median over samples of the per-sample median per-target relative error. "
         "`mean_secs` / `mean_tokens` are means over the samples (valid + failed).", "",
         "| model | condition | valid/total | median_err% | mean_secs | mean_tokens |",
         "|---|---|---|---|---|---|"]
    for m in models:
        for cond in ["with_skill", "without_skill"]:
            cr = [r for r in rows if r["model"] == m and r["condition"] == cond]
            if not cr:
                continue
            a = agg(cr)
            L.append(f"| {m} | {cond} | {a['n_valid']}/{a['n']} | "
                     f"{a['med'] if a['med'] is not None else '—'} | "
                     f"{a['secs'] if a['secs'] is not None else '—'} | "
                     f"{a['toks'] if a['toks'] is not None else '—'} |")
    L += ["", f"Models present: {', '.join(models) if models else '(none)'}. "
              f"Per-sample detail in `all_results.csv`; raw arrays in `results/<model>/npy/`."]
    open(os.path.join(RESULTS, "SUMMARY.md"), "w").write("\n".join(L) + "\n")
    print("aggregated models:", models, "| total rows:", len(rows))


if __name__ == "__main__":
    main()
