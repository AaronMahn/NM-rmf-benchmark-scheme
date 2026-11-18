#!/usr/bin/env python3
"""
Side-by-side re-scoring: original GT vs recalibrated M-R GT.

The recalibration ONLY changes the M_ns / R_ns columns (4,5) to the skill's
bundled-pipeline values; matter columns (0-3) are bit-identical. Samples 22/76
use a skip-and-continue solver to recover the final central-pressure point.

The ORIGINAL scoring (SUMMARY.md / all_results.csv) is left untouched and remains
primary. This script writes SUMMARY_recal.md only, comparing per-model M/R errors
under the two ground-truth versions.
"""
import json
import statistics as st
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = REPO / "results"
ORIG_GT = np.load(HERE / "targets_100x6.original.npy")
RECAL_GT = np.load(HERE / "targets_100x6.recal.npy")


def relerr(pred, gt):
    return abs(pred - gt) / max(abs(gt), 1e-9) * 100


def main():
    rows = []
    for result_path in sorted(RESULTS.glob("*/results.json")):
        rows += json.loads(result_path.read_text(encoding="utf-8")).get("rows", [])

    models = sorted(set(r["model"] for r in rows))
    L = ["# M-R recalibration: original vs recalibrated scoring", "",
         "Recalibrated GT replaces M_ns/R_ns with the **skill's bundled-pipeline** values "
         "(all 100 samples; samples 22 & 76 use skip-and-continue beta-EOS solving). "
         "Matter quantities are unchanged. The original `SUMMARY.md` remains the primary scorecard.", "",
         "Per model/condition: median over the model's **valid** samples of the M and R relative "
         "error (%), under the original GT vs the recalibrated GT.", "",
         "| model | condition | n_valid | M err orig→recal | R err orig→recal |",
         "|---|---|---|---|---|"]

    def med_errs(cr):
        mO, mR, rO, rR = [], [], [], []
        for r in cr:
            if not r.get("valid") or not r.get("pred"):
                continue
            s = r["sample"]; pred = r["pred"]
            mO.append(relerr(pred[4], ORIG_GT[s, 4]));  mR.append(relerr(pred[4], RECAL_GT[s, 4]))
            rO.append(relerr(pred[5], ORIG_GT[s, 5]));  rR.append(relerr(pred[5], RECAL_GT[s, 5]))
        f = lambda v: round(st.median(v), 2) if v else None
        return len(mO), f(mO), f(mR), f(rO), f(rR)

    for m in models:
        for cond in ["with_skill", "without_skill"]:
            cr = [r for r in rows if r["model"] == m and r["condition"] == cond]
            if not cr:
                continue
            n, moO, moR, roO, roR = med_errs(cr)
            if n == 0:
                L.append(f"| {m} | {cond} | 0 | — | — |")
            else:
                L.append(f"| {m} | {cond} | {n} | {moO}% → {moR}% | {roO}% → {roR}% |")

    L += ["", "Reading: for **with-skill** rows the recalibrated columns should drop toward ~0 "
              "(GT now equals the skill pipeline, so a faithful skill-follower matches it). "
              "**without-skill** rows reflect each model's own crust/EOS choices and need not improve."]
    out = HERE / "SUMMARY_recal.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", out)
    print("\n".join(L))


if __name__ == "__main__":
    main()
