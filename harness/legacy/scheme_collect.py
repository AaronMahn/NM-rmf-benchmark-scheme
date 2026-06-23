#!/usr/bin/env python3
"""
Collect P26 scheme results into a self-contained bundle. NO SCORING.
(The scoring metric will be designed later; this just gathers raw predictions,
ground truth, validity, time and token usage so any metric can be computed.)

Reads:  results/scheme_manifest.json, results/cells/<cell>/N.{npy,meta.json}, gt/targets_100x6.npy
Writes: results/scheme_bundle/{manifest.json, results.json, results.csv, npy/<agent>_<N>.npy}

Run on the remote after scheme_run.py finishes:
    python3 scheme_collect.py            # bundle ALL models in the manifest
    python3 scheme_collect.py deepseek   # bundle ONLY deepseek (recommended: one model per bundle)
Then pull results/scheme_bundle/ to the local project folder's by_model/<model>/.
"""
import os, sys, json, csv, shutil
import numpy as np

ROOT = os.path.expanduser("~/P26exp")
MAN  = f"{ROOT}/results/scheme_manifest.json"
GT   = np.load(f"{ROOT}/gt/targets_100x6.npy")
COLS = ["binding_energy_MeV", "pressure_MeV_fm3", "compressibility_MeV",
        "symmetry_energy_MeV", "M_msun", "R_km"]
OUT  = f"{ROOT}/results/scheme_bundle"


def main():
    man = json.load(open(MAN))
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    if only:
        man = {k: v for k, v in man.items() if k in only}
        print("collecting only:", list(man))
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(f"{OUT}/npy", exist_ok=True)
    rows = []
    for model, info in man.items():
        for cond_key, agent in info["agents"].items():       # with_skill / without_skill
            for n in info["samples"]:
                cell = f"{ROOT}/results/cells/{agent}"
                mp, npp = f"{cell}/{n}.meta.json", f"{cell}/{n}.npy"
                rec = {"model": model, "condition": cond_key, "agent": agent, "sample": n,
                       "valid": None, "secs": None, "tokens": None,
                       "model_used": None, "exit": None, "pred": None, "gt": GT[n].tolist()}
                if os.path.exists(mp):
                    m = json.load(open(mp))
                    rec.update(valid=m.get("valid"), secs=m.get("secs"),
                               tokens=(m.get("usage") or {}).get("total"),
                               model_used=m.get("model_used"), exit=m.get("exit"),
                               pred=m.get("prediction"))
                if os.path.exists(npp):
                    try:
                        a = np.load(npp).astype(float).squeeze()
                        if a.shape == (6,):
                            rec["pred"] = a.tolist()
                            shutil.copy(npp, f"{OUT}/npy/{agent}_{n}.npy")
                    except Exception:
                        pass
                rows.append(rec)
    json.dump({"manifest": man, "rows": rows}, open(f"{OUT}/results.json", "w"), indent=2)
    with open(f"{OUT}/results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "condition", "sample", "valid", "secs", "tokens", "model_used", "exit"]
                   + [f"pred_{c}" for c in COLS] + [f"gt_{c}" for c in COLS])
        for r in rows:
            pred = r["pred"] if r["pred"] else [None] * 6
            w.writerow([r["model"], r["condition"], r["sample"], r["valid"], r["secs"],
                        r["tokens"], r["model_used"], r["exit"]] + list(pred) + list(r["gt"]))
    shutil.copy(MAN, f"{OUT}/manifest.json")
    print("bundle written:", OUT, "| rows:", len(rows))


if __name__ == "__main__":
    main()
