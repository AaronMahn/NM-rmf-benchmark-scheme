import sys, os, json, glob
from pathlib import Path
import numpy as np
# ROOT = host benchmark root (cells, session stores, output bundles live here)
ROOT=os.environ.get("P26_ROOT",os.path.expanduser("~/P26exp")); HOME=os.path.expanduser("~")
# REPO = this repository (the canonical ground truth is a repo artifact, not host runtime)
REPO=Path(os.environ.get("P26_REPO", Path(__file__).resolve().parent.parent))
GT=np.load(str(REPO/"data"/"gt"/"targets_100x6.npy"))
MAN=f"{ROOT}/results/parallel_manifest.json"
COLS=["binding_energy_MeV","pressure_MeV_fm3","compressibility_MeV","symmetry_energy_MeV","M_msun","R_km"]
man=json.load(open(MAN))
models=sys.argv[1].split(",") if len(sys.argv)>1 else list(man)

def recover_tokens(agent,sid):
    jf=glob.glob(f"{HOME}/.openclaw/agents/{agent}/sessions/{sid}.jsonl")
    if not jf: return None
    tot=0
    def walk(x):
        if isinstance(x,dict):
            if 'totalTokens' in x or ('input' in x and 'output' in x): yield x
            for v in x.values(): yield from walk(v)
        elif isinstance(x,list):
            for v in x: yield from walk(v)
    for ln in open(jf[0]):
        try: o=json.loads(ln)
        except: continue
        for u in walk(o): tot += (u.get('input',0) or 0)+(u.get('output',0) or 0)
    return int(tot) or None

for model in models:
    info=man[model]; samples=info["samples"]
    rows=[]
    for cond_key,agents in info["agents"].items():   # with_skill / without_skill
        for i,agent in enumerate(agents):
            s=samples[i]; cell=f"{ROOT}/results/cells/{agent}"
            mp=f"{cell}/{s}.meta.json"; npp=f"{cell}/{s}.npy"
            rec={"model":model,"condition":cond_key,"agent":agent,"sample":s,
                 "valid":False,"secs":None,"tokens":None,"exit":None,"pred":None,"gt":GT[s].tolist()}
            if os.path.exists(mp):
                m=json.load(open(mp))
                rec["secs"]=m.get("secs"); rec["exit"]=m.get("exit")
                rec["tokens"]=(m.get("usage") or {}).get("total") or recover_tokens(agent,m.get("sid",""))
            if os.path.exists(npp):
                try:
                    a=np.load(npp).astype(float).squeeze()
                    if a.shape==(6,) and np.all(np.isfinite(a)):
                        rec["pred"]=a.tolist(); rec["valid"]=True
                except Exception: pass
            rows.append(rec)
    out=f"{ROOT}/results/par_bundle_{model}"
    os.makedirs(out,exist_ok=True)
    json.dump({"manifest":{model:info},"rows":rows}, open(f"{out}/results.json","w"), indent=2)
    # quick scorecard
    print(f"\n=== {model} (parallel) ===")
    for cond in ("with_skill","without_skill"):
        cr=[r for r in rows if r["condition"]==cond]
        for r in cr:
            if r["valid"]:
                e=[abs(r["pred"][k]-r["gt"][k])/max(abs(r["gt"][k]),1e-9)*100 for k in range(6)]
                print(f"  {cond[:4]:4} s{r['sample']}: maxErr={max(e):.2f}% matter={max(e[:4]):.2f}% M={e[4]:.2f}% R={e[5]:.2f}% secs={r['secs']} tok={r['tokens']}")
            else:
                print(f"  {cond[:4]:4} s{r['sample']}: INVALID exit={r['exit']} secs={r['secs']}")
