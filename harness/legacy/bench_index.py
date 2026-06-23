import os
#!/usr/bin/env python3
"""Benchmark index: per-cell validity, accuracy, TOKENS used, and TIME consumed.
Usage: python3 bench_index.py [gt_npy] [results/cells dir]"""
import sys, os, json
import numpy as np
COLS=['binding_energy_MeV','pressure_MeV_fm3','compressibility_MeV','symmetry_energy_MeV','M_msun','R_km']
gt_path=sys.argv[1] if len(sys.argv)>1 else os.path.expanduser('~/P26exp/gt/targets_100x6.npy')
cells_dir=sys.argv[2] if len(sys.argv)>2 else os.path.expanduser('~/P26exp/results/cells')
gt=np.load(gt_path)
rows=[]
for cell in sorted(os.listdir(cells_dir)):
    d=os.path.join(cells_dir,cell)
    if not os.path.isdir(d): continue
    metas=sorted(f for f in os.listdir(d) if f.endswith('.meta.json'))
    if not metas: continue
    n_att=n_valid=0; toks=0; secs=0.0; errs=[]; model=''
    for mf in metas:
        m=json.load(open(os.path.join(d,mf)))
        n_att+=1
        model=m.get('model_used') or model
        secs+=float(m.get('secs') or 0)
        u=m.get('usage') or {}
        toks+=int(u.get('total') or 0)
        sample=int(mf.split('.')[0])
        npy=os.path.join(d,'%d.npy'%sample)
        if m.get('valid') and os.path.exists(npy):
            try:
                a=np.load(npy).astype(float).squeeze()
                if a.shape==(6,) and np.all(np.isfinite(a)):
                    n_valid+=1
                    g=gt[sample]
                    errs.append(np.median([abs(a[j]-g[j])/max(abs(g[j]),1e-9) for j in range(6)]))
            except Exception: pass
    overall=('%.1f%%'%(100*np.median(errs))) if errs else '--'
    rows.append((cell,model or '--',n_att,n_valid,overall,toks,round(secs,0)))

w=max(len(r[0]) for r in rows) if rows else 10
print(f"{'cell':<{w}}  {'model_used':<26} {'att':>3} {'val':>3} {'err':>7} {'tokens':>9} {'time_s':>7} {'tok/s':>6}")
print('-'*(w+70))
TT=TS=0
for cell,model,att,val,err,toks,secs in rows:
    tps=('%.0f'%(toks/secs)) if secs>0 and toks>0 else '--'
    print(f"{cell:<{w}}  {model[:26]:<26} {att:>3} {val:>3} {err:>7} {toks:>9,} {int(secs):>7} {tps:>6}")
    TT+=toks; TS+=secs
print('-'*(w+70))
print(f"{'TOTAL':<{w}}  {'':<26} {'':>3} {'':>3} {'':>7} {TT:>9,} {int(TS):>7}")
print(f"\ntokens = usage.total per sample (prompt+output, incl. cache reads); time = wall seconds per openclaw run.")
