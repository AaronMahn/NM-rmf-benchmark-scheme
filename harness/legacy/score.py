#!/usr/bin/env python3
import sys, os, json
import numpy as np
COLS = ['binding_energy_MeV','pressure_MeV_fm3','compressibility_MeV','symmetry_energy_MeV','M_msun','R_km']
gt_path = sys.argv[1]
gt = np.load(gt_path)  # (100,6)
cells = sys.argv[2:]   # list of cell dirs
def load_pred(p):
    try:
        a = np.load(p, allow_pickle=False).astype(float).squeeze()
        if a.shape != (6,): return None
        if not np.all(np.isfinite(a)): return None
        return a
    except Exception:
        return None
print('GT shape', gt.shape, '| GT[0]=', np.round(gt[0],3))
for cell in cells:
    name = os.path.basename(cell.rstrip('/'))
    att = []
    ap = os.path.join(cell,'attempted.txt')
    if os.path.exists(ap):
        att = [int(x) for x in open(ap).read().split(',') if x!='']
    else:
        att = sorted(int(f[:-4]) for f in os.listdir(cell) if f.endswith('.npy'))
    valid, rel = 0, {c:[] for c in COLS}
    for n in att:
        a = load_pred(os.path.join(cell,'%d.npy'%n))
        if a is None: continue
        valid += 1
        g = gt[n]
        for j,c in enumerate(COLS):
            denom = max(abs(g[j]), 1e-9)
            rel[c].append(abs(a[j]-g[j])/denom)
    print('\n=== %s ===' % name)
    print('attempted=%d  valid_npy=%d  valid_rate=%.0f%%' %
          (len(att), valid, 100.0*valid/max(len(att),1)))
    if valid:
        meds = [np.median(rel[c]) for c in COLS]
        for c,m in zip(COLS,meds):
            print('  %-22s median_rel_err=%6.1f%%' % (c, 100*m))
        print('  %-22s median_rel_err=%6.1f%%' % ('OVERALL(6-target)', 100*np.median([x for c in COLS for x in rel[c]])))
