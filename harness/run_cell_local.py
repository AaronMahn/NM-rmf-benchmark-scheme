#!/usr/bin/env python3
import sys, os, subprocess, json, time, shutil, re
from runtime_config import OPENCLAW_HOME, OPENCLAW_WORKSPACE_DIR
try:
    import numpy as np
except Exception:
    np = None
# args: agent cell_dir tasks_dir runtag ids_csv ws_dir intended_model [timeout] [store]
agent, cell_dir, tasks_dir, runtag, ids_csv, ws_dir, intended_model = sys.argv[1:8]
per_to = int(sys.argv[8]) if len(sys.argv) > 8 else 900
store = sys.argv[9] if len(sys.argv) > 9 else os.path.join(os.path.dirname(os.path.dirname(cell_dir.rstrip('/'))), 'store.jsonl')
cell = os.path.basename(cell_dir.rstrip('/'))
expected_core = intended_model.split('/')[-1]          # e.g. claude-opus-4.8 ; gpt-5.4
ids = [int(x) for x in ids_csv.split(',') if x != '']
os.makedirs(cell_dir, exist_ok=True)
logdir = os.path.join(cell_dir, 'logs'); os.makedirs(logdir, exist_ok=True)
prog = os.path.join(cell_dir, 'progress.txt')
env = dict(os.environ)
KEEP = {'AGENTS.md','BOOTSTRAP.md','HEARTBEAT.md','IDENTITY.md','SOUL.md','TOOLS.md',
        'USER.md','MEMORY.md','skills',OPENCLAW_WORKSPACE_DIR,'memory','.git','out'}
def clean_ws():
    if not ws_dir or not os.path.isdir(ws_dir): return
    for name in os.listdir(ws_dir):
        if name in KEEP: continue
        p = os.path.join(ws_dir, name)
        try: shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        except OSError: pass
def log(m):
    with open(prog,'a') as f: f.write(time.strftime('%H:%M:%S')+' '+m+'\n')
def find_usage(o):
    found={}
    def w(x):
        if isinstance(x,dict):
            ks=set(x.keys())
            if {'input','output'}<=ks or 'promptTokens' in ks or {'inputTokens','outputTokens'}<=ks:
                for k,v in x.items():
                    if isinstance(v,(int,float)): found[k]=v
            for v in x.values(): w(v)
        elif isinstance(x,list):
            for v in x: w(v)
    w(o); return found
def parse_provider_model(txt):
    p=re.search(r'"provider":\s*"([^"]*)"', txt); m=re.search(r'"model":\s*"([^"]*)"', txt)
    return (p.group(1) if p else '', m.group(1) if m else '')

log('START cell=%s agent=%s intended=%s n=%d' % (cell, agent, intended_model, len(ids)))
for n in ids:
    npy = os.path.join(cell_dir, '%d.npy' % n)
    meta = os.path.join(cell_dir, '%d.meta.json' % n)
    # ---- RESUME: skip if we already have a VALID cached result (correct model) ----
    if os.path.exists(npy) and os.path.exists(meta):
        try:
            mm = json.load(open(meta))
            if mm.get('valid'):
                log('sample %d CACHED (model=%s) skip' % (n, mm.get('model_used'))); continue
        except Exception: pass
    clean_ws()
    task = open(os.path.join(tasks_dir, '%d.txt' % n), encoding='utf-8').read()
    try:
        if os.path.exists(npy): os.remove(npy)
    except OSError: pass
    msg = (task + '\n\n请将最终结果保存为 numpy 文件：' + npy +
           '\n要求：np.ndarray, shape=(6,), dtype=float, 顺序为 '
           '[binding_energy_MeV, pressure_MeV_fm3, compressibility_MeV, symmetry_energy_MeV, M_msun, R_km]。'
           '注意：题目给定的耦合常数为随机抽样得到的固定输入值，并非经验/物理拟合值，因此正常核物质的经验规律（如结合能约-16 MeV、K约240 MeV等）在这里不适用。请直接用题目给定的参数做正向计算得到这六个量，严禁扫描、拟合或调整任何参数去匹配任何经验值。完成后只回复 SAVED。')
    sid = '%s_%s_s%d' % (agent, runtag, n)
    t0 = time.time(); exitc='?'; usage={}; prov=mod=''
    try:
        # clear stale session locks for this agent (a prior force-killed sample can orphan a lock)
        import glob as _glob
        for _lk in _glob.glob(str(OPENCLAW_HOME / 'agents' / agent / 'sessions' / '*.lock')):
            try: os.remove(_lk)
            except OSError: pass
        r = subprocess.run(['openclaw','agent','--local','--agent',agent,'--session-id',sid,
                            '--message',msg,'--json','--timeout',str(per_to)],
                           env=env, capture_output=True, text=True, timeout=per_to+200)
        out=(r.stdout or ''); open(os.path.join(logdir,'%d.log'%n),'w').write(out+'\n---STDERR---\n'+(r.stderr or ''))
        exitc=r.returncode; prov,mod=parse_provider_model(out)
        try: usage=find_usage(json.loads(out[out.index('{'):]))
        except Exception: pass
    except subprocess.TimeoutExpired:
        exitc='TIMEOUT'
    secs=round(time.time()-t0,1)
    if not usage:
        try:
            import glob as _g2
            jf=_g2.glob(str(OPENCLAW_HOME / 'agents' / agent / 'sessions' / (sid + '.jsonl')))
            tot=0
            def _walk(x):
                if isinstance(x,dict):
                    if 'totalTokens' in x or ('input' in x and 'output' in x): yield x
                    for v in x.values(): yield from _walk(v)
                elif isinstance(x,list):
                    for v in x: yield from _walk(v)
            if jf:
                for ln in open(jf[0]):
                    try: o=json.loads(ln)
                    except: continue
                    for u in _walk(o):
                        tot += (u.get('input',0) or 0)+(u.get('output',0) or 0)
            if tot: usage={'total':int(tot)}
        except Exception: pass
    npy_ok=os.path.exists(npy)
    model_ok = expected_core in (mod or '')
    valid = bool(npy_ok)
    pred=None
    if npy_ok and np is not None:
        try:
            a=np.load(npy).astype(float).squeeze()
            if a.shape==(6,) and np.all(np.isfinite(a)): pred=a.tolist()
            else: valid=False
        except Exception: valid=False
    rec={'ts':time.strftime('%Y-%m-%dT%H:%M:%S'),'cell':cell,'sample':n,
         'intended_model':intended_model,'provider_used':prov,'model_used':mod,
         'model_match':model_ok,'npy':npy_ok,'valid':valid,'secs':secs,
         'usage':usage,'prediction':pred,'sid':sid,'exit':exitc}
    json.dump(rec, open(meta,'w'))
    with open(store,'a') as f: f.write(json.dumps(rec)+'\n')
    log('sample %d model=%s valid=%s npy=%s %.0fs' % (n, mod or '?', valid, npy_ok, secs))
open(os.path.join(cell_dir,'DONE'),'w').write('done\n'); log('ALL DONE')
