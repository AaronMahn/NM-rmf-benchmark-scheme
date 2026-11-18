import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json,numpy as np,emcee,multiprocessing as mp
import mcmc,rmf,vpaths
B=vpaths.WORK

def calc(q):
 try:return rmf.observables(mcmc.top(q))[0]
 except Exception:return np.full(8,np.nan)
def raw(q):
 a,b,c,d,go,gr,c3,ms=(q*mcmc.QS).T;t=ms/a
 return np.column_stack((-t,go,gr,b*t**3,c*t**4,c3,d*t**6,ms))
def summarize():
 chains=[];ds=[];acc=[];quant=[];burn=4000
 for seed in [1234,5678]:
  j=np.load(B/'mcmc'/f'chain_{seed}.npz');c=j['chain'][burn:];chains.append(c);acc.append(j['acceptance'].tolist());quant.append(np.quantile(raw(c.reshape(-1,8)),[.16,.5,.84],axis=0).tolist());ds.append(json.loads((B/'mcmc'/f'diagnostics_{seed}.json').read_text()))
 n=min(len(c) for c in chains);c=np.concatenate([v[-n:] for v in chains],axis=1);tau=emcee.autocorr.integrated_time(c,quiet=True);r=mcmc.rhat(c[::10]);rng=np.random.default_rng(20261112)
 # Uniform sample of retained states for derived-observable CIs; all samples for coupling CIs.
 flat=c.reshape(-1,8);draw=flat[rng.choice(len(flat),12000,replace=False)]
 with mp.get_context('fork').Pool(6) as pool:obs=np.array(pool.map(calc,draw,chunksize=30))
 if not np.all(np.isfinite(obs)):raise RuntimeError('posterior derived observables failed')
 np.savez_compressed(vpaths.result_dir('A1')/'posterior_draws.npz',q=draw,couplings=raw(draw),observables=obs)
 names=['n0','E','K','Esym','L','m_sigma','Mmax','R14'];coupling_names=rmf.NAMES[:8]
 cqt=np.quantile(raw(flat),[.16,.5,.84],axis=0);oqt=np.quantile(obs,[.16,.5,.84],axis=0)
 result={'seeds':[1234,5678],'steps_per_ensemble':[len(x)+burn for x in chains],'walkers_per_ensemble':64,'burn_steps':burn,'retained_states':len(flat),'derived_draws':len(draw),'acceptance_mean':[float(np.mean(x)) for x in acc],'tau_q_steps':tau.tolist(),'retained_length_over_tau':(n/tau).tolist(),'nominal_ESS':(len(flat)/tau).tolist(),'rank_split_Rhat_walkers_diagnostic':r.tolist(),'Rhat_caveat':'Walkers within each ensemble are correlated; Rhat is a diagnostic, not independent-chain convergence proof. Two independent ensembles and their quantiles are also compared.','independent_ensemble_coupling_quantiles':quant,'coupling_68pct':{name:dict(zip(['q16','median','q84'],v)) for name,v in zip(coupling_names,cqt.T.tolist())},'observable_68pct':{name:dict(zip(['q16','median','q84'],v)) for name,v in zip(names,oqt.T.tolist())},'model':'sigma6 without c4, joint NICER nominal Scott KDE','prior':'Flat in eight raw original couplings within fit.py LO/HI, same NM Gaussian terms and soft one-sided Mmax term as legacy. Transformation Jacobian included.'}
 (vpaths.result_dir('A1')/'posterior_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':summarize()
