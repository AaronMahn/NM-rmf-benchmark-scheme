import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,numpy as np,multiprocessing as mp
from pathlib import Path
import rmf,nicer,vpaths
from fit import Objective
B=vpaths.WORK

def fine(q):
 import mcmc
 p=mcmc.top(q);obj=Objective(True);s,o,t,c,_=obj.evaluate(p);sf,of,tf,cf,_=obj.evaluate(p,True)
 return np.r_[of[6]-o[6],of[7]-o[7],sf-s]
def derive(q):
 import mcmc
 return rmf.observables(mcmc.top(q))[0]
if __name__=='__main__':
 j=np.load(vpaths.result_dir('A1')/'posterior_draws.npz');rng=np.random.default_rng(1912);q=j['q'][rng.choice(len(j['q']),100,replace=False)]
 with mp.get_context('fork').Pool(6) as pool:v=np.array(pool.map(fine,q))
 out={'fine_minus_sampling':{'n_draws':100,'columns':['Mmax','R14','joint_score'],'median':np.median(v,axis=0).tolist(),'q05':np.quantile(v,.05,axis=0).tolist(),'q95':np.quantile(v,.95,axis=0).tolist(),'max_abs':np.max(np.abs(v),axis=0).tolist()},'independent_ensemble_observable_quantiles':{}}
 for seed in [1234,5678]:
  c=np.load(B/'mcmc'/f'chain_{seed}.npz')['chain'][4000:].reshape(-1,8);q=c[rng.choice(len(c),3000,replace=False)]
  with mp.get_context('fork').Pool(6) as pool:o=np.array(pool.map(derive,q,chunksize=30))
  out['independent_ensemble_observable_quantiles'][str(seed)]=np.quantile(o,[.16,.5,.84],axis=0).tolist()
 (vpaths.result_dir('A1')/'posterior_checks.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
