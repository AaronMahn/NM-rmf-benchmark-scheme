"""Exact evaluated posterior, no emulator. Flat priors in raw published coupling ranges.
Sample sigma-sector invariants to remove the redundant field-rescaling direction.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,sys,time,argparse,multiprocessing as mp
from pathlib import Path
import numpy as np,emcee
from scipy.special import ndtri
from scipy.stats import rankdata
import rmf,vpaths
from fit import LO,HI,SCALE,Objective,TM1_START_VECTOR
BASE=vpaths.WORK
QS=np.array([55.,2.,.002,3e-9,12.,5.,100.,600.])
OBJ=None
def toq(p):
 t=-p[0];return np.array([p[7]/t,p[3]/t**3,p[4]/t**4,p[6]/t**6,p[1],p[2],p[5],p[7]])/QS
def top(q):
 a,b,c,d,go,gr,c3,ms=q*QS;t=ms/a
 return np.array([-t,go,gr,b*t**3,c*t**4,c3,d*t**6,ms,0.])
def init():
 global OBJ
 OBJ=Objective(joint=True)
def logprob(q):
 p=top(q)
 if np.any(p<LO) or np.any(p>HI):return -np.inf
 try:
  s,*_=OBJ.evaluate(p)
  if not np.isfinite(s):return -np.inf
  # det d(raw sigma couplings)/d(a,b,c,d,ms)=t^14/a, constants cancel.
  a=q[0]*QS[0];t=-p[0]
  return -.5*s+14*np.log(t)-np.log(a)
 except (ValueError,ZeroDivisionError,np.linalg.LinAlgError):return -np.inf

def rhat(x):
 # Per-walker rank/split diagnostic only; ensemble walkers are dependent.
 # This is not a formal independent-chain Rhat convergence proof.
 n=x.shape[0]//2;x=np.concatenate((x[:n],x[-n:]),axis=1);n=len(x)
 z=ndtri((rankdata(x.reshape(-1,x.shape[-1]),axis=0)-.375)/(x.shape[0]*x.shape[1]+.25)).reshape(x.shape)
 W=np.mean(np.var(z,axis=0,ddof=1),axis=0);B=n*np.var(np.mean(z,axis=0),axis=0,ddof=1)
 return np.sqrt(((n-1)/n*W+B/n)/W)

def run(steps,workers,seed,walkers,resume=False):
 init();np.random.seed(seed);rng=np.random.default_rng(seed)
 best=json.loads(vpaths.reference_or_result('A1','joint.json').read_text());q0=toq(rmf.vector(best['couplings']))
 if seed==5678:
  q0=toq(np.array(TM1_START_VECTOR))
 out=BASE/'mcmc';out.mkdir(exist_ok=True);f=out/f'chain_{seed}.npz';history=[]
 if resume and f.exists():
  j=np.load(f);history=[j['chain']];start=j['chain'][-1];oldlp=j['log_prob'];oldaccept=j['acceptance'];np.random.set_state(tuple(j['random_state'].tolist())) if 'random_state' in j else None
 else:
  start=q0+rng.normal(size=(walkers,8))*.002
  for i in range(walkers):
   while not np.isfinite(logprob(start[i])):start[i]=q0+rng.normal(size=8)*.0002
  oldlp=None
 t=time.time()
 with mp.get_context('fork').Pool(workers,initializer=init) as pool:
  sampler=emcee.EnsembleSampler(walkers,8,logprob,pool=pool,moves=[(emcee.moves.DEMove(),.6),(emcee.moves.StretchMove(),.3),(emcee.moves.DESnookerMove(),.1)])
  state=start
  for begin in range(0,steps,500):
   state=sampler.run_mcmc(state,min(500,steps-begin),progress=False,skip_initial_state_check=True)
   chain=np.concatenate(history+[sampler.get_chain()]);lp=sampler.get_log_prob();lp=np.concatenate([oldlp,lp]) if oldlp is not None else lp
   try:tau=emcee.autocorr.integrated_time(chain[len(chain)//3:],quiet=True).tolist()
   except Exception:tau=None
   nold=len(history[0]) if history else 0
   acceptance=(oldaccept*nold+sampler.acceptance_fraction*sampler.iteration)/len(chain) if nold else sampler.acceptance_fraction
   np.savez_compressed(f,chain=chain,log_prob=lp,acceptance=acceptance)
   diag={'seed':seed,'steps':len(chain),'walkers':walkers,'acceptance':float(np.mean(acceptance)),'tau_after_first_third':tau,'seconds_current_invocation':time.time()-t}
   (out/f'diagnostics_{seed}.json').write_text(json.dumps(diag,indent=2));print(json.dumps(diag),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=12000);ap.add_argument('--workers',type=int,default=6);ap.add_argument('--seed',type=int,default=1234);ap.add_argument('--walkers',type=int,default=64);ap.add_argument('--resume',action='store_true');a=ap.parse_args();run(a.steps,a.workers,a.seed,a.walkers,a.resume)
