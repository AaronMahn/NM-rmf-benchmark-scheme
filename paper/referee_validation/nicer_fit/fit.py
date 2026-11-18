import sys,json,time,argparse
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution,minimize
import rmf,vpaths
LO=np.array([-11.5,10.5,2.5,900,-40,0,-.003,460,0.]);HI=np.array([-8.5,14.5,6,2000,40,320,.003,720,5000.])
SCALE=np.array([10,12,5,1400,50,100,.001,600,1000.])
# TM1-based optimiser start vector, in rmf.NAMES order. Preserve full precision
# for the differential evolution seed and the second MCMC ensemble.
TM1_START_VECTOR = (
    -9.928239713298373,  # g_sigma
    12.86075717568663,  # g_omega
    3.894553761910849,  # g_rho
    1742.6982394398517,  # g2
    -0.6180965592076041,  # g3
    107.64089391516593,  # c3
    -0.00028972417831420893,  # g_s6
    4.92380197397712e2,  # m_sigma
    0.0,  # c4
)
class Objective:
 def __init__(self,joint=False,extension=False,bw=1.,radio=False):
  self.joint=joint;self.ext=extension;self.n=9 if extension else 8;self.bw=bw;self.radio=radio;self.best=1e99;self.calls=0
  if joint:
   from nicer import JointLikelihood
   self.like=JointLikelihood(bw,retain_radio=radio)
 def evaluate(self,p,fine=False):
  ob,c,co=rmf.observables(p,fine=fine)
  t=rmf.terms(ob)
  if self.joint:
   if self.radio:t[6]=0. # Do not reuse J0740 radio mass as an additional lower-bound penalty.
   t=np.r_[t[:7],self.like.scores(c)]
  return float(sum(t)),ob,t,c,co
 def __call__(self,z):
  self.calls+=1;p=np.r_[z*SCALE[:self.n],[] if self.ext else [0.]]
  if np.any(p<LO) or np.any(p>HI):return 1e8
  try:
   s,ob,t,c,co=self.evaluate(p)
   if not np.isfinite(s):return 1e8
   if s<self.best:self.best=s
   return s
  except (ValueError,ZeroDivisionError,np.linalg.LinAlgError):return 1e8

def run(joint=False,ext=False,maxiter=100,bw=1.,radio=False):
 name=('joint' if joint else 'gaussian')+('_lambda' if ext else '')+(f'_bw{bw:g}' if bw!=1 else '')+('_radio' if radio else '')
 out=vpaths.result_dir('A2' if ext else 'A1');obj=Objective(joint,ext,bw,radio)
 bounds=list(zip(LO[:obj.n]/SCALE[:obj.n],HI[:obj.n]/SCALE[:obj.n]))
 start=np.array(TM1_START_VECTOR[:obj.n])/SCALE[:obj.n]
 if ext:start[-1]=.5
 t=time.time();print(name,'start',obj(start),flush=True)
 de=differential_evolution(obj,bounds,x0=start,seed=20261112,popsize=10,maxiter=maxiter,tol=1e-6,polish=False,updating='immediate')
 print('DE',de.fun,obj.calls,time.time()-t,flush=True)
 res=minimize(obj,de.x,method='Powell',bounds=bounds,options={'maxiter':80,'xtol':1e-5,'ftol':1e-6})
 z=de.x if de.fun<res.fun else res.x
 # Nelder-Mead handles the residual small TOV discretisation noise better than finite gradients.
 res2=minimize(obj,z,method='Nelder-Mead',options={'maxiter':2500,'xatol':1e-6,'fatol':1e-5})
 if res2.fun<obj(z):z=res2.x
 p=np.r_[z*SCALE[:obj.n],[] if ext else [0.]]
 score,ob,terms,c,co=obj.evaluate(p);sf,of,tf,cf,cof=obj.evaluate(p,True)
 record={'case':name,'couplings':dict(zip(rmf.NAMES,p)),'observables_names':['n0','E','K','Esym','L','m_sigma','Mmax','R14'],'observables':ob.tolist(),'score':score,'terms':terms.tolist(),'fine_observables':of.tolist(),'fine_score':sf,'fine_terms':tf.tolist(),'calls':obj.calls,'seconds':time.time()-t,'bounds':dict(zip(rmf.NAMES,zip(LO.tolist(),HI.tolist()))),'convergence_note':'multistart global DE + Powell + Nelder-Mead, finite numerical resolution; no claim of unique/global optimum'}
 (out/(name+'.json')).write_text(json.dumps(record,indent=2));np.savetxt(out/(name+'_mr.csv'),cf,delimiter=',',header='central_density_fm-3,mass_msun,radius_km');np.savetxt(out/(name+'_eos.csv'),cof,delimiter=',',header='density_fm-3,pressure_MeVfm-3,energy_MeVfm-3')
 print(json.dumps(record,indent=2),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--joint',action='store_true');ap.add_argument('--lambda-v',dest='ext',action='store_true');ap.add_argument('--iterations',type=int,default=100);ap.add_argument('--bw',type=float,default=1.);ap.add_argument('--radio',action='store_true');a=ap.parse_args();run(a.joint,a.ext,a.iterations,a.bw,a.radio)
