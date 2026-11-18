from pathlib import Path
import json,sys,numpy as np
from numba import njit
import rmf,nicer,vpaths
@njit(cache=True)
def exact_density(pts,s,cov):
 q=np.linalg.inv(cov);hm=np.sqrt(cov[0,0]);norm=2*np.pi*np.sqrt(np.linalg.det(cov))*len(s);out=np.zeros(len(pts))
 for i in range(len(pts)):
  m,r=pts[i];lo=np.searchsorted(s[:,0],m-8*hm);hi=np.searchsorted(s[:,0],m+8*hm)
  for j in range(lo,hi):
   dm=m-s[j,0];dr=r-s[j,1];out[i]+=np.exp(-.5*(q[0,0]*dm*dm+2*q[0,1]*dm*dr+q[1,1]*dr*dr))
 return out/norm

def direct_scores(c,bw=1.,dm=.0015):
 c=c[:np.argmax(c[:,1])+1];out=[];like=nicer.JointLikelihood(bw)
 for name,(grid,fn,scale) in zip(nicer.SOURCES,like.tables):
  s=nicer.load_samples(name);s=s[np.argsort(s[:,0])];cov=np.cov(s.T)*len(s)**(-1/3)*bw*bw
  lo=max(1.,c[0,1],s[:,0].min()-8*np.sqrt(cov[0,0]));hi=min(3.,c[-1,1],s[:,0].max()+8*np.sqrt(cov[0,0]));m=np.linspace(lo,hi,int((hi-lo)/dm)+2);r=np.interp(m,c[:,1],c[:,2])
  dens=exact_density(np.column_stack((m,r)),s,cov);v=dens/nicer.prior_mass(name,m);v[~nicer.support(name,m,r)]=0
  out.append(-2*np.log(max(np.trapezoid(v,m)/scale,1e-300)))
 return np.array(out)

def main():
 from fit import Objective
 out={}
 for folder,name in [('A1','gaussian'),('A1','joint'),('A2','gaussian_lambda'),('A2','joint_lambda')]:
  try:f=vpaths.reference_or_result(folder,name+'.json')
  except FileNotFoundError:continue
  j=json.loads(f.read_text());p=rmf.vector(j['couplings']);_,c,co=rmf.observables(p,fine=True)
  ob=rmf.nuclear(p);v=[]
  for bw in [.7,1.,1.4]:
   like=nicer.JointLikelihood(bw);grid=like.scores(c);direct=direct_scores(c,bw);v.append({'bandwidth_multiplier':bw,'FFT_scores':grid.tolist(),'direct_scores':direct.tolist(),'max_abs_grid_error':float(np.max(np.abs(grid-direct)))})
  out[name]=v;print(name,v,flush=True)
 (vpaths.result_dir('A1')/'kde_validation.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
