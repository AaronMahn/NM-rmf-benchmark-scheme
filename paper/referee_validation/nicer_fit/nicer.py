"""Joint KDE likelihood p(M,R)/original marginal M,R prior.
FFT binned KDE with full Scott covariance; exact KDE cross-check in validate.py.
No random KDE subsampling. All equal-weight posterior points used.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import vpaths
import numpy as np,json
from scipy.signal import fftconvolve
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm
BASE=vpaths.WORK;BASE.mkdir(parents=True,exist_ok=True)
DATA=BASE/'data'
SOURCES={'J0030':(None,None,205.53),'J0740':(2.082,.0703,346.53),'J0437':(1.418,.044,173.69)}
def prior_mass(name,m):
 mu,sig,_=SOURCES[name]
 return np.ones_like(m) if mu is None else norm.pdf(m,mu,sig)
def support(name,m,r):
 # Identical rectangular limits and physical-domain conditions in the sampled region.
 # Rotation correction to polar radius from AlGendy & Morsink (2014), X-PSI spacetime.
 freq=SOURCES[name][2];x=1.476625*m/r
 epsilon=(2*np.pi*freq)**2*(r*1e5)**3/(1.327164e26*m)
 rpolar=r*(1+epsilon*(-.788+1.030*x))
 g0=1.327164e26*m/(r*1e5)**2/np.sqrt(np.maximum(1-2*x,1e-12))
 # Full Eq.21/Table5 gravity, as in X-PSI effective_gravity_universal.pyx.
 ae=(-.791+.776*x+(-1.315+2.431*x)*epsilon*x-1.172*x*epsilon**2)*epsilon
 d60=(13.47-27.13*x)*epsilon*x
 ap=(1.138-1.431*x+(.653-2.864*x)*epsilon*x+.975*x*epsilon**2-d60)*epsilon
 b=d60*epsilon; a=ap-ae
 mu=np.clip(-b/np.where(np.abs(2*a)>1e-15,2*a,1e-15),0,1)
 gv=1+ae+a*mu**2+b*mu
 gmin=g0*np.minimum(np.minimum(1+ae,1+ap+b),gv)
 gmax=g0*np.maximum(np.maximum(1+ae,1+ap+b),gv)
 return (m>=1)&(m<=3)&(r>=3*1.476625)&(r<=16)&(rpolar>3*1.476625*m)&(gmin>10**13.7)&(gmax<1e15)
def load_samples(name):
 return np.load(BASE/'samples'/f'{name}.npz')['samples']
def make_grid(name,bw=1.,dm=.0015,dr=.015):
 s=load_samples(name);cov=np.cov(s.T)*len(s)**(-1/3)*bw*bw
 hm,hr=np.sqrt(np.diag(cov));m=np.arange(max(1,s[:,0].min()-8*hm),min(3,s[:,0].max()+8*hm)+dm/2,dm);r=np.arange(3*1.476625,16+dr/2,dr)
 edgesm=np.r_[m-dm/2,m[-1]+dm/2];edgesr=np.r_[r-dr/2,r[-1]+dr/2]
 hist=np.histogram2d(s[:,0],s[:,1],bins=(edgesm,edgesr))[0]/len(s)
 km=np.arange(-np.ceil(8*hm/dm),np.ceil(8*hm/dm)+1)*dm;kr=np.arange(-np.ceil(8*hr/dr),np.ceil(8*hr/dr)+1)*dr
 mm,rr=np.meshgrid(km,kr,indexing='ij');q=np.linalg.inv(cov)
 kern=np.exp(-.5*(q[0,0]*mm*mm+2*q[0,1]*mm*rr+q[1,1]*rr*rr))/(2*np.pi*np.sqrt(np.linalg.det(cov)))
 dens=fftconvolve(hist,kern,mode='same');dens[dens<dens.max()*1e-13]=0
 prior=prior_mass(name,m);like=dens/prior[:,None];MM,RR=np.meshgrid(m,r,indexing='ij');like[~support(name,MM,RR)]=0
 # Units/constants: marginal flat radius density omitted, flat M prior J0030 omitted;
 # Gaussian prior density included in full. Positive score shifts fixed per-source.
 (BASE/'samples').mkdir(exist_ok=True);np.savez_compressed(BASE/'samples'/f'{name}_grid_bw{bw:g}.npz',m=m,r=r,density=dens,likelihood=like,cov=cov)
 return m,r,dens,like,cov
class JointLikelihood:
 def __init__(self,bw=1.,retain_radio=False):
  self.tables=[];self.names=list(SOURCES);self.bw=bw
  for name in self.names:
   f=BASE/'samples'/f'{name}_grid_bw{bw:g}.npz'
   if not f.exists():make_grid(name,bw)
   j=np.load(f);m,r=j['m'],j['r'];v=j['likelihood']
   if retain_radio:
    v=v*prior_mass(name,m)[:,None]
   self.tables.append((m,RegularGridInterpolator((m,r),v,bounds_error=False,fill_value=0.),float(v.max())))
 def scores(self,c):
  c=c[:np.argmax(c[:,1])+1];c=c[c[:,1]>.5]
  if len(c)<2 or np.any(np.diff(c[:,1])<=0):return np.full(3,1e7)
  out=[]
  for name,(grid,fn,scale) in zip(self.names,self.tables):
   hi=min(c[-1,1],grid[-1]);lo=max(c[0,1],grid[0])
   if hi<=lo:out.append(1e7);continue
   # Include model endpoints to avoid dependence on mass-grid phase.
   m=np.r_[lo,grid[(grid>lo)&(grid<hi)],hi];r=np.interp(m,c[:,1],c[:,2]);v=fn(np.column_stack((m,r)))
   integ=np.trapezoid(v,m)/scale # fixed source-dependent additive constants only
   out.append(-2*np.log(max(integ,1e-300)))
  return np.array(out)
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--bw',type=float,default=1.);a=ap.parse_args()
 for name in SOURCES:
  s=load_samples(name);make_grid(name,a.bw);print(name,len(s),np.quantile(s,[.16,.5,.84],axis=0).tolist(),flush=True)
