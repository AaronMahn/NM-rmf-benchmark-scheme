"""Fast checked TM1+sigma6+(.5*c4*omega²*rho²) backend, MeV/fm units.
No change to released toolkit. Equations/energy validated independently against it.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import vpaths
import numpy as np
from numba import njit
from scipy.optimize import brentq, minimize_scalar
H=197.326968; H3=H**3
vpaths.check_toolkit()
BPS=np.loadtxt(vpaths.BPS)
BPS=BPS[BPS[:,0]<.08]
NAMES=['g_sigma','g_omega','g_rho','g2','g3','c3','g_s6','m_sigma','c4']
@njit(cache=True)
def fermi(k,m):
 if k<=0:return 0.,0.
 if m<=0:return np.nan,np.nan
 t=k/m; a=np.sqrt(1+t*t)
 rs=m**3/(2*np.pi**2)*(t*a-np.arcsinh(t))
 e=m**4/(8*np.pi**2)*(t*a*(1+2*t*t)-np.arcsinh(t))
 return rs,e
@njit(cache=True)
def state(x,n,p,beta):
 gs,go,gr,g2,g3,c3,g6,ms,c4=p
 s,o,r,y=x; m=939+gs*s; nb=n*H3; np_=nb*y; nn=nb-np_
 kp=(3*np.pi**2*max(np_,0))**(1/3); kn=(3*np.pi**2*max(nn,0))**(1/3)
 rsp,ep=fermi(kp,m); rsn,en=fermi(kn,m)
 mun=np.sqrt(kn*kn+m*m)+go*o-gr*r; mup=np.sqrt(kp*kp+m*m)+go*o+gr*r
 mue=np.sqrt(kp*kp+.511**2)
 z=np.empty(4)
 z[0]=(ms*ms*s+g2*s*s+g3*s**3+g6*s**5+gs*(rsp+rsn))/max(abs(gs)*nb,1)
 z[1]=(783**2*o+c3*o**3+c4*o*r*r-go*nb)/max(go*nb,1)
 z[2]=(770**2*r+c4*o*o*r-gr*(np_-nn))/max(gr*nb,1)
 z[3]=(mun-mup-mue)/500 if beta else y-.5
 u=.5*ms*ms*s*s+g2*s**3/3+g3*s**4/4+g6*s**6/6
 e=ep+en+u-.5*783**2*o*o-c3*o**4/4-.5*770**2*r*r-.5*c4*o*o*r*r+go*o*nb+gr*r*(np_-nn)
 ee=fermi(kp,.511)[1] if beta else 0.
 et=(e+ee)/H3
 pr=(nn*mun+np_*mup+(np_*mue if beta else 0.))/H3-et
 return z,et,pr
@njit(cache=True)
def solve(n,p,beta,x):
 x=x.copy()
 for it in range(60):
  z,e,pr=state(x,n,p,beta); err=np.max(np.abs(z))
  if err<2e-10:return x,e,pr,err
  jac=np.empty((4,4))
  for j in range(4):
   xx=x.copy(); h=1e-5*max(abs(x[j]),.1);xx[j]+=h
   zz,_,_=state(xx,n,p,beta);jac[:,j]=(zz-z)/h
  try: dx=np.linalg.solve(jac,-z)
  except: return x,np.nan,np.nan,1.
  ok=False
  for k in range(20):
   xx=x+dx*2.**(-k)
   if xx[0]<0 or 939+p[0]*xx[0]<1 or xx[1]<0 or xx[3]<1e-9 or xx[3]>.5000001:continue
   zz,_,_=state(xx,n,p,beta)
   if np.all(np.isfinite(zz)) and np.max(np.abs(zz))<err: x=xx;ok=True;break
  if not ok:return x,np.nan,np.nan,err
 return x,np.nan,np.nan,err
@njit(cache=True)
def snm(n,p):
 x,e,pr,err=solve(n,p,False,np.array([20.,15.,0.,.5]))
 return x,e,pr,err
@njit(cache=True)
def symmetry(n,p):
 x,e,pr,err=snm(n,p)
 m=939+p[0]*x[0];k=(1.5*np.pi**2*n*H3)**(1/3)
 return k*k/(6*np.sqrt(k*k+m*m))+p[2]**2*n*H3/(2*(770**2+p[8]*x[1]**2))
def nuclear(p,legacy=False):
 n=float(minimize_scalar(lambda n: snm(n,p)[1]/n-939,bounds=(.05,.25),method='bounded',options={'xatol':1e-5 if legacy else 1e-9}).x)
 x,e,pr,err=snm(n,p)
 if err>1e-8 or not np.isfinite(e): raise ValueError('SNM failed')
 h=1e-4
 em=snm(n-h,p)[1]/(n-h)-939; ep=snm(n+h,p)[1]/(n+h)-939; e0=e/n-939
 K=9*(2*n*(ep-em)/(2*h)+n*n*(ep-2*e0+em)/(h*h))
 S=symmetry(n,p);L=3*n*(symmetry(n+h,p)-symmetry(n-h,p))/(2*h)
 return np.array([n,e0,K,S,L,p[7]])
@njit(cache=True)
def core(p,grid):
 rows=np.empty((len(grid),3));x=np.array([15.,10.,-3.,.05]);errmax=0.
 for i in range(len(grid)):
  x,e,pr,err=solve(grid[i],p,True,x);errmax=max(errmax,err)
  if not np.isfinite(e) or err>1e-8:return rows[:i],errmax
  rows[i]=np.array([grid[i],pr,e])
 return rows,errmax
@njit(cache=True)
def merge(crust,co):
 tab=np.vstack((crust,co));tab=tab[np.argsort(tab[:,1])];out=np.empty_like(tab);j=0;lp=-1.;le=-1.
 for i in range(len(tab)):
  if tab[i,1]>lp+1e-15 and tab[i,2]>le+1e-15:
   out[j]=tab[i];j+=1;lp=tab[i,1];le=tab[i,2]
 return out[:j]
@njit(cache=True)
def interp(p,ps,es):
 i=np.searchsorted(ps,p)-1;i=max(0,min(i,len(ps)-2))
 return es[i]+(p-ps[i])*(es[i+1]-es[i])/(ps[i+1]-ps[i])
@njit(cache=True)
def rhs(r,m,p,ps,es):
 if p<=0:return 0.,0.
 e=interp(p,ps,es)
 return 4*np.pi*r*r*e,-(e+p)*(m+4*np.pi*r**3*p)/(r*(r-2*m))
@njit(cache=True)
def star(pc,ps,es,dr,smooth=True):
 r=1e-6;p=pc*1.3234e-6;m=4*np.pi*r**3*interp(p,ps,es)/3
 while p>1e-20*1.3234e-6 and r<100:
  a,b=rhs(r,m,p,ps,es);c,d=rhs(r+dr/2,m+dr*a/2,p+dr*b/2,ps,es);e,f=rhs(r+dr/2,m+dr*c/2,p+dr*d/2,ps,es);g,h=rhs(r+dr,m+dr*e,p+dr*f,ps,es)
  oldp=p;oldm=m
  m+=dr/6*(a+2*c+2*e+g);p+=dr/6*(b+2*d+2*f+h);r+=dr
  if smooth and p<=0:
   frac=oldp/(oldp-p);r-=dr*(1-frac);m=oldm+frac*(m-oldm)
  if not np.isfinite(m) or not np.isfinite(p) or 2*m/r>=.999:return np.nan,np.nan
 return m/1.4766,r
@njit(cache=True)
def stars(tab,centers,dr,smooth=True):
 ps=tab[:,1]*1.3234e-6;es=tab[:,2]*1.3234e-6
 out=np.empty((len(centers),3))
 for i in range(len(centers)):
  n=centers[i];pc=interp(n,tab[:,0],tab[:,1])
  m,r=star(pc,ps,es,dr,smooth);out[i]=np.array([n,m,r])
 return out

def curve(p,fine=False,legacy=False):
 if legacy:
  grid=np.round(np.arange(.08,1.20001,.02 if not fine else .005),4)
 else:grid=np.linspace(.08,1.8,100 if not fine else 345)
 co,err=core(p,grid)
 if len(co)!=len(grid):raise ValueError('core failed')
 tab=merge(BPS,co)
 if legacy:
  rows=tab[(tab[:,0]>=.02)&(tab[:,0]<=1.2)];st=max(1,len(rows)//(70 if fine else 30));centers=rows[::st,0];dr=.005 if fine else .02
 else:centers=np.linspace(.11,1.7,55 if not fine else 160);dr=.04 if not fine else .005
 c=stars(tab,centers,dr,not legacy)
 if not np.all(np.isfinite(c)):raise ValueError('TOV failed')
 return c,co,err

def observables(p,fine=False,legacy=False):
 nm=nuclear(p,legacy);c,co,err=curve(p,fine,legacy);im=np.argmax(c[:,1]);cs=c[:im+1]
 if len(cs)<2 or np.any(np.diff(cs[cs[:,1]>.5,1])<=0):raise ValueError('unstable curve')
 r14=np.interp(1.4,cs[:,1],cs[:,2]) if cs[-1,1]>=1.4 else np.nan
 return np.r_[nm,c[im,1],r14],c,co
TARGET=np.array([.16,-16,230,30.9,52.5,600]);SIGMA=np.array([.05,1,30,1.9,17.5,200])
def terms(obs):return np.r_[((obs[:6]-TARGET)/SIGMA)**2,(min(0.,obs[6]-2.08)/.07)**2,((obs[7]-12.7)/1.1)**2]
def vector(d):return np.array([d.get(n,0.) for n in NAMES])
