

def main():
    """Cross-start polishing and raw MAP audit from independent posterior chains.
    Preserves pre-refinement outputs; never changes the sampled likelihood.
    """
    import json,numpy as np
    from pathlib import Path
    from scipy.optimize import minimize
    import rmf,mcmc,vpaths
    from fit import Objective,SCALE
    B=vpaths.WORK
    out={}
    for ext in [False,True]:
     folder=vpaths.result_dir('A2' if ext else 'A1');obj=Objective(True,ext);n=obj.n;cases=list(folder.glob('joint*.json'));candidates=[]
     for f in cases:
      if 'radio' in f.stem:continue
      j=json.loads(f.read_text());p=rmf.vector(j['couplings']);score=obj.evaluate(p)[0];candidates.append((score,p,str(f.name)))
     if not ext:
      for seed in [1234,5678]:
       j=np.load(B/'mcmc'/f'chain_{seed}.npz');c=j['chain'][4000:].reshape(-1,8);lp=j['log_prob'][4000:].ravel();a=c[:,0]*mcmc.QS[0];ms=c[:,7]*mcmc.QS[7];t=ms/a;score=-2*(lp-14*np.log(t)+np.log(a));idx=np.argmin(score);candidates.append((float(score[idx]),mcmc.top(c[idx]),f'chain_{seed}'))
     candidates.sort(key=lambda x:x[0]);score,p,source=candidates[0];print('best cross-start',ext,score,source,flush=True)
     res=minimize(obj,p[:n]/SCALE[:n],method='Nelder-Mead',options={'maxiter':5000,'xatol':1e-7,'fatol':1e-7})
     if res.fun<score:p=np.r_[res.x*SCALE[:n],[] if ext else [0.]]
     name='joint_lambda' if ext else 'joint';path=folder/(name+'.json');j=json.loads(path.read_text());(folder/(name+'_before_polish.json')).write_text(json.dumps(j,indent=2))
     s,o,t,c,co=obj.evaluate(p);sf,of,tf,cf,cof=obj.evaluate(p,True);j.update(couplings=dict(zip(rmf.NAMES,p)),observables=o.tolist(),score=s,terms=t.tolist(),fine_observables=of.tolist(),fine_score=sf,fine_terms=tf.tolist(),cross_start_source=source,cross_start_scores=[(x[0],x[2]) for x in candidates],polish_calls=obj.calls)
     path.write_text(json.dumps(j,indent=2));np.savetxt(folder/(name+'_mr.csv'),cf,delimiter=',',header='central_density_fm-3,mass_msun,radius_km');np.savetxt(folder/(name+'_eos.csv'),cof,delimiter=',',header='density_fm-3,pressure_MeVfm-3,energy_MeVfm-3');print(name,s,sf,of,flush=True)
     out[name]=j
    (vpaths.result_dir('A1')/'map_audit.json').write_text(json.dumps(out,indent=2))


if __name__ == '__main__':
    main()
