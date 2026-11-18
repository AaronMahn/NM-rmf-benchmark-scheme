

def main():
    """Parent Codex check: direct momentum quadrature vs analytic sea integrals and benchmark K."""
    import json,math,sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    import vpaths
    import numpy as np
    from scipy.integrate import quad
    from enjl_model import ENJL,F0,F1,HBARC
    B=vpaths.result_dir('A4');out={}
    for name in ['eNJL1','eNJL3sr1','eNJL1msr1']:
     model=ENJL(name);rows=[]
     for mass in [0.,.3,model.M0]:
      for k in [.8,model.L,3.]:
       a=quad(lambda p:p*p/np.sqrt(p*p+mass*mass)/np.pi**2,0,k,epsabs=1e-12)[0]
       e=quad(lambda p:p*p*np.sqrt(p*p+mass*mass)/np.pi**2,0,k,epsabs=1e-12)[0]
       rows.append({'M_fm-1':mass,'k_fm-1':k,'F0_error':float(F0(mass,k)-a),'F1_error':float(F1(mass,k)-e)})
     n=.16;h=1e-4;e0=model.EA(n);em=model.EA(n-h);ep=model.EA(n+h);K=9*(2*n*(ep-em)/(2*h)+n*n*(ep-2*e0+em)/h**2)
     saved=json.load(open(vpaths.reference_or_result('A4','six_vectors.json')))
     out[name]={'quadrature_checks':rows,'K_from_benchmark_difference':K,'K_from_pressure_derivative':model.K(n),'K_difference':float(K-model.K(n)),'Esym_fd_minus_analytic':float(model.Esym_fd(n)-model.Esym_analytic(n)),'vacuum_gap_residual_MeV':float(model.gap(model.M0,0,0)*HBARC)}
    (B/'independent_check.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
    assert max(abs(v[k]) for v in out.values() for k in ['K_difference','Esym_fd_minus_analytic'])<.002
    assert max(abs(row[k]) for v in out.values() for row in v['quadrature_checks'] for k in ['F0_error','F1_error'])<1e-10


if __name__ == '__main__':
    main()
