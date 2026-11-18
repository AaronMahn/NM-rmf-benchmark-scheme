

def main():
    import sys,ast,json,time
    from pathlib import Path
    import numpy as np
    import rmf,vpaths
    AR=vpaths.TOOLS
    for p in [AR,AR/'NMtoolsV1',AR/'TOVtools']:sys.path.insert(0,str(p))
    from NMtoolsV1.analytic_layer import build_analytic_model
    from TOVtools.tool_beta_eff import compile_beta_effective_model
    from TOVtools.beta_eos import solve_beta_equilibrium_point_numeric
    from scipy.optimize import root
    defn=json.loads((vpaths.SIGMA6/'model.json').read_text())
    ns={'pk':list(defn['param_keys']),'terms':list(defn['lagrangian_terms']),'extra':list(defn['extra_symbol_names'])}
    ns['pk'].append("P['c4']");ns['terms'].append("sp.Rational(1,2)*P['c4']*o**2*r**2")
    a=build_analytic_model(ns['pk'],['s','o','r'],ns['extra'],ns['terms'])
    out={'point_checks':[],'resolution_checks':{}}
    for folder,name in [('A1','gaussian'),('A1','joint'),('A2','gaussian_lambda'),('A2','joint_lambda')]:
     j=json.load(open(vpaths.reference_or_result(folder,name+'.json')));p=rmf.vector(j['couplings']);pd=dict(j['couplings']);pd.update(m_N=939,m_omega=783,m_rho=770);cm=compile_beta_effective_model(a,pd)
     for n in [.08,.16,.6,1.2]:
      x,e,pr,err=rmf.solve(n,p,True,np.array([20.,15.,-2.,.07]))
      if not np.isfinite(e):raise RuntimeError('backend failed validation')
      pt,sol=solve_beta_equilibrium_point_numeric(cm,pd,n,field_guess=x[:3]);r=cm.eom_numeric(x[:3],n*(1-x[3]),n*x[3]);scale=max(n*rmf.H3,1.)
      out['point_checks'].append({'case':name,'density':n,'max_normalized_symbolic_EOM':float(np.max(np.abs(r))/scale),'energy_difference_MeVfm3':float(e-pt.epsilon_total_mev_fm3),'pressure_difference_MeVfm3':float(pr-pt.pressure_total_mev_fm3),'proton_fraction_difference':float(x[3]-pt.rho_p/n),'field_residual':float(err)})
     # Separate integration and density-grid refinement, same model parameters.
     co,_=rmf.core(p,np.linspace(.08,1.8,689));tab=rmf.merge(rmf.BPS,co);c=rmf.stars(tab,np.linspace(.11,1.7,320),.0025);im=np.argmax(c[:,1]);r14=np.interp(1.4,c[:im+1,1],c[:im+1,2]);of=j['fine_observables']
     out['resolution_checks'][name]={'refined_Mmax':float(c[im,1]),'refined_R14':float(r14),'delta_Mmax_from_fine':float(c[im,1]-of[6]),'delta_R14_from_fine':float(r14-of[7]),'core_density_step':.0025,'TOV_dr_km':.0025,'central_density_points':320}
     np.savetxt(vpaths.result_dir(folder)/(name+'_mr_refined.csv'),c,delimiter=',',header='central_density_fm-3,mass_msun,radius_km')
     print(name,out['resolution_checks'][name],flush=True)
    out['max_energy_difference']=max(abs(x['energy_difference_MeVfm3']) for x in out['point_checks']);out['max_pressure_difference']=max(abs(x['pressure_difference_MeVfm3']) for x in out['point_checks'])
    (vpaths.result_dir('A1')/'backend_validation.json').write_text(json.dumps(out,indent=2));print('max errors',out['max_energy_difference'],out['max_pressure_difference'],flush=True)


if __name__ == '__main__':
    main()
