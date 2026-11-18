"""
End-to-end spot check: reproduce one sample's neutron-star M and R with the skill's
bundled pipeline and compare to the shipped ground truth (should be ~0% error).

Self-contained example for sample 0 (its Lagrangian + couplings are inlined). Paths are
repo-relative; override the repo root with P26_REPO if needed.
"""
import sys, json, time, os
from pathlib import Path
import numpy as np
T0 = time.time()

HERE = Path(__file__).resolve().parent
REPO  = Path(os.environ.get("P26_REPO", HERE.parents[1]))
TR    = REPO/'skill'/'rmf-toolkit'/'scripts'/'agentV1-tools'
GTDIR = HERE
for p in [TR, TR/'NMtoolsV1', TR/'TOVtools']: sys.path.insert(0, str(p))
from NMtoolsV1.analytic_layer import build_analytic_model
from TOVtools.tool_beta_eff import compile_beta_effective_model
from TOVtools.beta_eos import build_beta_equilibrium_eos_table
from TOVtools.tov_solver import build_eos_interpolator, integrate_tov_star
from TOVtools.hybrid_eos import read_bps_table, build_hybrid_eos_from_tables

SAMPLE = 0
term_texts = ["sp.Integer(1) * psibar_i_dslash_psi - P['m_N'] * psibar_psi","-sp.Rational(1,2) * P['m_sigma']**2 * s**2","-sp.Rational(1,3) * P['g2'] * s**3","-sp.Rational(1,4) * P['g3'] * s**4","-sp.Rational(1,2) * P['g4'] * s**2 * o**2","-sp.Rational(1,2) * P['g5'] * s**2 * r**2","-P['g6'] * s * r**2","-P['g7'] * s * o**2","+sp.Rational(1,2) * P['m_omega']**2 * o**2","+sp.Rational(1,4) * P['c3'] * o**4","+sp.Rational(1,2) * P['c4'] * o**2 * r**2","+sp.Rational(1,2) * P['m_rho']**2 * r**2","+sp.Rational(1,4) * P['d3'] * r**4","-sp.Rational(1,2) * P['m_a0']**2 * a**2","+sp.Rational(1,4) * P['b3'] * a**4","+sp.Rational(1,2) * P['b4'] * a**2 * r**2","+sp.Rational(1,2) * P['b5'] * s * a**2","+sp.Rational(1,2) * P['b6'] * a**2 * s**2","+sp.Rational(1,2) * P['b7'] * a**2 * o**2","+P['b8'] * a * r * o * s","+P['b9'] * a * r * o","-P['g_omega'] * o * psibar_gamma0_psi"]
params = {'m_N':939.0,'m_sigma':889.6250962607628,'m_omega':783.0,'m_rho':770.0,'m_a0':980.0,'g_omega':14.233526990555276,'g2':1379.4006071296847,'g3':5.5568215125475655,'g4':1.0392922850907635,'g5':0.9277672718190012,'g6':38.30213962874116,'g7':121.83700640963632,'c3':163.82140326142618,'c4':220.69650537554404,'d3':5.056194961778198,'b3':82.2573463223434,'b4':2.6437886168977425,'b5':116.83670628467347,'b6':211.28128734295873,'b7':0.1318618956702511,'b8':16.683375350494476,'b9':-9.829400023851724}

pk = [f"P['{k}']" for k in params]
analytic = build_analytic_model(pk, ['s','o','r','a'],
    ['psibar_i_dslash_psi','psibar_psi','psibar_gamma0_psi','psibar_gamma0_tau3_psi','psibar_tau3_psi'], term_texts)
compiled = compile_beta_effective_model(analytic, params)
rho_grid = np.unique(np.concatenate([np.linspace(0.08,0.3,80),np.linspace(0.305,1.0,70),np.linspace(1.05,4.0,60)]))
core = build_beta_equilibrium_eos_table(None,None,list(map(float,rho_grid)),compiled_model=compiled,params=params,field_guess=None)
core = [p for p in core if np.isfinite(p.pressure_total_mev_fm3) and np.isfinite(p.epsilon_total_mev_fm3)]
core.sort(key=lambda x: x.rho_b)
pc = [p.pressure_total_mev_fm3 for p in core if abs(p.rho_b-4.0) < 1e-9]
pc = pc[0] if pc else None

bps = read_bps_table(str(TR/'TOVtools'/'bps.dat'))
hyb = build_hybrid_eos_from_tables(bps, core, rho_min_core=0.08)
star = integrate_tov_star(build_eos_interpolator(hyb.merged_table), float(pc), dr_km=0.005, p_surface_mev_fm3=1e-12)
M_fix, R_fix = star.mass_msun, star.radius_km

gt = np.load(str(REPO/'data'/'gt'/'targets_100x6.npy'))[SAMPLE]
gtM, gtR = float(gt[4]), float(gt[5])
out = {'sample':SAMPLE, 'secs':round(time.time()-T0,1), 'pc_MeV_fm3':round(pc,2) if pc else None,
       'merged_rows':len(hyb.merged_table), 'bps_rows':len(bps),
       'M':round(M_fix,4), 'R':round(R_fix,4), 'gt_M':round(gtM,4), 'gt_R':round(gtR,4),
       'M_relerr_%':round(100*abs(M_fix-gtM)/max(abs(gtM),1e-9),3),
       'R_relerr_%':round(100*abs(R_fix-gtR)/max(abs(gtR),1e-9),3)}
print(json.dumps(out, indent=2))
