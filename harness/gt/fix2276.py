"""Recompute the two M-R rows that need skip-and-continue beta-EOS solving."""
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("P26_REPO", HERE.parents[1]))
TR = REPO/'skill'/'rmf-toolkit'/'scripts'/'agentV1-tools'
GTDIR = HERE
TASKS = REPO/'data'/'tasks'
for p in [TR, TR/'NMtoolsV1', TR/'TOVtools']: sys.path.insert(0, str(p))
from NMtoolsV1.analytic_layer import build_analytic_model
from TOVtools.tool_beta_eff import compile_beta_effective_model
from TOVtools.beta_eos import solve_beta_equilibrium_point_numeric
from TOVtools.tov_solver import build_eos_interpolator, integrate_tov_star
from TOVtools.hybrid_eos import read_bps_table, build_hybrid_eos_from_tables

bps = read_bps_table(str(TR/'TOVtools'/'bps.dat'))
BIL = ['psibar_i_dslash_psi','psibar_psi','psibar_gamma0_psi','psibar_gamma0_tau3_psi','psibar_tau3_psi']
FIELDS = ['s','o','r','a']

def parse(n):
    txt = (TASKS/f'{n}.txt').read_text()
    lag = txt.split('拉氏量为')[1].split('耦合系数为')[0]
    cpl = txt.split('耦合系数为')[1].split('基于此计算')[0]
    tt = [re.match(r'^-\s+(.*\S)\s*$', l).group(1) for l in lag.splitlines()
          if re.match(r'^-\s+(.*\S)\s*$', l) and ("psibar" in l or "P[" in l or '*' in l)]
    pp = {}
    for l in cpl.splitlines():
        m = re.match(r'^-\s+([A-Za-z]\w*)\s*=\s*([-+0-9.eE]+)\s*$', l)
        if m: pp[m.group(1)] = float(m.group(2))
    return tt, pp

def robust_core(compiled, params, rho_grid):
    out=[]; yp=0.1; x=None; skipped=[]
    for rho_b in rho_grid:
        try:
            pt,x = solve_beta_equilibrium_point_numeric(compiled, params, float(rho_b), proton_fraction_guess=yp, field_guess=x)
            if math.isfinite(pt.pressure_total_mev_fm3) and math.isfinite(pt.epsilon_total_mev_fm3):
                out.append(pt); yp = pt.rho_p/pt.rho_b if pt.rho_b>0 else yp
            else:
                skipped.append(round(float(rho_b),4))
        except (ValueError, RuntimeError, OverflowError, FloatingPointError):
            skipped.append(round(float(rho_b),4))
    return out, skipped

def compute_MR(n):
    tt,pp = parse(n)
    pk = [f"P['{k}']" for k in pp]
    analytic = build_analytic_model(pk, FIELDS, BIL, tt)
    compiled = compile_beta_effective_model(analytic, pp)
    rho_grid = np.unique(np.concatenate([np.linspace(0.08,0.3,80),np.linspace(0.305,1.0,140),np.linspace(1.05,4.0,80)]))
    core, skipped = robust_core(compiled, pp, rho_grid)
    core = sorted(core, key=lambda x: x.rho_b)
    pc = [p.pressure_total_mev_fm3 for p in core if abs(p.rho_b-4.0) < 1e-9]
    if not pc:
        return None, None, len(skipped)
    hyb = build_hybrid_eos_from_tables(bps, core, rho_min_core=0.08)
    star = integrate_tov_star(build_eos_interpolator(hyb.merged_table), float(pc[0]), dr_km=0.005, p_surface_mev_fm3=1e-12)
    return star.mass_msun, star.radius_km, len(skipped)

recalibrated_path = GTDIR/'targets_100x6.recal.npy'
g = np.load(str(recalibrated_path))
if g.shape != (100, 6):
    raise ValueError(f"Expected target shape (100, 6); got {g.shape}")
for n in (22, 76):
    M, R, n_skipped = compute_MR(n)
    if M is None:
        raise RuntimeError(f"Sample {n}: no central-pressure point at rho=4.0 fm^-3")
    g[n,4] = M; g[n,5] = R
    print(f"wrote s{n}: M={M} R={R}  (densities skipped: {n_skipped})")
np.save(str(recalibrated_path), g)
original = np.load(str(GTDIR/'targets_100x6.original.npy'))
if not np.array_equal(original[:, :4], g[:, :4]):
    raise AssertionError("M-R recalibration changed a matter column")
print("recalibrated M/R rows written; matter columns remain bit-identical")

shipped = np.load(str(REPO/'data'/'gt'/'targets_100x6.npy'))
relative = np.abs(g[:, 4:6] - shipped[:, 4:6]) / np.maximum(np.abs(shipped[:, 4:6]), 1e-12)
max_relative = float(np.max(relative))
print(f"maximum relative difference from shipped M/R: {max_relative:.3e}")
if max_relative > 1e-6:
    raise AssertionError(f"Regenerated M/R differs from shipped ground truth by {max_relative:.3e}")
