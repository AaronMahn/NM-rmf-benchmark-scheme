"""Recompute the M and R columns from the original benchmark targets."""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("P26_REPO", HERE.parents[1]))
TR    = REPO/'skill'/'rmf-toolkit'/'scripts'/'agentV1-tools'
TASKS = REPO/'data'/'tasks'
for p in [TR, TR/'NMtoolsV1', TR/'TOVtools']: sys.path.insert(0, str(p))
from NMtoolsV1.analytic_layer import build_analytic_model
from TOVtools.tool_beta_eff import compile_beta_effective_model
from TOVtools.beta_eos import build_beta_equilibrium_eos_table
from TOVtools.tov_solver import build_eos_interpolator, integrate_tov_star
from TOVtools.hybrid_eos import read_bps_table, build_hybrid_eos_from_tables

BPS = str(TR/'TOVtools'/'bps.dat')
BILINEARS = ['psibar_i_dslash_psi','psibar_psi','psibar_gamma0_psi','psibar_gamma0_tau3_psi','psibar_tau3_psi']
FIELDS = ['s','o','r','a']
bps = read_bps_table(BPS)

def parse_task(n):
    txt = (TASKS/f'{n}.txt').read_text()
    # split lagrangian vs couplings vs compute
    lag = txt.split('拉氏量为')[1].split('耦合系数为')[0]
    cpl = txt.split('耦合系数为')[1].split('基于此计算')[0]
    term_texts = []
    for line in lag.splitlines():
        m = re.match(r'^-\s+(.*\S)\s*$', line)
        if m and ('psibar' in m.group(1) or 'P[' in m.group(1) or '*' in m.group(1)):
            term_texts.append(m.group(1))
    params = {}
    for line in cpl.splitlines():
        m = re.match(r'^-\s+([A-Za-z]\w*)\s*=\s*([-+0-9.eE]+)\s*$', line)
        if m: params[m.group(1)] = float(m.group(2))
    return term_texts, params

def compute_MR(term_texts, params):
    pk = [f"P['{k}']" for k in params]
    analytic = build_analytic_model(pk, FIELDS, BILINEARS, term_texts)
    compiled = compile_beta_effective_model(analytic, params)
    rho_grid = np.unique(np.concatenate([np.linspace(0.08,0.3,80),np.linspace(0.305,1.0,70),np.linspace(1.05,4.0,60)]))
    core = build_beta_equilibrium_eos_table(None,None,list(map(float,rho_grid)),compiled_model=compiled,params=params,field_guess=None)
    core=[p for p in core if np.isfinite(p.pressure_total_mev_fm3) and np.isfinite(p.epsilon_total_mev_fm3)]
    core.sort(key=lambda x:x.rho_b)
    pc=[p.pressure_total_mev_fm3 for p in core if abs(p.rho_b-4.0)<1e-9]
    if not pc: return None,None,None
    pc=pc[0]
    hyb = build_hybrid_eos_from_tables(bps, core, rho_min_core=0.08)
    eos = build_eos_interpolator(hyb.merged_table)
    star = integrate_tov_star(eos, float(pc), dr_km=0.005, p_surface_mev_fm3=1e-12)
    return star.mass_msun, star.radius_km, pc

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "targets_100x6.original.npy")
    parser.add_argument("--out", type=Path, default=HERE / "targets_100x6.recal.npy")
    parser.add_argument("--report", type=Path, default=HERE / "recalibration_run_report.json")
    args = parser.parse_args()

    gt_dir = HERE.resolve()
    for label in ("out", "report"):
        target = getattr(args, label).expanduser().resolve()
        try:
            target.relative_to(gt_dir)
        except ValueError as error:
            raise ValueError(f"--{label} must write under harness/gt") from error
        setattr(args, label, target)

    original = np.load(args.input)
    if original.shape != (100, 6):
        raise ValueError(f"Expected original target shape (100, 6); got {original.shape}")
    regenerated = original.copy()
    rows = []
    t0 = time.time()
    for sample in range(100):
        try:
            terms, params = parse_task(sample)
            mass, radius, central_pressure = compute_MR(terms, params)
            if mass is None or radius is None or central_pressure is None:
                raise RuntimeError("no central-pressure point at rho=4.0 fm^-3")
            old_mass, old_radius = map(float, original[sample, 4:6])
            regenerated[sample, 4] = mass
            regenerated[sample, 5] = radius
            rows.append(
                {
                    "sample": sample,
                    "old_M": old_mass,
                    "new_M": float(mass),
                    "old_R": old_radius,
                    "new_R": float(radius),
                }
            )
        except (IndexError, KeyError, ValueError, RuntimeError, OverflowError, FloatingPointError) as error:
            rows.append({"sample": sample, "error": str(error)})

    if not np.array_equal(regenerated[:, :4], original[:, :4]):
        raise AssertionError("M-R recalibration changed a matter column")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, regenerated)
    successful = [row for row in rows if "new_M" in row]
    errors = [row for row in rows if "error" in row]
    summary = {
        "seconds": round(time.time() - t0, 1),
        "successful": len(successful),
        "errors": errors,
        "matter_columns_unchanged": True,
        "output": args.out.name,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if errors:
        print("Samples with errors require explicit handling by fix2276.py or review.")


if __name__ == "__main__":
    main()
