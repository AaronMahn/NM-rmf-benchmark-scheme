"""Regenerate benchmark observables with explicit status, without editing shipped GT."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
DEFAULT_TOOLS = REPO / 'skill/rmf-toolkit/scripts/agentV1-tools'


def parse_task(path):
    text = Path(path).read_text()
    lag = text.split('拉氏量为')[1].split('耦合系数为')[0]
    params_text = text.split('耦合系数为')[1].split('基于此计算')[0]
    terms = [m.group(1) for line in lag.splitlines()
             if (m := re.match(r'^-\s+(.*\S)\s*$', line))]
    params = {k: float(v) for k, v in re.findall(
        r'^-\s+([A-Za-z]\w*)\s*=\s*([-+0-9.eE]+)\s*$', params_text, re.M)}
    fields = [f for f in ('s', 'o', 'r', 'a') if re.search(rf'(?<!\w){f}(?!\w)', lag)]
    return terms, params, fields


def compute(sample, tools, grid_kind='published'):
    import numpy as np
    for p in (tools, tools/'NMtoolsV1', tools/'TOVtools'):
        sys.path.insert(0, str(p))
    from NMtoolsV1.analytic_layer import build_analytic_model
    from NMtoolsV1.nuclear_properties import (
        binding_energy_from_epsilon, pressure_from_epsilon,
        compressibility_from_epsilon, symmetry_energy_from_epsilon)
    from NMtoolsV1.numeric_layer import HBARC
    from TOVtools.tool_beta_eff import compile_beta_effective_model
    from TOVtools.beta_eos import solve_beta_equilibrium_point_numeric
    from TOVtools.hybrid_eos import build_hybrid_eos_with_bundled_crust
    from TOVtools.tov_solver import build_eos_interpolator, integrate_tov_star
    checked = (tools/'NMtoolsV1/solver_checks.py').exists()
    if checked:
        from NMtoolsV1.solver_checks import checked_field_solve
    else:
        from scipy.optimize import fsolve
    terms, params, fields = parse_task(REPO/'data/tasks'/f'{sample}.txt')
    analytic = build_analytic_model([f"P['{k}']" for k in params], fields,
        ['psibar_i_dslash_psi','psibar_psi','psibar_gamma0_psi',
         'psibar_gamma0_tau3_psi','psibar_tau3_psi'], terms)
    model = compile_beta_effective_model(analytic, params)
    seed = np.array([35.0 if f == 's' else 22.0 if f == 'o' else 0.0 for f in fields])
    residuals, bad_old_status = [], []
    cache = {}

    def epsilon(rn, rp):
        key = (float(rn), float(rp))
        if key not in cache:
            fn = lambda x: model.eom_numeric(x, rn, rp)
            if checked:
                x, diagnostic = checked_field_solve(fn, seed, fields, rn+rp,
                    lambda x: (model.mass_p_numeric(x), model.mass_n_numeric(x)))
                residuals.append(diagnostic['relative_residual'])
            else:
                x, info, ier, message = fsolve(fn, seed, xtol=1e-10, maxfev=20000, full_output=True)
                norm = float(np.max(np.abs(fn(x)))/max((rn+rp)*HBARC**3, 1.0))
                residuals.append(norm)
                if ier != 1 or norm > 1e-8:
                    bad_old_status.append({'ier':int(ier),'relative_residual':norm})
            cache[key] = float(model.energy_numeric(x, rn, rp, params))
        return cache[key]

    result = {'sample':sample, 'status':'failed', 'checked':checked, 'grid':grid_kind,
              'active_fields':fields}
    try:
        matter = [binding_energy_from_epsilon(epsilon, .08, .08, params['m_N']),
                  pressure_from_epsilon(epsilon, .08, .08),
                  compressibility_from_epsilon(epsilon, .16, params['m_N']),
                  symmetry_energy_from_epsilon(epsilon, .16, params['m_N'])]
        result.update(matter=matter, max_matter_field_residual=max(residuals),
                      unchecked_matter_failures=bad_old_status)
    except Exception as exc:
        result['matter_error'] = f'{type(exc).__name__}: {exc}'
    if grid_kind == 'uniform':
        grid = np.round(np.arange(.08, 4.00001, .02), 10)
    else:
        grid = np.unique(np.r_[np.linspace(.08,.3,80),
                              np.linspace(.305,1,140 if sample in (22,76) else 70),
                              np.linspace(1.05,4,80 if sample in (22,76) else 60)])
    points, x0, yp = [], None, .1
    failed_densities = []
    try:
        for density in grid:
            try:
                point, x0 = solve_beta_equilibrium_point_numeric(
                    model, params, float(density), proton_fraction_guess=yp, field_guess=x0)
            except Exception as exc:
                failed_densities.append({'rho_b':float(density),'error':f'{type(exc).__name__}: {exc}'})
                # Do not skip a failed equation and silently interpolate over it.
                raise
            yp = point.rho_p/point.rho_b
            points.append(point)
        central_pressure = float(points[-1].pressure_total_mev_fm3)
        if not np.isfinite(central_pressure) or central_pressure <= 0:
            raise RuntimeError('nonpositive/nonfinite central pressure')
        hybrid = build_hybrid_eos_with_bundled_crust(points, rho_min_core=.08)
        star = integrate_tov_star(build_eos_interpolator(hybrid.merged_table), central_pressure,
                                  dr_km=.005, p_surface_mev_fm3=1e-12)
        mr = [float(star.mass_msun), float(star.radius_km)]
        if not np.all(np.isfinite(mr)) or min(mr) <= 0 or mr[1] >= 100:
            raise RuntimeError('invalid TOV endpoint')
        nonmonotonic = bool(np.any(np.diff([p.pressure_total_mev_fm3 for p in points])<=0))
        if nonmonotonic:
            result['formal_mr'] = mr
            result['mr_error'] = 'nonmonotonic pressure-density core; formal TOV value is not verified GT'
        else:
            result['mr'] = mr
        result.update(central_pressure=central_pressure,
                      max_beta_residual=max(abs(p.delta_beta) for p in points),
                      beta_grid_points=len(points),
                      nonmonotonic_core=nonmonotonic)
    except Exception as exc:
        result['mr_error'] = f'{type(exc).__name__}: {exc}'
        result['failed_densities'] = failed_densities
    if 'matter' in result and 'mr' in result:
        result['vector'] = result['matter'] + result['mr']
        result['status'] = 'ok'
    elif result.get('nonmonotonic_core'):
        result['status'] = 'nonmonotonic_core'
    gt = np.load(REPO/'data/gt/targets_100x6.npy')[sample]
    result['shipped_gt'] = gt.tolist()
    if 'vector' in result:
        result['delta'] = (np.asarray(result['vector'])-gt).tolist()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', type=int)
    parser.add_argument('--ids', default=','.join(map(str,range(100))))
    parser.add_argument('--tools', type=Path, default=DEFAULT_TOOLS)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--grid', choices=['published','uniform'], default='published')
    args = parser.parse_args()
    if args.sample is not None:
        started=time.monotonic()
        try:
            row=compute(args.sample,args.tools.resolve(),args.grid)
        except Exception as exc:
            row={'sample':args.sample,'status':'failed','error':f'{type(exc).__name__}: {exc}'}
        row['seconds']=time.monotonic()-started
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(row,indent=2,allow_nan=False))
        return
    args.out.mkdir(parents=True,exist_ok=True)
    source=REPO/'data/gt/targets_100x6.npy'
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
             PYTHONPYCACHEPREFIX=str(args.out/'pycache'))
    def worker(sample):
        output=args.out/f'task_{sample}.json'
        command=[sys.executable,str(Path(__file__).resolve()),'--sample',str(sample),
                 '--tools',str(args.tools),'--out',str(output),'--grid',args.grid]
        try:
            p=subprocess.run(command,env=env,capture_output=True,text=True,timeout=args.timeout)
            if p.returncode or not output.exists():
                raise RuntimeError(p.stderr[-1500:])
            return json.loads(output.read_text())
        except subprocess.TimeoutExpired:
            row={'sample':sample,'status':'timeout','seconds':args.timeout}
        except Exception as exc:
            row={'sample':sample,'status':'failed','error':str(exc)}
        output.write_text(json.dumps(row,indent=2))
        return row
    ids=[int(x) for x in args.ids.split(',')]
    rows=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(worker,n):n for n in ids}
        for job in as_completed(jobs):
            row=job.result();rows.append(row)
            print(json.dumps({k:row[k] for k in ('sample','status','seconds') if k in row}),flush=True)
    rows.sort(key=lambda x:x['sample'])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    (args.out/'audit.json').write_text(json.dumps({'shipped_gt_sha256':before,'rows':rows},indent=2))
    flat=[]
    for r in rows:
        f={k:v for k,v in r.items() if not isinstance(v,(list,dict))}
        for i,value in enumerate(r.get('delta',[])):f[f'delta_{i}']=value
        flat.append(f)
    keys=sorted(set().union(*(r.keys() for r in flat)))
    with (args.out/'audit.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=keys);w.writeheader();w.writerows(flat)
    import numpy as np
    working=np.full((100,6),np.nan)
    for row in rows:
        if 'matter' in row:working[row['sample'],:4]=row['matter']
        if 'mr' in row:working[row['sample'],4:]=row['mr']
    np.save(args.out/'audited_targets.npy',working)


if __name__=='__main__':
    main()
