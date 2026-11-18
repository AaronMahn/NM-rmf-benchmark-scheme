

def main():
    """A4: numerical sensitivity of the six-vector and M_max to the gap-root grid, the core density
    grid and the TOV step. Writes convergence.json; scratch output is redirected to NNSTAR_VALIDATION_WORK/scratch.

    Usage:  python convergence_check.py
    """
    import json
    import sys
    from pathlib import Path

    import numpy as np

    sys.dont_write_bytecode = True
    CODE = Path(__file__).resolve().parent
    sys.path.insert(0, str(CODE))
    import run_a4 as A
    from enjl_model import ENJL

    HERE = A.HERE

    SCRATCH = A.vpaths.WORK / 'scratch' / 'convergence'
    SCRATCH.mkdir(parents=True, exist_ok=True)
    A.OUT_MR = SCRATCH

    CASES = {
        'baseline': dict(n_grid=3001, step=0.02, dr=0.005),
        'fine_gap_grid': dict(n_grid=9001, step=0.02, dr=0.005),
        'fine_core_grid': dict(n_grid=3001, step=0.01, dr=0.005),
        'fine_tov_step': dict(n_grid=3001, step=0.02, dr=0.001),
    }


    def one(name, n_grid, step, dr):
        A.RHO_GRID = np.unique(np.round(np.append(np.arange(0.08, 4.0500001, step), [3.95, 4.0, 4.05]), 10))
        A.DR_KM = dr
        model = ENJL(name, n_grid=n_grid)
        nm = A.nm_at_016(model)
        mr = A.mr_analysis(model, A.beta_table(model))
        return dict(E=nm['binding_energy_MeV'], P=nm['pressure_MeV_fm3'], K=nm['compressibility_MeV'],
                    Esym=nm['symmetry_energy_MeV'], M4=mr['M_at_rhoc'], R4=mr['R_at_rhoc'],
                    Mmax=mr['Mmax'], R14=mr['R14'])


    out = {}
    for name in ('eNJL1', 'eNJL3sr1', 'eNJL1msr1'):
        out[name] = {c: one(name, **kw) for c, kw in CASES.items()}
        b = out[name]['baseline']
        for c, v in out[name].items():
            print(name, f'{c:15s}', {k: (round(x, 5) if x is not None else None) for k, x in v.items()},
                  'max|d| vs baseline:', max(abs(v[k] - b[k]) for k in v if v[k] is not None and b[k] is not None), flush=True)
    (HERE / 'convergence.json').write_text(json.dumps(dict(cases=CASES, results=out), indent=1))


if __name__ == '__main__':
    main()
