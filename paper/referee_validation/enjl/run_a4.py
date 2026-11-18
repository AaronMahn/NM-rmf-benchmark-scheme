"""A4: validate the eNJL implementation against Pais et al. (2016) Table II and compute the
benchmark six-vectors (E/A, P, K, E_sym at 0.16 fm^-3; M, R at rho_c = 4.0 fm^-3) with the
released benchmark conventions (npe beta equilibrium, bundled BPS crust below 0.08 fm^-3,
RK4 TOV with dr = 0.005 km and p_surface = 1e-12 MeV fm^-3).

Usage:  python run_a4.py [model ...]     (default: all Table I models)
Writes into NNSTAR_VALIDATION_WORK/results/A4 (never into reference/).
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True       # never write __pycache__ into the released toolkit
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE.parent))
import vpaths
vpaths.check_toolkit()
TOOLS = vpaths.TOOLS
HERE = vpaths.result_dir('A4')       # all outputs go to NNSTAR_VALIDATION_WORK/results/A4
sys.path[:0] = [str(CODE), str(TOOLS), str(TOOLS / 'TOVtools'), str(TOOLS / 'NMtoolsV1')]   # TOOLS: beta_eos imports NMtoolsV1.* as a package

import numpy as np
from scipy.optimize import brentq

import beta_eos                      # released: electron EOS (m_e = 0.511 MeV)
import hybrid_eos                    # released: BPS crust + fixed 0.08 fm^-3 split
import nuclear_properties as nmp     # released: finite-difference E/A, P, K, E_sym, L
import tov_solver                    # released: RK4 TOV
from enjl_model import ENJL, TABLE1, TABLE2, TABLE3, PAPER_NAMES, HBARC, kf

RHO_NM = 0.16
RHO_C = 4.0
DR_KM = 0.005
P_SURF = 1e-12
RHO_CORE_MIN = 0.08
# Same core grid as A3's diagnostic of the released pipeline.
RHO_GRID = np.unique(np.round(np.append(np.arange(0.08, 4.0500001, 0.02), [3.95, 4.0, 4.05]), 10))
RHOC_GRID = np.unique(np.round(np.concatenate([np.arange(0.20, 1.60, 0.02), np.arange(1.6, 4.0001, 0.1)]), 10))

OUT_EOS = HERE / 'eos'
OUT_MR = HERE / 'mr'
OUT_6 = HERE / 'six_vectors'


# ---------------------------------------------------------------------------------------------
def table2_rows(model: ENJL):
    """Table II quantities (a) at the nominal rho0 = 0.148 fm^-3 printed in the paper and
    (b) at the P = 0 saturation point of the published (rounded) couplings."""
    t = TABLE2[model.name]
    at148 = dict(rho0=0.148, BA=model.EA(0.148), P=model.P_sym(0.148), K=model.K(0.148),
                 Esym=model.Esym_analytic(0.148), L=model.L_slope(0.148))
    sat = model.saturation()
    sat['P'] = model.P_sym(sat['rho0'])
    rows = []
    for label, d in (('at_rho_0.148', at148), ('at_P=0', sat)):
        row = dict(model=model.name, paper_name=PAPER_NAMES[model.name], evaluated=label)
        for k in ('rho0', 'BA', 'K', 'Esym', 'L'):
            row[k] = d[k]
            row[f'{k}_paper'] = t[k]
            row[f'd_{k}'] = d[k] - t[k]
        row['P_MeV_fm3'] = d['P']
        rows.append(row)
    return rows


def chiral_density(model: ENJL):
    """Symmetric matter: density where the chosen branch first has M < 1 MeV (m = 0 models)."""
    if model.m != 0.0:
        return None
    lo, hi = 0.15, 1.5
    if model.state(hi / 2, hi / 2)['M_MeV'] >= 1.0:
        return None
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if model.state(mid / 2, mid / 2)['M_MeV'] < 1.0:
            hi = mid
        else:
            lo = mid
    return hi


def thermo_checks(model: ENJL):
    """Eqs. (7)-(8) chemical potentials vs. numerical d eps / d rho_i; P vs rho^2 d(E/A)/drho."""
    out = []
    for rp, rn in ((0.08, 0.08), (0.03, 0.13), (0.2, 0.2), (0.05, 0.45)):
        st = model.state(rp, rn)
        h = 1e-6
        mun = (model.state(rp, rn + h)['eps'] - model.state(rp, rn - h)['eps']) / (2 * h)
        mup = (model.state(rp + h, rn)['eps'] - model.state(rp - h, rn)['eps']) / (2 * h)
        out.append(dict(rho_p=rp, rho_n=rn, dmu_n=st['mu_n'] - mun, dmu_p=st['mu_p'] - mup))
    r, h = 0.16, 1e-5
    p_fd = r * r * (model.EA(r + h) - model.EA(r - h)) / (2 * h)
    return dict(mu_checks=out, P_eq7_8=model.P_sym(r), P_rho2_dEdrho=p_fd)


def nm_at_016(model: ENJL):
    """The four nuclear-matter outputs with the released finite-difference definitions."""
    eps = model.eps            # (rho_n, rho_p) -> MeV fm^-3
    m0 = model.M0_MeV
    return dict(
        binding_energy_MeV=nmp.binding_energy_from_epsilon(eps, RHO_NM / 2, RHO_NM / 2, m0),
        binding_energy_MeV_mN939=nmp.binding_energy_from_epsilon(eps, RHO_NM / 2, RHO_NM / 2, 939.0),
        pressure_MeV_fm3=nmp.pressure_from_epsilon(eps, RHO_NM / 2, RHO_NM / 2),
        compressibility_MeV=nmp.compressibility_from_epsilon(eps, RHO_NM, m0),
        symmetry_energy_MeV=nmp.symmetry_energy_from_epsilon(eps, RHO_NM, m0),
        symmetry_slope_MeV=nmp.symmetry_slope_from_epsilon(eps, RHO_NM, m0),
        M_star_over_M0=model.state(RHO_NM / 2, RHO_NM / 2)['M_MeV'] / m0,
    )


# ---------------------------------------------------------------------------------------------
def beta_point(model: ENJL, rho_b: float):
    """npe matter: minimum of eps_total over the proton fraction among (mu_n = mu_p + mu_e) roots
    and the y_p = 0 edge (pure neutron matter, no electrons)."""
    def total(yp):
        rp = rho_b * yp
        st = model.state(rp, rho_b - rp)
        mue = beta_eos.electron_chemical_potential(rp)
        eps_e = beta_eos.electron_energy_density_mev_fm3(rp)
        eps = st['eps'] + eps_e
        P = st['P'] + beta_eos.electron_pressure_mev_fm3(rp)
        return st, mue, eps, P

    def resid(yp):
        st, mue, _, _ = total(yp)
        return st['mu_n'] - st['mu_p'] - mue

    ys = np.linspace(0.0, 0.5, 41)
    fs = [resid(y) for y in ys]
    cands = [0.0]
    for i in range(len(ys) - 1):
        if fs[i] * fs[i + 1] < 0:
            cands.append(brentq(resid, ys[i], ys[i + 1], xtol=1e-12, rtol=1e-12))
    best = None
    for y in cands:
        st, mue, eps, P = total(y)
        if best is None or eps < best[3] - 1e-12:
            best = (y, st, mue, eps, P)
    y, st, mue, eps, P = best
    rp = rho_b * y
    return dict(rho_b=rho_b, y_p=y, n_beta_roots=len(cands) - 1,
                beta_residual_MeV=(st['mu_n'] - st['mu_p'] - mue) if y > 0 else 0.0,
                mu_n=st['mu_n'], mu_p=st['mu_p'], mu_e=mue, M_MeV=st['M_MeV'],
                n_gap_roots=st['n_stationary'], E_deficit=st['E_deficit'], M_lowest_MeV=st['M_lowest_MeV'],
                kFn_over_Lambda=kf(rho_b - rp) / model.L,
                epsilon_total_mev_fm3=eps, pressure_total_mev_fm3=P)


def beta_table(model: ENJL):
    return [beta_point(model, float(r)) for r in RHO_GRID]


def eos_diagnostics(model: ENJL, tab):
    rho = np.array([p['rho_b'] for p in tab])
    P = np.array([p['pressure_total_mev_fm3'] for p in tab])
    e = np.array([p['epsilon_total_mev_fm3'] for p in tab])
    yp = np.array([p['y_p'] for p in tab])
    dP, de = np.diff(P), np.diff(e)
    cs2 = dP / de
    mid = 0.5 * (rho[1:] + rho[:-1])
    upto = rho <= RHO_C + 1e-12
    mupto = mid <= RHO_C + 1e-12

    def first(mask, x):
        idx = np.nonzero(mask)[0]
        return float(x[idx[0]]) if idx.size else None

    def last(mask, x):
        idx = np.nonzero(mask)[0]
        return float(x[idx[-1]]) if idx.size else None

    for p, c in zip(tab, np.append(cs2, np.nan)):
        p['cs2_forward'] = float(c)
    defi = np.array([p['E_deficit'] for p in tab])
    kfl = np.array([p['kFn_over_Lambda'] for p in tab])
    return dict(
        P_nonmonotonic_intervals=[[float(rho[i]), float(rho[i + 1])] for i in np.nonzero((dP <= 0) & mupto)[0]],
        max_cs2_upto_rhoc=float(np.nanmax(cs2[mupto])),
        first_acausal_rho=first((cs2 > 1) & mupto, mid),
        first_negative_cs2_rho=first((cs2 <= 0) & mupto, mid),
        first_rho_kFn_above_Lambda=first(kfl > 1, rho),
        first_rho_gap_root_not_Hartree_min=first(defi > 1e-6, rho),
        E_deficit_at_rhoc_MeV_fm3=float(np.interp(RHO_C, rho, defi)),
        first_rho_pure_neutron=first((yp == 0) & upto, rho),
        last_rho_with_protons=last((yp > 0) & upto, rho),
        y_p_at_rhoc=float(np.interp(RHO_C, rho, yp)),
        max_abs_beta_residual_MeV=float(np.max(np.abs([p['beta_residual_MeV'] for p in tab]))),
    )


def star(eos, rho, P, rhoc):
    pc = float(np.interp(rhoc, rho, P))
    s = tov_solver.integrate_tov_star(eos, pc, dr_km=DR_KM, p_surface_mev_fm3=P_SURF)
    return pc, s.mass_msun, s.radius_km


def mr_analysis(model: ENJL, tab):
    """TOV on the core table truncated at the first pressure maximum. Beyond it dP/drho <= 0, and
    the released merge (sort by P, keep strictly increasing P and eps) would splice post-maximum
    points of much larger eps into the low-pressure part of the table and corrupt every star."""
    Pall = np.array([p['pressure_total_mev_fm3'] for p in tab])
    bad = np.nonzero((np.diff(Pall) <= 0) | (Pall[1:] <= 0))[0]
    tab = tab[:bad[0] + 1] if bad.size else tab
    rho_top = tab[-1]['rho_b']
    out = dict(rho_core_top=rho_top, core_truncated=bool(bad.size))
    hyb = hybrid_eos.build_hybrid_eos_with_bundled_crust(tab, rho_min_core=RHO_CORE_MIN)
    eos = tov_solver.build_eos_interpolator(hyb.merged_table)
    rho = np.array([p['rho_b'] for p in tab])
    P = np.array([p['pressure_total_mev_fm3'] for p in tab])
    n_core_kept = sum(1 for r in hyb.merged_table if r['segment'] == 'core')
    if rho_top >= RHO_C:
        pc4, M4, R4 = star(eos, rho, P, RHO_C)
        dM = (star(eos, rho, P, RHO_C + 0.02)[1] - star(eos, rho, P, RHO_C - 0.02)[1]) / 0.04
    else:
        pc4 = M4 = R4 = dM = None
    curve = [(rc,) + star(eos, rho, P, rc) for rc in RHOC_GRID if rc <= rho_top]
    rc, pcs, Ms, Rs = (np.array(x) for x in zip(*curve))
    i = int(np.argmax(Ms))
    # parabolic refinement of M_max on the rho_c grid
    if 0 < i < len(Ms) - 1:
        a, b, c = np.polyfit(rc[i - 1:i + 2], Ms[i - 1:i + 2], 2)
        rc_max = -b / (2 * a)
        pc_m, Mmax, R_Mmax = star(eos, rho, P, rc_max)
    else:
        rc_max, Mmax, R_Mmax = rc[i], Ms[i], Rs[i]
    R14 = rc14 = None
    rising = Ms[:i + 1]
    if rising.size > 1 and rising[0] < 1.4 < rising.max():
        j = int(np.nonzero(rising >= 1.4)[0][0])
        rc14 = brentq(lambda x: star(eos, rho, P, x)[1] - 1.4, rc[j - 1], rc[j], xtol=1e-6)
        R14 = star(eos, rho, P, rc14)[2]
    with open(OUT_MR / f'{model.name}_mr.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rho_c_fm3', 'P_c_MeV_fm3', 'M_msun', 'R_km'])
        w.writerows(curve)
    out.update(M_at_rhoc=M4, R_at_rhoc=R4, P_c_at_rhoc=pc4, dM_drhoc_at_rhoc=dM,
               Mmax=float(Mmax), R_Mmax=float(R_Mmax), rhoc_Mmax=float(rc_max), R14=R14, rhoc14=rc14,
               Mmax_at_core_top=bool(i == len(Ms) - 1),
               rhoc_beyond_Mmax=bool(RHO_C > rc_max), hybrid_core_points_kept=n_core_kept,
               hybrid_core_points_total=int(np.sum(rho >= RHO_CORE_MIN)),
               match_pressure=hyb.match_pressure)
    return out


def status(model: ENJL, diag, mr):
    """Explicit validity status of (M, R) at rho_c = 4.0. Hard failures -> not a reference value."""
    hard, soft = [], []
    if diag['P_nonmonotonic_intervals']:
        hard.append(f"dP/drho <= 0 from {diag['P_nonmonotonic_intervals'][0][0]:.2f} fm^-3 (mechanically unstable; "
                    f"no TOV star with rho_c above {mr['rho_core_top']:.2f})")
    if mr['Mmax_at_core_top']:
        hard.append('M(rho_c) still rising at the top of the usable core: M_max not reached')
    if diag['first_acausal_rho'] is not None:
        hard.append(f"acausal (c_s^2 > 1) from {diag['first_acausal_rho']:.3f} fm^-3")
    if diag['first_rho_pure_neutron'] is not None and diag['first_rho_pure_neutron'] < RHO_C:
        hard.append(f"E_sym collapse: y_p = 0 (pure neutron matter) from {diag['first_rho_pure_neutron']:.2f} fm^-3")
    if mr['rhoc_beyond_Mmax']:
        soft.append(f"rho_c = 4.0 is past M_max (rho_c,max = {mr['rhoc_Mmax']:.3f}); unstable branch")
    if diag['first_rho_kFn_above_Lambda'] is not None:
        soft.append(f"k_F,n > Lambda from {diag['first_rho_kFn_above_Lambda']:.2f} fm^-3 (beyond cutoff)")
    if diag['first_rho_gap_root_not_Hartree_min'] is not None:
        soft.append(f"gap root no longer the Hartree minimum from {diag['first_rho_gap_root_not_Hartree_min']:.2f} fm^-3")
    s = 'invalid' if hard else ('formal (convention value, outside model validity)' if soft else 'valid')
    return dict(MR_status=s, hard_failures=hard, caveats=soft)


# ---------------------------------------------------------------------------------------------
def run_model(name: str):
    t0 = time.time()
    model = ENJL(name)
    res = dict(model=name, paper_name=PAPER_NAMES[name], couplings=TABLE1[name], M0_MeV=model.M0_MeV,
               Lambda_MeV=model.p['Lambda'], kF_equals_Lambda_sym_rho=2 * model.L ** 3 / (3 * math.pi ** 2),
               kF_equals_Lambda_pnm_rho=model.L ** 3 / (3 * math.pi ** 2))
    res['table2'] = table2_rows(model)
    res['rho_chi_sym'] = chiral_density(model)
    res['rho_chi_paper'] = TABLE3.get(name, {}).get('rho_chi')
    res['thermo'] = thermo_checks(model)
    res['nm_016'] = nm_at_016(model)
    tab = beta_table(model)
    res['eos_diag'] = eos_diagnostics(model, tab)
    with open(OUT_EOS / f'{name}_beta_eos.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(tab[0].keys()))
        w.writeheader()
        w.writerows(tab)
    res['mr'] = mr_analysis(model, tab)
    res['status'] = status(model, res['eos_diag'], res['mr'])
    res['paper_table3'] = TABLE3.get(name)
    res['seconds'] = time.time() - t0
    return res


def main(names):
    for d in (OUT_EOS, OUT_MR, OUT_6):
        d.mkdir(parents=True, exist_ok=True)
    results = []
    for n in names:
        r = run_model(n)
        results.append(r)
        print(f"{n:10s} {r['seconds']:6.1f}s  NM={[round(r['nm_016'][k], 4) for k in ('binding_energy_MeV', 'pressure_MeV_fm3', 'compressibility_MeV', 'symmetry_energy_MeV')]}"
              f"  M,R(4.0)={r['mr']['M_at_rhoc']},{r['mr']['R_at_rhoc']}  Mmax={r['mr']['Mmax']:.3f}@{r['mr']['rhoc_Mmax']:.3f}"
              f"  R14={r['mr']['R14']}  {r['status']['MR_status']}", flush=True)
    (HERE / 'validation.json').write_text(json.dumps(results, indent=1, default=float))

    with open(HERE / 'table2_validation.csv', 'w', newline='') as f:
        rows = [row for r in results for row in r['table2']]
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    cols = ['binding_energy_MeV', 'pressure_MeV_fm3', 'compressibility_MeV', 'symmetry_energy_MeV', 'M_msun', 'R_km']
    six = {}
    with open(HERE / 'six_vectors.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model'] + cols + ['M_formal', 'R_formal', 'MR_status', 'Mmax', 'R_Mmax', 'rhoc_Mmax', 'R14'])
        for r in results:
            nm, mr = r['nm_016'], r['mr']
            ok = r['status']['MR_status'] != 'invalid' and mr['M_at_rhoc'] is not None
            vec = [nm['binding_energy_MeV'], nm['pressure_MeV_fm3'], nm['compressibility_MeV'], nm['symmetry_energy_MeV'],
                   mr['M_at_rhoc'] if ok else float('nan'), mr['R_at_rhoc'] if ok else float('nan')]
            vec = [float('nan') if v is None else v for v in vec]
            np.save(OUT_6 / f"{r['model']}.npy", np.array(vec))
            six[r['model']] = dict(zip(cols, vec), M_formal=mr['M_at_rhoc'], R_formal=mr['R_at_rhoc'], **r['status'])
            w.writerow([r['model']] + vec + [mr['M_at_rhoc'], mr['R_at_rhoc'], r['status']['MR_status'],
                                              mr['Mmax'], mr['R_Mmax'], mr['rhoc_Mmax'], mr['R14']])
    (HERE / 'six_vectors.json').write_text(json.dumps(dict(columns=cols, conventions=dict(
        rho_NM=RHO_NM, rho_c=RHO_C, E_over_A='eps/rho - M0 (vacuum mass, 938.98-939.00 MeV)',
        K='9 (2 rho dE/drho + rho^2 d2E/drho2) = 9 dP/drho, released FD h=1e-4',
        Esym='released FD in delta, h=1e-3', matter='n p e (m_e = 0.511 MeV), no muons',
        crust='bundled bps.dat below 0.08 fm^-3, core at/above (hybrid_eos)', tov=f'RK4 dr={DR_KM} km, p_surface={P_SURF}',
        central_pressure='np.interp of beta-equilibrium P on the 0.02 fm^-3 core grid'), vectors=six),
        indent=1, default=float))


if __name__ == '__main__':
    main(sys.argv[1:] or list(TABLE1))
