#!/usr/bin/env python3
"""
run_paper_model.py — Paper → RMF model → validation pipeline
=============================================================
Reads a physics paper (local PDF or arXiv ID), extracts the RMF model
definition using an LLM, runs the full physics pipeline, then compares
computed results against the paper's reported values.

Usage
-----
    # From a local PDF:
    python skill/rmf-toolkit/scripts/run_paper_model.py --pdf /path/to/paper.pdf

    # From an arXiv ID:
    python skill/rmf-toolkit/scripts/run_paper_model.py --arxiv 2303.12345

    # Set the API key in the environment:
    ANTHROPIC_API_KEY=... python skill/rmf-toolkit/scripts/run_paper_model.py --arxiv 2303.12345

    # Skip the run-and-compare step (extraction + save JSON only):
    python skill/rmf-toolkit/scripts/run_paper_model.py --pdf paper.pdf --extract-only

    # Load a previously extracted definition and run/compare (skip re-extraction):
    python skill/rmf-toolkit/scripts/run_paper_model.py --definition results/MyModel_paper/mymodel_definition.json

Outputs (all under results/{MODEL_NAME}_paper/)
-----------------------------------------------
  {name}_definition.json     — extracted model definition (model_io format)
  {name}_paper_reported.json — NMP + NS observables extracted from paper
  {name}_nuclear_properties.json
  {name}_beta_eos.json
  {name}_hybrid_eos.json
  {name}_mr.json
  {name}_mr.png
  {name}_comparison_report.json
  {name}_comparison_plot.png
  {name}_run.log
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
TOOLS_DIR = SCRIPT_DIR / 'agentV1-tools'
for _path in (TOOLS_DIR, TOOLS_DIR / 'NMtoolsV1', TOOLS_DIR / 'TOVtools'):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize_scalar

from NMtoolsV1.nuclear_properties import (
    binding_energy_from_epsilon,
    pressure_from_epsilon,
    compressibility_from_epsilon,
    symmetry_energy_from_epsilon,
    symmetry_slope_from_epsilon,
)
from TOVtools.beta_eos import solve_beta_equilibrium_point_numeric
from TOVtools.hybrid_eos import read_bps_table, build_hybrid_eos_from_tables
from TOVtools.tov_solver import build_eos_interpolator, integrate_tov_star
from model_io import load_model_definition, build_model_from_definition, save_model_definition
from paper_extractor import extract_model_from_paper

BPS_PATH = TOOLS_DIR / 'TOVtools' / 'bps.dat'


def display_output_path(path):
    """Keep local output logs portable and free of machine-specific paths."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Paper → RMF model → validation')
    src = p.add_mutually_exclusive_group()
    src.add_argument('--pdf',        help='Path to local PDF file')
    src.add_argument('--arxiv',      help='arXiv ID (e.g. 2303.12345)')
    src.add_argument('--definition', help='Path to an existing definition JSON (skip extraction)')
    p.add_argument('--extract-only', action='store_true',
                   help='Only extract and save definition; skip run and compare')
    p.add_argument('--mr-sample', type=int, default=80,
                   help='MR central-density samples (default 80)')
    p.add_argument('--out', type=Path, default=REPO_ROOT / 'results',
                   help='Output root directory (default: repository results/)')
    args = p.parse_args()
    if args.mr_sample < 1:
        p.error('--mr-sample must be at least 1')
    return args


# ---------------------------------------------------------------------------
# Physics pipeline (same logic as run_model.py, returns dicts)
# ---------------------------------------------------------------------------

def run_physics_pipeline(m, log, mr_sample=80):
    """Run nuclear properties → beta EOS → hybrid EOS → MR.
    Returns (nm_dict, mr_list, hybrid)."""

    # Nuclear matter properties
    def be_snm(rho_b):
        return binding_energy_from_epsilon(m.epsilon_func, rho_b / 2.0, rho_b / 2.0, m.params['m_N'])
    rho0 = float(minimize_scalar(be_snm, bounds=(0.05, 0.25), method='bounded',
                                 options={'xatol': 1e-5, 'maxiter': 200}).x)
    nm = {
        'rho0_fm3':       rho0,
        'BE_rho0_MeV':    float(be_snm(rho0)),
        'P_rho0_MeV_fm3': float(pressure_from_epsilon(m.epsilon_func, rho0 / 2.0, rho0 / 2.0)),
        'K_rho0_MeV':     float(compressibility_from_epsilon(m.epsilon_func, rho0, m.params['m_N'])),
        'Esym_rho0_MeV':  float(symmetry_energy_from_epsilon(m.epsilon_func, rho0, m.params['m_N'])),
        'L_rho0_MeV':     float(symmetry_slope_from_epsilon(m.epsilon_func, rho0, m.params['m_N'])),
    }
    log('  rho0={rho0_fm3:.5f} BE={BE_rho0_MeV:.3f} K={K_rho0_MeV:.2f} Esym={Esym_rho0_MeV:.2f} L={L_rho0_MeV:.2f}'.format(**nm))

    # Beta EOS
    core_eos = []
    yp_guess, field_guess = 0.1, list(m.initial_field_guess)
    for i, rho_b in enumerate(m.rho_grid, 1):
        point, field_guess = solve_beta_equilibrium_point_numeric(
            m.compiled, m.params, rho_b,
            proton_fraction_guess=yp_guess, field_guess=field_guess,
        )
        yp_guess = point.rho_p / point.rho_b if point.rho_b > 0 else yp_guess
        core_eos.append(point)
        if i % 200 == 0 or i == 1 or i == len(m.rho_grid):
            log(f'  beta {i}/{len(m.rho_grid)} rho_b={rho_b:.4f} Yp={point.rho_p/point.rho_b:.4f}')

    # Hybrid EOS
    bps_cut = [x for x in read_bps_table(str(BPS_PATH)) if x.rho_b <= 0.08]
    hybrid = build_hybrid_eos_from_tables(bps_cut, core_eos, rho_min_core=0.08)

    # MR
    eos_interp = build_eos_interpolator(hybrid.merged_table)
    merged_rows = sorted(hybrid.merged_table, key=lambda x: x['rho_b'])
    core_rows = [r for r in merged_rows if 0.01 <= r['rho_b'] <= 1.2]
    stride = max(1, len(core_rows) // mr_sample)
    sampled = core_rows[::stride][:mr_sample]

    mr_results = []
    for row in sampled:
        star = integrate_tov_star(eos_interp, row['pressure_total_mev_fm3'],
                                  dr_km=0.001, p_surface_mev_fm3=1e-20)
        mr_results.append({
            'rho_c':      float(row['rho_b']),
            'pc_mev_fm3': float(row['pressure_total_mev_fm3']),
            'mass_msun':  float(star.mass_msun),
            'radius_km':  float(star.radius_km),
        })

    return nm, mr_results, hybrid, core_eos


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

def _pct(computed, reported):
    if reported is None or reported == 0:
        return None
    return (computed - reported) / abs(reported) * 100.0


def build_comparison_report(nm, mr_results, paper_reported, params, definition):
    masses_msun = [x['mass_msun'] for x in mr_results]
    radii_km    = [x['radius_km']  for x in mr_results]
    mmax_computed = max(masses_msun)

    # R at 1.4 Msun (interpolate)
    r14 = _interpolate_radius(mr_results, 1.4)
    r13 = _interpolate_radius(mr_results, 1.3)

    pr = paper_reported or {}

    report = {
        'model_name': definition.get('model_name', '?'),

        'parameters': {
            k: {'extracted': v} for k, v in params.items()
        },

        'nuclear_matter_properties': {
            'rho0_fm3':      {'computed': nm['rho0_fm3'],     'paper': pr.get('rho0_fm3'),     'delta_pct': _pct(nm['rho0_fm3'],     pr.get('rho0_fm3'))},
            'BE_rho0_MeV':   {'computed': nm['BE_rho0_MeV'],  'paper': pr.get('BE_rho0_MeV'),  'delta_pct': _pct(nm['BE_rho0_MeV'],  pr.get('BE_rho0_MeV'))},
            'K_rho0_MeV':    {'computed': nm['K_rho0_MeV'],   'paper': pr.get('K_rho0_MeV'),   'delta_pct': _pct(nm['K_rho0_MeV'],   pr.get('K_rho0_MeV'))},
            'Esym_rho0_MeV': {'computed': nm['Esym_rho0_MeV'],'paper': pr.get('Esym_rho0_MeV'),'delta_pct': _pct(nm['Esym_rho0_MeV'],pr.get('Esym_rho0_MeV'))},
            'L_rho0_MeV':    {'computed': nm['L_rho0_MeV'],   'paper': pr.get('L_rho0_MeV'),   'delta_pct': _pct(nm['L_rho0_MeV'],   pr.get('L_rho0_MeV'))},
        },

        'neutron_star_observables': {
            'Mmax_Msun': {'computed': mmax_computed,    'paper': pr.get('Mmax_Msun'),    'delta_pct': _pct(mmax_computed,    pr.get('Mmax_Msun'))},
            'R14_km':    {'computed': r14,              'paper': pr.get('R14_km'),        'delta_pct': _pct(r14,              pr.get('R14_km'))},
            'R13_km':    {'computed': r13,              'paper': pr.get('R13_km'),        'delta_pct': _pct(r13,              pr.get('R13_km'))},
        },

        'validation_status': None,  # filled below
    }

    # Compute overall pass/fail
    nmp_deltas = [v['delta_pct'] for v in report['nuclear_matter_properties'].values()
                  if v['delta_pct'] is not None]
    ns_deltas  = [v['delta_pct'] for v in report['neutron_star_observables'].values()
                  if v['delta_pct'] is not None]

    max_nmp = max((abs(d) for d in nmp_deltas), default=None)
    max_ns  = max((abs(d) for d in ns_deltas),  default=None)

    if max_nmp is None and max_ns is None:
        status = 'no_paper_values'
    elif (max_nmp or 0) < 2.0 and (max_ns or 0) < 3.0:
        status = 'PASS'
    elif (max_nmp or 0) < 5.0 and (max_ns or 0) < 8.0:
        status = 'WARN'
    else:
        status = 'FAIL'

    report['validation_status'] = status
    return report


def _interpolate_radius(mr_results, target_mass):
    """Linearly interpolate radius at a given target mass (Msun). Returns None if out of range."""
    pts = sorted(mr_results, key=lambda x: x['mass_msun'])
    for i in range(len(pts) - 1):
        m1, m2 = pts[i]['mass_msun'], pts[i+1]['mass_msun']
        r1, r2 = pts[i]['radius_km'],  pts[i+1]['radius_km']
        if m1 <= target_mass <= m2 or m2 <= target_mass <= m1:
            t = (target_mass - m1) / (m2 - m1) if m2 != m1 else 0
            return r1 + t * (r2 - r1)
    return None


# ---------------------------------------------------------------------------
# Comparison report printing
# ---------------------------------------------------------------------------

def print_report(report):
    print()
    print("=" * 65)
    print(f"  VALIDATION REPORT — {report['model_name']}   [{report['validation_status']}]")
    print("=" * 65)

    print("\n  Nuclear Matter Properties")
    print(f"  {'Property':<18} {'Paper':>10} {'Computed':>10} {'Δ%':>8}")
    print("  " + "-" * 50)
    for k, v in report['nuclear_matter_properties'].items():
        paper_s = f"{v['paper']:.4g}" if v['paper'] is not None else "—"
        comp_s  = f"{v['computed']:.4g}" if v['computed'] is not None else "—"
        delt_s  = f"{v['delta_pct']:+.2f}%" if v['delta_pct'] is not None else "—"
        flag    = "  ⚠" if v['delta_pct'] is not None and abs(v['delta_pct']) > 2 else ""
        print(f"  {k:<18} {paper_s:>10} {comp_s:>10} {delt_s:>8}{flag}")

    print("\n  Neutron Star Observables")
    print(f"  {'Observable':<18} {'Paper':>10} {'Computed':>10} {'Δ%':>8}")
    print("  " + "-" * 50)
    for k, v in report['neutron_star_observables'].items():
        paper_s = f"{v['paper']:.4g}" if v['paper'] is not None else "—"
        comp_s  = f"{v['computed']:.4g}" if v['computed'] is not None else "—"
        delt_s  = f"{v['delta_pct']:+.2f}%" if v['delta_pct'] is not None else "—"
        flag    = "  ⚠" if v['delta_pct'] is not None and abs(v['delta_pct']) > 5 else ""
        print(f"  {k:<18} {paper_s:>10} {comp_s:>10} {delt_s:>8}{flag}")

    print()


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

def make_comparison_plot(mr_results, paper_reported, report, out_path):
    masses = [x['mass_msun'] for x in mr_results]
    radii  = [x['radius_km']  for x in mr_results]
    mmax   = max(masses)
    pr     = paper_reported or {}

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(radii, masses, lw=2, color='royalblue',
            label=f"{report['model_name']} (computed, Mmax={mmax:.3f} M☉)")

    # Mark paper Mmax if available
    if pr.get('Mmax_Msun'):
        ax.axhline(pr['Mmax_Msun'], color='tomato', ls='--', lw=1.2,
                   label=f"Paper Mmax = {pr['Mmax_Msun']:.3f} M☉")

    # Mark paper R1.4 if available
    if pr.get('R14_km'):
        ax.axvline(pr['R14_km'], color='forestgreen', ls=':', lw=1.2,
                   label=f"Paper R₁.₄ = {pr['R14_km']:.2f} km")

    # Computed R1.4 dot
    r14_c = report['neutron_star_observables']['R14_km']['computed']
    if r14_c:
        ax.plot(r14_c, 1.4, 'o', color='royalblue', ms=7, zorder=5,
                label=f"Computed R₁.₄ = {r14_c:.2f} km")

    ax.set_xlim(8.0, 18.0)
    ax.set_ylim(0.0, max(masses) * 1.1)
    ax.set_xlabel('R [km]', fontsize=12)
    ax.set_ylabel('M [M$_\\odot$]', fontsize=12)
    ax.set_title(f"{report['model_name']} — computed vs. paper  [{report['validation_status']}]",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- EXTRACTION PHASE ----
    if args.definition:
        # Skip extraction; load existing definition
        def_path = Path(args.definition)
        definition = load_model_definition(def_path)
        paper_reported = {}
        print(f"Loaded definition from {def_path}", flush=True)
    else:
        # Extract from paper
        result = extract_model_from_paper(
            pdf_path  = args.pdf,
            arxiv_id  = args.arxiv,
            verbose   = True,
        )
        definition     = result.model_definition
        paper_reported = result.paper_reported

        if result.extraction_notes:
            print(f"\n  Extraction notes: {result.extraction_notes}", flush=True)

    name = str(definition.get('model_name', 'unknown_model'))
    name_lower = ''.join(char.lower() if char.isalnum() else '_' for char in name).strip('_')
    if not name_lower:
        name_lower = 'unknown_model'

    # ---- OUTPUT DIR ----
    out_dir = args.out.expanduser().resolve() / f'{name_lower}_paper'
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f'{name_lower}_run.log'
    log_path.write_text('', encoding='utf-8')

    def log(*parts):
        line = ' '.join(str(x) for x in parts)
        print(line, flush=True)
        with log_path.open('a', encoding='utf-8') as fh:
            fh.write(line + '\n')

    # Save definition and paper_reported
    def_out = out_dir / f'{name_lower}_definition.json'
    save_model_definition(definition, def_out)
    log(f'saved definition → {display_output_path(def_out)}')

    pr_out = out_dir / f'{name_lower}_paper_reported.json'
    pr_out.write_text(json.dumps(paper_reported, indent=2), encoding='utf-8')
    log(f'saved paper_reported → {display_output_path(pr_out)}')

    if paper_reported:
        log('\nPaper-reported values:')
        for k, v in paper_reported.items():
            if v is not None:
                log(f'  {k} = {v}')

    if args.extract_only:
        print(f'\nExtraction complete. Results in: {out_dir}', flush=True)
        return

    # ---- PHYSICS PIPELINE ----
    log('\nBuilding model …')
    m = build_model_from_definition(definition)

    log('\n--- Nuclear matter properties ---')
    nm, mr_results, hybrid, core_eos = run_physics_pipeline(m, log, args.mr_sample)

    # Save intermediate files
    nm_path = out_dir / f'{name_lower}_nuclear_properties.json'
    nm_path.write_text(json.dumps(nm, indent=2), encoding='utf-8')

    beta_path = out_dir / f'{name_lower}_beta_eos.json'
    beta_path.write_text(json.dumps([x.to_dict() for x in core_eos], indent=2), encoding='utf-8')

    hybrid_path = out_dir / f'{name_lower}_hybrid_eos.json'
    hybrid_path.write_text(json.dumps(hybrid.to_dict(), indent=2), encoding='utf-8')

    mr_path = out_dir / f'{name_lower}_mr.json'
    mr_path.write_text(json.dumps(mr_results, indent=2), encoding='utf-8')

    # ---- COMPARISON ----
    log('\n--- Comparison ---')
    report = build_comparison_report(nm, mr_results, paper_reported, m.params, definition)
    print_report(report)

    report_path = out_dir / f'{name_lower}_comparison_report.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    log(f'saved report → {display_output_path(report_path)}')

    # ---- PLOTS ----
    # MR + comparison plot
    cmp_png = out_dir / f'{name_lower}_comparison_plot.png'
    make_comparison_plot(mr_results, paper_reported, report, cmp_png)
    log(f'saved comparison plot → {display_output_path(cmp_png)}')

    # Simple MR-only PNG
    mr_png = out_dir / f'{name_lower}_mr.png'
    masses = [x['mass_msun'] for x in mr_results]
    radii  = [x['radius_km']  for x in mr_results]
    plt.figure(figsize=(6, 5))
    plt.plot(radii, masses, lw=2)
    plt.xlim(8.0, 18.0)
    plt.xlabel('R [km]')
    plt.ylabel('M [M$_\\odot$]')
    plt.title(f'{name} M-R')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(mr_png, dpi=200)
    plt.close()

    log(f'\ndone — results in {display_output_path(out_dir)}')
    print(f'\nResults in: {out_dir}', flush=True)
    print(f'Validation: {report["validation_status"]}', flush=True)


if __name__ == '__main__':
    main()
