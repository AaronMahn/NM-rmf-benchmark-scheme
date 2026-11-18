#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TOOLKIT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[2]
TOOLS_DIR = SCRIPT_DIR / "agentV1-tools"
for path in (TOOLS_DIR, TOOLS_DIR / "NMtoolsV1", TOOLS_DIR / "TOVtools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_io import build_model_from_definition, load_model_definition
from NMtoolsV1.nuclear_properties import (
    binding_energy_from_epsilon,
    compressibility_from_epsilon,
    pressure_from_epsilon,
    symmetry_energy_from_epsilon,
    symmetry_slope_from_epsilon,
)
from TOVtools.beta_eos import solve_beta_equilibrium_point_numeric
from TOVtools.hybrid_eos import build_hybrid_eos_from_tables, read_bps_table
from TOVtools.tov_solver import build_eos_interpolator, integrate_tov_star

BPS_PATH = TOOLS_DIR / "TOVtools" / "bps.dat"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RMF nuclear-matter and neutron-star pipeline.")
    parser.add_argument("--model", required=True, help="Model JSON path or filename under the toolkit models/.")
    parser.add_argument("--out", required=True, help="Output directory; relative paths are rooted at the repository.")
    parser.add_argument("--mr-samples", type=int, default=160, help="Number of central-density samples for M-R.")
    return parser.parse_args()


def resolve_model_path(value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if not candidate.is_absolute():
        candidate = TOOLKIT_ROOT / candidate
        if candidate.is_file():
            return candidate.resolve()
        named = TOOLKIT_ROOT / "models" / f"{value}.json"
        if named.is_file():
            return named.resolve()
    raise FileNotFoundError(f"Model definition not found: {value}")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sample_rows(rows: list[dict], count: int) -> list[dict]:
    if count < 2:
        raise ValueError("--mr-samples must be at least 2")
    if len(rows) < 2:
        raise ValueError("The hybrid EOS has fewer than two core rows")
    n = min(count, len(rows))
    indices = sorted({round(i * (len(rows) - 1) / (n - 1)) for i in range(n)})
    return [rows[index] for index in indices]


def radius_at_mass(masses: list[float], radii: list[float], max_index: int, target: float) -> float:
    for index in range(max_index):
        m1, m2 = masses[index], masses[index + 1]
        if m1 <= target <= m2 or m2 <= target <= m1:
            if m1 == m2:
                return (radii[index] + radii[index + 1]) / 2.0
            fraction = (target - m1) / (m2 - m1)
            return radii[index] + fraction * (radii[index + 1] - radii[index])
    raise ValueError(f"The sampled stable branch does not bracket {target:.1f} Msun")


def run_pipeline(model_path: Path, output_dir: Path, mr_samples: int) -> dict:
    definition = load_model_definition(model_path)
    model = build_model_from_definition(definition)
    name = model.model_name
    stem = name.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    def binding_energy(rho_b: float) -> float:
        return binding_energy_from_epsilon(
            model.epsilon_func,
            rho_b / 2.0,
            rho_b / 2.0,
            model.params["m_N"],
        )

    from scipy.optimize import minimize_scalar

    saturation = minimize_scalar(
        binding_energy,
        bounds=(0.05, 0.25),
        method="bounded",
        options={"xatol": 1e-5, "maxiter": 200},
    )
    rho0 = float(saturation.x)
    nuclear = {
        "model_name": name,
        "rho0_fm3": rho0,
        "BE_rho0_MeV": float(binding_energy(rho0)),
        "P_rho0_MeV_fm3": float(
            pressure_from_epsilon(model.epsilon_func, rho0 / 2.0, rho0 / 2.0)
        ),
        "K_rho0_MeV": float(
            compressibility_from_epsilon(model.epsilon_func, rho0, model.params["m_N"])
        ),
        "Esym_rho0_MeV": float(
            symmetry_energy_from_epsilon(model.epsilon_func, rho0, model.params["m_N"])
        ),
        "L_rho0_MeV": float(
            symmetry_slope_from_epsilon(model.epsilon_func, rho0, model.params["m_N"])
        ),
    }
    write_json(output_dir / f"{stem}_nuclear_properties.json", nuclear)

    beta_rows = []
    proton_fraction = 0.1
    field_guess = list(model.initial_field_guess)
    for rho_b in model.rho_grid:
        point, field_guess = solve_beta_equilibrium_point_numeric(
            model.compiled,
            model.params,
            rho_b,
            proton_fraction_guess=proton_fraction,
            field_guess=field_guess,
        )
        if point.rho_b > 0.0:
            proton_fraction = point.rho_p / point.rho_b
        beta_rows.append(point.to_dict())
    write_json(output_dir / f"{stem}_beta_eos.json", beta_rows)

    bps = [row for row in read_bps_table(BPS_PATH) if row.rho_b <= 0.08]
    hybrid = build_hybrid_eos_from_tables(bps, beta_rows, rho_min_core=0.08)
    hybrid_dict = hybrid.to_dict()
    write_json(output_dir / f"{stem}_hybrid_eos.json", hybrid_dict)

    eos = build_eos_interpolator(hybrid.merged_table)
    core_rows = [
        row for row in hybrid.merged_table
        if row["segment"] == "core" and row["rho_b"] >= 0.08
    ]
    central_rows = sample_rows(core_rows, mr_samples)
    mr_curve = []
    for row in central_rows:
        star = integrate_tov_star(
            eos,
            float(row["pressure_total_mev_fm3"]),
            dr_km=0.001,
            p_surface_mev_fm3=1e-20,
        )
        mr_curve.append(
            {
                "rho_c_fm3": float(row["rho_b"]),
                "pc_MeV_fm3": float(row["pressure_total_mev_fm3"]),
                "M_msun": float(star.mass_msun),
                "R_km": float(star.radius_km),
            }
        )

    max_index = max(range(len(mr_curve)), key=lambda index: mr_curve[index]["M_msun"])
    max_row = mr_curve[max_index]
    masses = [row["M_msun"] for row in mr_curve]
    radii = [row["R_km"] for row in mr_curve]
    r14 = radius_at_mass(masses, radii, max_index, 1.4)
    summary = {
        "model_name": name,
        "rho0_fm3": nuclear["rho0_fm3"],
        "BE_rho0_MeV": nuclear["BE_rho0_MeV"],
        "Esym_rho0_MeV": nuclear["Esym_rho0_MeV"],
        "L_rho0_MeV": nuclear["L_rho0_MeV"],
        "K_rho0_MeV": nuclear["K_rho0_MeV"],
        "M_max_msun": max_row["M_msun"],
        "rho_c_at_M_max_fm3": max_row["rho_c_fm3"],
        "R_at_M_max_km": max_row["R_km"],
        "R_1p4_km": r14,
        "n_mr_samples": len(mr_curve),
    }
    write_json(output_dir / f"{stem}_mr_curve.json", mr_curve)
    write_json(output_dir / f"{stem}_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model)
    output_dir = Path(args.out).expanduser()
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    summary = run_pipeline(model_path, output_dir.resolve(), args.mr_samples)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
