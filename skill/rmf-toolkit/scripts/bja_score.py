#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
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

CATALOGUE_PATH = TOOLKIT_ROOT / "constraints" / "catalogue.json"
BPS_PATH = TOOLS_DIR / "TOVtools" / "bps.dat"


def load_catalogue(path: Path = CATALOGUE_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def joint_mr_likelihood(observables: dict, constraint: dict) -> float:
    """Reserved interface for a future non-Gaussian joint M-R likelihood."""
    raise NotImplementedError("Joint M-R likelihood is intentionally deferred.")


def _nuclear_observables(model) -> dict[str, float]:
    def binding_energy(rho_b: float) -> float:
        return binding_energy_from_epsilon(
            model.epsilon_func,
            rho_b / 2.0,
            rho_b / 2.0,
            model.params["m_N"],
        )

    from scipy.optimize import minimize_scalar

    rho0 = float(
        minimize_scalar(
            binding_energy,
            bounds=(0.05, 0.25),
            method="bounded",
            options={"xatol": 1e-5, "maxiter": 200},
        ).x
    )
    return {
        "rho0": rho0,
        "BE": float(binding_energy(rho0)),
        "P": float(pressure_from_epsilon(model.epsilon_func, rho0 / 2.0, rho0 / 2.0)),
        "K": float(compressibility_from_epsilon(model.epsilon_func, rho0, model.params["m_N"])),
        "Esym": float(symmetry_energy_from_epsilon(model.epsilon_func, rho0, model.params["m_N"])),
        "L": float(symmetry_slope_from_epsilon(model.epsilon_func, rho0, model.params["m_N"])),
        "m_sigma": float(model.params["m_sigma"]),
    }


def _radius_at_mass(curve: list[tuple[float, float]], target_mass: float) -> float:
    mass_ordered = sorted(curve, key=lambda row: row[1])
    for (r1, m1), (r2, m2) in zip(mass_ordered, mass_ordered[1:]):
        if m1 <= target_mass <= m2:
            if m1 == m2:
                return (r1 + r2) / 2.0
            fraction = (target_mass - m1) / (m2 - m1)
            return r1 + fraction * (r2 - r1)
    raise ValueError(f"The M-R curve does not bracket {target_mass:g} Msun")


def _mr_observables(
    model,
    sample_stride: int = 70,
    tov_step_km: float = 0.005,
) -> tuple[dict[str, float], list[dict]]:
    core_rows = []
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
        core_rows.append(point)

    bps = [row for row in read_bps_table(BPS_PATH) if row.rho_b <= 0.08]
    hybrid = build_hybrid_eos_from_tables(bps, core_rows, rho_min_core=0.08)
    eos = build_eos_interpolator(hybrid.merged_table)
    rows = [
        row for row in sorted(hybrid.merged_table, key=lambda row: row["rho_b"])
        if 0.02 <= row["rho_b"] <= 1.2
    ]
    stride = max(1, len(rows) // sample_stride)
    curve = []
    for row in rows[::stride]:
        star = integrate_tov_star(
            eos,
            row["pressure_total_mev_fm3"],
            dr_km=tov_step_km,
            p_surface_mev_fm3=1e-20,
        )
        curve.append((float(star.radius_km), float(star.mass_msun)))
    if not curve:
        raise RuntimeError("The M-R pipeline produced no stellar models")
    observables = {"Mmax": max(mass for _, mass in curve)}
    for mass in (1.4, 2.08):
        try:
            observables["R14" if mass == 1.4 else "R208"] = _radius_at_mass(curve, mass)
        except ValueError:
            continue
    return observables, [
        {"radius_km": radius, "mass_msun": mass}
        for radius, mass in curve
    ]


def evaluate_definition(
    definition: dict,
    constraint_set: dict,
    evaluation_settings: dict | None = None,
) -> tuple[dict, list[dict]]:
    settings = evaluation_settings or {}
    has_mr = any(
        constraint_set[key]["quantity"] in {"Mmax", "R14", "R208"}
        for key in constraint_set
    )
    model_definition = copy.deepcopy(definition)
    if has_mr:
        model_definition["rho_beta_min"] = 0.08
        model_definition["rho_beta_max"] = 1.2
        model_definition["rho_beta_step"] = float(settings.get("rho_beta_step", 0.005))
    model = build_model_from_definition(model_definition)
    observables = _nuclear_observables(model)
    curve = []
    if has_mr:
        mr, curve = _mr_observables(
            model,
            sample_stride=int(settings.get("mr_sample_stride", 70)),
            tov_step_km=float(settings.get("tov_dr_km", 0.005)),
        )
        observables.update(mr)
    return observables, curve


def score_observables(observables: dict, definitions: dict, selected: list[str]) -> dict:
    terms = {}
    total_chi2 = 0.0
    for key in selected:
        constraint = definitions[key]
        kind = constraint["type"]
        if kind == "joint_mr":
            chi2 = joint_mr_likelihood(observables, constraint)
            value = None
        else:
            quantity = constraint["quantity"]
            if quantity not in observables:
                raise ValueError(f"Missing observable {quantity!r} for constraint {key!r}")
            value = float(observables[quantity])
            target = float(constraint["central_value"])
            sigma = float(constraint["sigma"])
            if sigma <= 0.0:
                raise ValueError(f"Constraint {key!r} must have a positive sigma")
            delta = max(0.0, target - value) if kind == "one_sided_lower_bound" else value - target
            z = delta / sigma
            chi2 = z * z
        terms[key] = {
            "type": kind,
            "quantity": constraint["quantity"],
            "value": value,
            "central_value": float(constraint["central_value"]),
            "sigma": float(constraint["sigma"]),
            "chi2": chi2,
            "source_reference": constraint["source_reference"],
        }
        total_chi2 += chi2
    return {
        "terms": terms,
        "total_chi2": total_chi2,
        "total_loglike": -0.5 * total_chi2,
        "relative_weight": math.exp(-0.5 * total_chi2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score RMF observables against a constraint set.")
    parser.add_argument("--model", default="models/tm1_paper.json", help="Model JSON path.")
    parser.add_argument("--constraints", default="standard_nm", help="Constraint-set key.")
    parser.add_argument("--catalogue", type=Path, default=CATALOGUE_PATH, help="Constraint catalogue JSON.")
    parser.add_argument("--out", type=Path, default=None, help="Optional output JSON path.")
    parser.add_argument("--list-constraints", action="store_true", help="List constraint sets and entries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalogue = load_catalogue(args.catalogue)
    if args.list_constraints:
        print(json.dumps(catalogue, indent=2))
        return
    if args.constraints not in catalogue["sets"]:
        raise ValueError(f"Unknown constraint set {args.constraints!r}; use --list-constraints")
    selected = catalogue["sets"][args.constraints]
    constraints = catalogue["constraints"]
    definition_path = Path(args.model).expanduser()
    if not definition_path.is_file():
        definition_path = TOOLKIT_ROOT / definition_path
    if not definition_path.is_file():
        definition_path = TOOLKIT_ROOT / "models" / f"{args.model}.json"
    if not definition_path.is_file():
        raise FileNotFoundError(f"Model definition not found: {args.model}")
    observables, curve = evaluate_definition(
        load_model_definition(definition_path),
        {key: constraints[key] for key in selected},
        catalogue.get("evaluation", {}).get(args.constraints),
    )
    result = {
        "model": definition_path.stem,
        "constraint_set": args.constraints,
        "observables": observables,
        "score": score_observables(observables, constraints, selected),
        "mr_curve": curve,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        output_path = args.out.expanduser()
        if not output_path.is_absolute():
            output_path = REPO_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
