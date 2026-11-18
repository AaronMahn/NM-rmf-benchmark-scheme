#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
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

from bja_score import evaluate_definition, load_catalogue, score_observables
from model_io import load_model_definition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit selected RMF parameters to a constraint set.")
    parser.add_argument("--model", required=True, help="Model JSON path.")
    parser.add_argument("--constraints", default="standard_nm", help="Constraint-set key.")
    parser.add_argument("--free-params", required=True, help="Comma-separated parameter names to optimize.")
    parser.add_argument("--bounds", type=Path, default=None, help="JSON object mapping free parameters to [lower, upper].")
    parser.add_argument("--maxiter", type=int, default=100, help="Optimizer iteration limit.")
    parser.add_argument("--out", type=Path, default=Path("results/fit_result.json"), help="Output JSON path.")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    candidate = TOOLKIT_ROOT / path
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"Model definition not found: {value}")


def fit(
    definition: dict,
    constraints: dict,
    selected: list[str],
    free_params: list[str],
    bounds: dict[str, list[float]],
    maxiter: int,
    evaluation_settings: dict,
) -> dict:
    from scipy.optimize import minimize

    missing = [name for name in free_params if name not in definition["params"]]
    if missing:
        raise ValueError(f"Unknown free parameter(s): {', '.join(missing)}")
    missing_bounds = [name for name in free_params if name not in bounds]
    if missing_bounds:
        raise ValueError(f"Bounds required for: {', '.join(missing_bounds)}")
    parameter_bounds = []
    for name in free_params:
        pair = bounds[name]
        if len(pair) != 2 or float(pair[0]) >= float(pair[1]):
            raise ValueError(f"Invalid bounds for {name!r}: expected [lower, upper]")
        parameter_bounds.append((float(pair[0]), float(pair[1])))

    initial = [float(definition["params"][name]) for name in free_params]

    def objective(values: list[float]) -> float:
        candidate = copy.deepcopy(definition)
        candidate["params"].update(zip(free_params, map(float, values)))
        observables, _ = evaluate_definition(
            candidate,
            {key: constraints[key] for key in selected},
            evaluation_settings,
        )
        return float(score_observables(observables, constraints, selected)["total_chi2"])

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=parameter_bounds,
        options={"maxiter": maxiter},
    )
    best_definition = copy.deepcopy(definition)
    best_definition["params"].update(zip(free_params, map(float, result.x)))
    observables, curve = evaluate_definition(
        best_definition,
        {key: constraints[key] for key in selected},
        evaluation_settings,
    )
    score = score_observables(observables, constraints, selected)
    return {
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "constraint_set": selected,
        "free_params": free_params,
        "bounds": {name: bounds[name] for name in free_params},
        "best_params": {name: best_definition["params"][name] for name in free_params},
        "observables": observables,
        "score": score,
        "mr_curve": curve,
        "model_definition": best_definition,
    }


def main() -> None:
    args = parse_args()
    catalogue = load_catalogue()
    if args.constraints not in catalogue["sets"]:
        raise ValueError(f"Unknown constraint set {args.constraints!r}; use bja_score.py --list-constraints")
    model_path = resolve_path(args.model)
    definition = load_model_definition(model_path)
    free_params = [name.strip() for name in args.free_params.split(",") if name.strip()]
    if not free_params:
        raise ValueError("--free-params must include at least one parameter")

    if args.bounds:
        bounds = json.loads(args.bounds.read_text(encoding="utf-8"))
    else:
        bounds = definition.get("fit_bounds", {})
    selected = catalogue["sets"][args.constraints]
    result = fit(
        definition,
        catalogue["constraints"],
        selected,
        free_params,
        bounds,
        args.maxiter,
        catalogue.get("evaluation", {}).get(args.constraints, {}),
    )
    result["model"] = model_path.stem
    output_path = args.out.expanduser()
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("success", "message", "best_params", "score")}, indent=2))


if __name__ == "__main__":
    main()
