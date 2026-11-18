#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
VALIDATION_DIR = HERE.parent / "referee_validation"
sys.path.insert(0, str(VALIDATION_DIR / "nicer_fit"))

from fit import Objective
from rmf import vector

REFERENCE_DIR = VALIDATION_DIR / "reference"
FIT_REFERENCES = {
    "gaussian": REFERENCE_DIR / "A1" / "gaussian.json",
    "gaussian_lambda": REFERENCE_DIR / "A2" / "gaussian_lambda.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Table 3 sigma6 and sigma6+c4 Gaussian fits.")
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        required=True,
        help="Recompute observables and chi2 from the released A1/A2 couplings.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "evaluation_result.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def evaluate_fit(case: str):
    reference = json.loads(FIT_REFERENCES[case].read_text(encoding="utf-8"))
    couplings = reference["couplings"]
    chi2, observables, terms, curve, _ = Objective(
        extension=(case == "gaussian_lambda")
    ).evaluate(vector(couplings), fine=True)
    names = reference["observables_names"]
    reference_chi2 = reference["fine_score"]
    return (
        {
            "parameters": couplings,
            "observables": dict(zip(names, observables.tolist())),
            "chi2": chi2,
            "chi2_terms": dict(zip(names, terms.tolist())),
            "reference_chi2": reference_chi2,
            "delta_chi2_from_reference": chi2 - reference_chi2,
        },
        curve,
    )


def main() -> None:
    args = parse_args()
    results = {}
    summary = {}
    for case in FIT_REFERENCES:
        result, curve = evaluate_fit(case)
        result["mr_curve"] = [
            {"radius_km": radius, "mass_msun": mass} for _, mass, radius in curve.tolist()
        ]
        results[case] = result
        summary[case] = {
            key: result[key] for key in ("chi2", "reference_chi2", "delta_chi2_from_reference")
        }
    output_path = args.out.expanduser()
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
