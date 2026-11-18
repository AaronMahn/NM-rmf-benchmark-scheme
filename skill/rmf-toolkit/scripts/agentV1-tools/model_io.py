from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_TOOLS_DIR = Path(__file__).resolve().parent
for _tool_path in (_TOOLS_DIR, _TOOLS_DIR / "NMtoolsV1", _TOOLS_DIR / "TOVtools"):
    if str(_tool_path) not in sys.path:
        sys.path.insert(0, str(_tool_path))

from NMtoolsV1.analytic_layer import build_analytic_model
from NMtoolsV1.nuclear_properties import epsilon_function_factory, pressure_from_epsilon
from TOVtools.tool_beta_eff import compile_beta_effective_model


@dataclass
class ModelBuildResult:
    model_name: str
    description: str
    analytic: object
    params: dict[str, float]
    compiled: object
    epsilon_func: Callable
    pressure_func: Callable
    initial_field_guess: list[float]
    rho_grid: list[float]


def load_model_definition(path: str | Path) -> dict:
    """Read a model definition JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_model_from_definition(defn: dict) -> ModelBuildResult:
    """Build the analytic, numeric, and beta-equilibrium model objects."""
    param_keys = list(defn["param_keys"])
    declared_names = {
        match.group(1) if (match := re.match(r"P\[['\"](.*?)['\"]\]", key)) else key
        for key in param_keys
    }
    lagrangian_text = " ".join(defn.get("lagrangian_terms", []))
    params = {name: float(value) for name, value in defn["params"].items()}
    for name in params:
        if name not in declared_names and name in lagrangian_text:
            param_keys.append(f"P['{name}']")
            declared_names.add(name)

    field_names = defn["field_names"]
    extra_symbol_names = defn.get(
        "extra_symbol_names",
        [
            "psibar_i_dslash_psi",
            "psibar_psi",
            "psibar_gamma0_psi",
            "psibar_gamma0_tau3_psi",
            "psibar_tau3_psi",
        ],
    )
    initial_field_guess = list(defn.get("initial_field_guess", [20.0] * len(field_names)))
    rho_min = float(defn.get("rho_beta_min", 0.08))
    rho_max = float(defn.get("rho_beta_max", 1.2))
    rho_step = float(defn.get("rho_beta_step", 0.001))
    if rho_step <= 0 or rho_max < rho_min:
        raise ValueError("rho_beta_step must be positive and rho_beta_max >= rho_beta_min")
    n_steps = int(round((rho_max - rho_min) / rho_step))
    rho_grid = [rho_min + rho_step * i for i in range(n_steps + 1)]

    analytic = build_analytic_model(
        param_keys,
        field_names,
        extra_symbol_names,
        defn["lagrangian_terms"],
    )
    epsilon_func = epsilon_function_factory(
        analytic,
        params,
        initial_guess=initial_field_guess,
    )

    def pressure_func(rho_n: float, rho_p: float) -> float:
        return pressure_from_epsilon(epsilon_func, rho_n, rho_p)

    compiled = compile_beta_effective_model(analytic, params)
    return ModelBuildResult(
        model_name=defn.get("model_name", "unnamed_model"),
        description=defn.get("description", ""),
        analytic=analytic,
        params=params,
        compiled=compiled,
        epsilon_func=epsilon_func,
        pressure_func=pressure_func,
        initial_field_guess=initial_field_guess,
        rho_grid=rho_grid,
    )


def save_model_definition(defn: dict, path: str | Path) -> None:
    """Write a model definition JSON file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(defn, indent=2, ensure_ascii=False), encoding="utf-8")
