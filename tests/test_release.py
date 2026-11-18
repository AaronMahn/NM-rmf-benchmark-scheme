from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skill" / "rmf-toolkit" / "scripts"
TOOLS = SCRIPTS / "agentV1-tools"
for path in (SCRIPTS, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bja_score import evaluate_definition, load_catalogue, score_observables
from model_io import load_model_definition
from run_model import run_pipeline


@pytest.mark.slow
@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (
            "tm1_paper",
            {
                "rho0_fm3": (0.145, 0.001),
                "BE_rho0_MeV": (-15.9, 0.1),
                "Esym_rho0_MeV": (36.6, 0.1),
                "L_rho0_MeV": (109.9, 0.5),
                "K_rho0_MeV": (275.0, 1.0),
                "M_max_msun": (2.21, 0.01),
                "R_1p4_km": (14.60, 0.05),
            },
        ),
        (
            "nl3_paper",
            {
                "rho0_fm3": (0.151, 0.001),
                "BE_rho0_MeV": (-16.6, 0.1),
                "Esym_rho0_MeV": (38.7, 0.1),
                "L_rho0_MeV": (122.9, 0.5),
                "K_rho0_MeV": (264.0, 1.0),
                "M_max_msun": (2.78, 0.01),
                "R_1p4_km": (14.85, 0.05),
            },
        ),
        (
            "fsu_delta_m6p7_paper",
            {
                "rho0_fm3": (0.149, 0.001),
                "BE_rho0_MeV": (-16.4, 0.1),
                "Esym_rho0_MeV": (32.2, 0.1),
                "L_rho0_MeV": (52.5, 0.5),
                "K_rho0_MeV": (231.0, 1.0),
                "M_max_msun": (2.09, 0.01),
                "R_1p4_km": (12.98, 0.05),
            },
        ),
    ],
)
def test_table1_pipeline(model: str, expected: dict, tmp_path: Path) -> None:
    summary = run_pipeline(
        ROOT / "skill" / "rmf-toolkit" / "models" / f"{model}.json",
        tmp_path / model,
        mr_samples=160,
    )
    for key, (target, tolerance) in expected.items():
        assert abs(summary[key] - target) <= tolerance, (model, key, summary[key])


@pytest.mark.slow
def test_tm1_paper_constraint_score() -> None:
    catalogue = load_catalogue()
    set_name = "paper_sigma6_tm1"
    selected = catalogue["sets"][set_name]
    constraints = catalogue["constraints"]
    definition = load_model_definition(
        ROOT / "skill" / "rmf-toolkit" / "models" / "tm1_paper.json"
    )
    observables, _ = evaluate_definition(
        definition,
        {key: constraints[key] for key in selected},
        catalogue.get("evaluation", {}).get(set_name),
    )
    score = score_observables(observables, constraints, selected)
    assert abs(score["total_chi2"] - 25.495) <= 0.01


def test_fit_model_cli_writes_result(tmp_path: Path) -> None:
    bounds = tmp_path / "bounds.json"
    bounds.write_text(json.dumps({"g_sigma": [-11.0, -9.0]}), encoding="utf-8")
    output = tmp_path / "fit.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "fit_model.py"),
            "--model",
            str(ROOT / "skill" / "rmf-toolkit" / "models" / "tm1_paper.json"),
            "--constraints",
            "standard_nm",
            "--free-params",
            "g_sigma",
            "--bounds",
            str(bounds),
            "--maxiter",
            "1",
            "--out",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["free_params"] == ["g_sigma"]
    assert "g_sigma" in result["best_params"]


@pytest.mark.slow
def test_sigma6_evaluate_only(tmp_path: Path) -> None:
    output = tmp_path / "sigma6.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "paper" / "sigma6_tm1" / "fit_run.py"),
            "--evaluate-only",
            "--out",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(result.stdout)
    evaluations = json.loads(output.read_text(encoding="utf-8"))
    for case, expected in {"gaussian": 4.877, "gaussian_lambda": 0.120}.items():
        assert summary[case]["chi2"] == pytest.approx(expected, abs=0.01)
        assert abs(summary[case]["delta_chi2_from_reference"]) <= 0.01
        assert evaluations[case]["chi2"] == summary[case]["chi2"]


@pytest.mark.slow
def test_sample_zero_mr_reproduction() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "harness" / "gt" / "verify_star.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    assert report["M_relerr_%"] <= 1e-6
    assert report["R_relerr_%"] <= 1e-6


def test_gt_recalibration_refuses_to_write_shipped_targets(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "harness" / "gt" / "recalibrate_mr.py"),
            "--out",
            str(ROOT / "data" / "gt" / "targets_100x6.npy"),
            "--report",
            str(tmp_path / "report.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--out must write under harness/gt" in result.stderr


def test_release_hygiene() -> None:
    forbidden_paths = ("/" + "Users" + "/", "/" + "home" + "/")
    key_pattern = re.compile(r"sk-[A-Za-z0-9]{20,}")
    ip_pattern = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
    openclaw_default = "." + "openclaw"
    agent_memory_name = "agent_" + "memory.json"
    failures: list[str] = []
    openclaw_occurrences: list[Path] = []

    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in {".git", ".venv", "__pycache__"} for part in relative.parts):
            continue
        if openclaw_default in str(relative):
            failures.append(f"default path in file name: {relative}")
        if path.name == agent_memory_name:
            failures.append(f"forbidden file name: {relative}")
        if not path.is_file():
            continue
        content = path.read_bytes()
        if b"\0" in content:
            continue
        text = content.decode("utf-8", errors="ignore")
        for marker in forbidden_paths:
            if marker in text:
                failures.append(f"private absolute path in {relative}")
        if key_pattern.search(text):
            failures.append(f"API-key-like string in {relative}")
        if ip_pattern.search(text):
            failures.append(f"IPv4 address in {relative}")
        if agent_memory_name in text:
            failures.append(f"agent memory marker in {relative}")
        if openclaw_default in text:
            openclaw_occurrences.append(relative)

    if openclaw_occurrences != [Path("harness/runtime_config.py")]:
        failures.append(f"unexpected default-path occurrences: {openclaw_occurrences}")
    assert not failures, "\n".join(failures)


def test_paper_extractor_with_mocked_client_and_no_key(monkeypatch, tmp_path: Path) -> None:
    import paper_extractor

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(paper_extractor, "read_pdf_file", lambda _path: "mock paper text")
    response = json.dumps(
        {
            "model_name": "mock-model",
            "description": "Mock model",
            "param_keys": ["P['m_N']"],
            "field_names": ["s"],
            "lagrangian_terms": ["sp.Integer(1) * psibar_i_dslash_psi"],
            "params": {"m_N": 939.0},
            "paper_reported": {"rho0_fm3": None},
            "extraction_notes": "mock response",
        }
    )
    monkeypatch.setattr(
        paper_extractor,
        "_call_claude_api",
        lambda **_kwargs: response,
    )
    result = paper_extractor.extract_model_from_paper(
        pdf_path=tmp_path / "mock.pdf",
        verbose=False,
    )
    assert result.model_definition["model_name"] == "mock-model"
    assert result.paper_reported == {"rho0_fm3": None}
    assert result.source_text_chars == len("mock paper text")


def test_paper_runner_extract_only_with_mocked_client(monkeypatch, tmp_path: Path) -> None:
    from paper_extractor import PaperExtractionResult
    import run_paper_model

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        run_paper_model,
        "extract_model_from_paper",
        lambda **_kwargs: PaperExtractionResult(
            model_definition={
                "model_name": "mock model",
                "param_keys": ["P['m_N']"],
                "field_names": ["s"],
                "lagrangian_terms": ["sp.Integer(1) * psibar_i_dslash_psi"],
                "params": {"m_N": 939.0},
            },
            paper_reported={"rho0_fm3": None},
            extraction_notes="mock",
            raw_response="{}",
            source="pdf:mock",
            source_text_chars=15,
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_paper_model.py",
            "--pdf",
            str(tmp_path / "mock.pdf"),
            "--extract-only",
            "--out",
            str(tmp_path / "results"),
        ],
    )
    run_paper_model.main()
    assert (tmp_path / "results" / "mock_model_paper" / "mock_model_definition.json").is_file()


def test_paper_extractor_requires_environment_key(monkeypatch) -> None:
    import paper_extractor

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        paper_extractor._call_claude_api(system="", user="")
