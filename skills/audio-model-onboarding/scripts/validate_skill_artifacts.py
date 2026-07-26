"""Validate Phase 02 canonical skill files, templates, scripts, and fixtures."""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

from common import emit_json
from jsonschema import Draft202012Validator

from torch_dae.cards.validation import validate_model_card_path
from torch_dae.environment.runtime import RuntimeReportSink
from torch_dae.onboarding.contracts import (
    AnalysisReport,
    DependencyEvidenceRecord,
    EnvironmentCandidateGenerationResult,
    EnvironmentResolutionReport,
    EvidenceItem,
    SkillEvaluationScenario,
)
from torch_dae.onboarding.evaluation import evaluate_analysis_report
from torch_dae.onboarding.inspection import (
    InspectionBudget,
    generate_environment_candidates,
    inspect_dependencies,
    inspect_scenario_repository,
)
from torch_dae.onboarding.rendering import render_analysis_markdown

REQUIRED_MODES = {"analyze", "resolve-environment", "integrate", "verify", "card", "profile"}
REQUIRED_REFERENCES = {
    "workflow-overview.md",
    "evidence-policy.md",
    "repository-analysis.md",
    "environment-resolution.md",
    "source-strategy.md",
    "checkpoint-discovery.md",
    "architecture-and-embeddings.md",
    "integration-planning.md",
    "runtime-verification.md",
    "model-card-authoring.md",
    "lifecycle-and-decision-gates.md",
    "failure-classification.md",
    "synthetic-evaluation.md",
}
REQUIRED_TEMPLATES = {
    "technical-analysis-report.json",
    "technical-analysis-report.md",
    "environment-resolution-report.json",
    "integration-plan.md",
    "verification-plan.md",
    "decision-request.md",
    "model-card-draft.json",
}
REQUIRED_SCRIPTS = {
    "inspect_repository.py",
    "inspect_python_project.py",
    "inspect_dependencies.py",
    "inspect_checkpoints.py",
    "inspect_model_candidates.py",
    "inspect_output_candidates.py",
    "generate_environment_candidates.py",
    "validate_analysis_report.py",
    "validate_skill_artifacts.py",
    "render_analysis_report.py",
    "common.py",
}


def load_json(path: Path) -> object:
    return json.loads(path.read_text())


def schema_validate(path: Path, schema: Path) -> None:
    Draft202012Validator(load_json(schema)).validate(load_json(path))


def markdown_link_errors(root: Path, paths: tuple[Path, ...]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        text = path.read_text()
        for match in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", text):
            target = match.group(1).split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            candidate = (path.parent / target).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                errors.append(
                    f"markdown link escapes repository: {path.relative_to(root)} -> {target}"
                )
                continue
            if not candidate.exists():
                errors.append(f"markdown link target missing: {path.relative_to(root)} -> {target}")
    return errors


def expect_validation_failure(label: str, errors: list[str], func: Callable[[], object]) -> None:
    try:
        func()
    except Exception:
        return
    errors.append(f"behavioral smoke unexpectedly passed: {label}")


def behavioral_smoke_errors(root: Path) -> list[str]:
    errors: list[str] = []
    analysis_path = root / "tests/fixtures/valid/analysis-report.synthetic.json"
    environment_path = root / "tests/fixtures/valid/environment-resolution-report.synthetic.json"
    analysis_data = load_json(analysis_path)
    env_data = load_json(environment_path)
    if not isinstance(analysis_data, dict) or not isinstance(env_data, dict):
        return ["behavioral smoke fixture load failed"]

    verified_agent = copy.deepcopy(analysis_data)
    verified_agent["source_strategy_candidates"][0]["status"] = "verified_upstream_fact"
    verified_agent["evidence_items"][0]["kind"] = "agent_inference"
    verified_agent["evidence_items"][0]["claim_status"] = "reasoned_inference"
    verified_agent["evidence_items"][0]["rationale"] = "Agent inference is not proof."
    expect_validation_failure(
        "verified source strategy citing agent inference",
        errors,
        lambda: AnalysisReport.model_validate(verified_agent),
    )

    verified_runtime = copy.deepcopy(analysis_data)
    verified_runtime["source_strategy_candidates"][0]["status"] = "verified_upstream_fact"
    verified_runtime["evidence_items"][0]["kind"] = "runtime_observation"
    verified_runtime["evidence_items"][0]["claim_status"] = "verified_upstream_fact"
    verified_runtime["evidence_items"][0]["source_file"] = None
    expect_validation_failure(
        "verified source strategy citing runtime observation",
        errors,
        lambda: AnalysisReport.model_validate(verified_runtime),
    )

    generated_upstream = copy.deepcopy(analysis_data)
    generated_upstream["evidence_items"].append(
        {
            "evidence_id": "ev-generated-upstream",
            "kind": "source_file",
            "claim_status": "verified_upstream_fact",
            "description": "Generated environment artifact.",
            "source_file": "environments/card/environment.json",
        }
    )
    generated_upstream["source_strategy_candidates"][0]["status"] = "verified_upstream_fact"
    generated_upstream["source_strategy_candidates"][0]["evidence_ids"] = ["ev-generated-upstream"]
    generated_upstream["confidence_summary"]["verified_fact_count"] += 2
    generated_upstream["confidence_summary"]["unresolved_count"] -= 1
    expect_validation_failure(
        "generated environment artifact proving verified upstream fact",
        errors,
        lambda: AnalysisReport.model_validate(generated_upstream),
    )

    generic_package = copy.deepcopy(env_data)
    for item in generic_package["evidence_items"]:
        if item["evidence_id"] == "ev-package":
            item["package_name"] = None
            item["package_version"] = None
    expect_validation_failure(
        "official package selected from generic package evidence",
        errors,
        lambda: EnvironmentResolutionReport.model_validate(generic_package),
    )

    inferred_package = copy.deepcopy(env_data)
    for item in inferred_package["evidence_items"]:
        if item["evidence_id"] == "ev-package":
            item["kind"] = "agent_inference"
            item["claim_status"] = "reasoned_inference"
            item["source_file"] = None
            item["rationale"] = "Package identity remains inferred."
    expect_validation_failure(
        "agent inference proving official package identity",
        errors,
        lambda: EnvironmentResolutionReport.model_validate(inferred_package),
    )

    url_only_package = copy.deepcopy(env_data)
    for item in url_only_package["evidence_items"]:
        if item["evidence_id"] == "ev-package":
            item["claim_status"] = "verified_upstream_fact"
            item["source_file"] = None
            item["url"] = "https://example.invalid/arbitrary-package-metadata"
    expect_validation_failure(
        "arbitrary URL-only metadata proving official package identity",
        errors,
        lambda: EnvironmentResolutionReport.model_validate(url_only_package),
    )

    verification_script_package = copy.deepcopy(env_data)
    for item in verification_script_package["evidence_items"]:
        if item["evidence_id"] == "ev-package":
            item["source_file"] = "environments/synthetic/verify_environment.py"
    expect_validation_failure(
        "verification script proving official package identity",
        errors,
        lambda: EnvironmentResolutionReport.model_validate(verification_script_package),
    )

    unresolved_gate = copy.deepcopy(env_data)
    unresolved_gate["source_strategy_decision_gates"] = [
        {
            "question_id": "q-arbitrary-source-choice",
            "classification": "needs_user_decision",
            "description": "Choose between source strategies.",
            "alternatives": ["official_package", "pinned_official_git_repository"],
            "evidence_ids": ["ev-package"],
            "default_if_deferred": "do not promote",
            "failure_classification": None,
        }
    ]
    expect_validation_failure(
        "arbitrary source-strategy gate blocks promotion",
        errors,
        lambda: EnvironmentResolutionReport.model_validate(unresolved_gate),
    )

    fingerprint = str(env_data["environment_fingerprint"])
    sink = RuntimeReportSink(
        root / ".torch-dae",
        "reports",
        "environments",
        "synthetic",
        fingerprint,
    )
    diagnostic = sink.record_event(operation="verification-script", status="success")
    diagnostic_report = copy.deepcopy(env_data)
    diagnostic_report["verification_report_or_diagnostic_reference"] = diagnostic
    try:
        EnvironmentResolutionReport.model_validate(diagnostic_report)
    except Exception as exc:
        errors.append(f"actual Phase 01 diagnostic reference rejected: {exc}")

    checkpoint_reference = copy.deepcopy(env_data)
    checkpoint_reference["verification_report_or_diagnostic_reference"] = (
        "reports/checkpoints/synthetic/check.json"
    )
    expect_validation_failure(
        "checkpoint report reference rejected for environment promotion",
        errors,
        lambda: EnvironmentResolutionReport.model_validate(checkpoint_reference),
    )

    try:
        EvidenceItem.model_validate(
            {
                "evidence_id": "ev-github",
                "kind": "source_file",
                "claim_status": "locally_observed_behavior",
                "description": "Observed CI workflow.",
                "source_file": ".github/workflows/ci.yml",
            }
        )
        DependencyEvidenceRecord.model_validate(
            {
                "normalized_name": "python",
                "raw_declaration": "python==3.11",
                "constraint": "==3.11",
                "exact_version": "3.11",
                "source_file": ".github/workflows/ci.yml",
                "source_section": "matrix.python-version",
                "dependency_kind": "locked",
                "valid": True,
                "evidence_id": "ev-ci-python",
            }
        )
    except Exception as exc:
        errors.append(f".github evidence path was not accepted: {exc}")

    dependency_evidence = inspect_dependencies(
        root / "tests/skills/fixtures/synthetic_onboarding/unpinned_dependencies",
        budget=InspectionBudget(),
    )
    records = dependency_evidence.get("dependency_records", ())
    if not any(
        record.get("raw_declaration") == "numpy<1.24"
        and record.get("constraint") == "<1.24"
        and record.get("dependency_kind") == "conda"
        for record in records
    ):
        errors.append("conda numpy range was not parsed as a version constraint")
    if not any(
        record.get("source_file") == ".github/workflows/ci.yml"
        and record.get("source_section") == "matrix.python-version"
        and record.get("raw_declaration") == "python==3.10"
        for record in records
    ):
        errors.append("CI matrix list dependency was not parsed with preserved .github path")

    hidden = load_json(root / "tests/skills/golden/hidden-checkpoint-helper.analysis.json")
    candidates = hidden.get("checkpoint_candidates", ()) if isinstance(hidden, dict) else ()
    if not candidates or candidates[0].get("helper_symbol") != "get_pretrained_checkpoint_url":
        errors.append("hidden checkpoint golden does not record helper symbol")
    hidden_scenario = SkillEvaluationScenario.model_validate_json(
        (root / "tests/skills/scenario_expectations/hidden-checkpoint-helper.json").read_text()
    )
    hidden_observation = inspect_scenario_repository(
        root / "tests/skills/fixtures/synthetic_onboarding/hidden_checkpoint_helper",
        scenario_id="hidden-checkpoint-helper",
    )
    wrong_hash_report = copy.deepcopy(hidden)
    wrong_hash_report["checkpoint_candidates"][0]["hash_evidence"] = "1" * 64
    checkpoint_failures = evaluate_analysis_report(
        hidden_scenario,
        AnalysisReport.model_validate(wrong_hash_report),
        hidden_observation,
    )
    if not any("hash was not associated" in failure for failure in checkpoint_failures):
        errors.append("checkpoint A accepted checkpoint B hash")

    with TemporaryDirectory() as temporary:
        ci_root = Path(temporary)
        workflow = ci_root / ".github/workflows/ci.yml"
        workflow.parent.mkdir(parents=True)
        workflow.write_text(
            "jobs:\n"
            "  test:\n"
            "    strategy:\n"
            "      matrix:\n"
            '        python-version: ["3.11"]\n'
            "    steps:\n"
            "      - uses: actions/setup-python@v5\n"
            "        with:\n"
            "          python-version: ${{ matrix.python-version }}\n"
        )
        (ci_root / "pyproject.toml").write_text(
            "[project]\nname = 'validator-ci-fixture'\nversion = '0.1.0'\n"
        )
        (ci_root / "environment.yml").write_text("dependencies:\n  - python=>3.9\n")
        ci_result = generate_environment_candidates(ci_root)
        ci_records = ci_result["dependency_records"]
        if any("${{ matrix.python-version }}" in item["raw_declaration"] for item in ci_records):
            errors.append("GitHub Actions expression became a dependency record")
        if ci_result["candidates"][0]["python_version"] != "3.11":
            errors.append("invalid dependency record erased valid exact CI Python version")
        if any(item.startswith("python ") for item in ci_result["unresolved_constraints"]):
            errors.append("invalid dependency record created unresolved Python constraints")
        if "dependency_conflict" in ci_result["candidates"][0]["predicted_failure_risks"]:
            errors.append("invalid dependency record created dependency conflict risk")

    budget = InspectionBudget()
    inspect_scenario_repository(
        root / "tests/skills/fixtures/synthetic_onboarding/official_package",
        scenario_id="official-package",
        budget=budget,
    )
    if budget.files_visited <= 0 or budget.bytes_read <= 0:
        errors.append("scenario inspection did not use the shared inspection budget")
    if (
        "test_real_git_grounded_scenario"
        not in (root / "tests/onboarding/test_phase02_evaluation.py").read_text()
    ):
        errors.append("real-Git grounded Phase 02 integration test is missing")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path, help="torch-dae repository root")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--synthetic-root",
        type=Path,
        default=None,
        help="override synthetic fixture root for mutation tests",
    )
    parser.add_argument(
        "--golden-root",
        type=Path,
        default=None,
        help="override golden report root for mutation tests",
    )
    args = parser.parse_args()
    root = args.repository.resolve()
    skill = root / "skills/audio-model-onboarding"
    if not skill.exists():
        payload = {
            "valid": False,
            "missing_modes": [],
            "missing_references": [],
            "missing_templates": [],
            "missing_scripts": [],
            "errors": [f"missing skill directory: {skill}"],
            "profile_reserved": False,
        }
        if args.json:
            emit_json(payload)
        else:
            print(payload)
        return 2
    text = (skill / "SKILL.md").read_text()
    missing_modes = sorted(mode for mode in REQUIRED_MODES if f"## `{mode}` Mode" not in text)
    missing_refs = sorted(
        name for name in REQUIRED_REFERENCES if not (skill / "references" / name).exists()
    )
    missing_templates = sorted(
        name for name in REQUIRED_TEMPLATES if not (skill / "templates" / name).exists()
    )
    missing_scripts = sorted(
        name for name in REQUIRED_SCRIPTS if not (skill / "scripts" / name).exists()
    )
    errors: list[str] = []

    stale_license_access_phrases = (
        "license_or_" + "access_blocker",
        "licensing/" + "access constraint",
        "license/" + "access implications",
        "ambiguous license " + "evidence",
    )
    for phrase in stale_license_access_phrases:
        if phrase in text:
            errors.append(
                f"stale combined {'license/' + 'access'} wording remains in SKILL.md: {phrase}"
            )
    if "Next allowed lifecycle transition: `integrated`" in text:
        errors.append("stale integrated lifecycle transition remains in SKILL.md")
    errors.extend(
        markdown_link_errors(
            root,
            tuple(sorted(skill.glob("*.md")))
            + tuple(sorted((skill / "references").glob("*.md")))
            + tuple(sorted((skill / "templates").glob("*.md")))
            + tuple(sorted((root / "docs").glob("*.md"))),
        )
    )

    template_dir = skill / "templates"
    try:
        analysis = AnalysisReport.model_validate_json(
            (template_dir / "technical-analysis-report.json").read_text()
        )
        schema_validate(
            template_dir / "technical-analysis-report.json",
            root / "schemas/analysis-report.schema.json",
        )
        if (template_dir / "technical-analysis-report.md").read_text() != render_analysis_markdown(
            analysis
        ):
            errors.append("technical-analysis-report.md is not rendered from the JSON template")
    except Exception as exc:
        errors.append(f"technical analysis template failed validation: {exc}")
    try:
        EnvironmentResolutionReport.model_validate_json(
            (template_dir / "environment-resolution-report.json").read_text()
        )
        schema_validate(
            template_dir / "environment-resolution-report.json",
            root / "schemas/environment-resolution-report.schema.json",
        )
        if "<card-id>" in (template_dir / "environment-resolution-report.json").read_text():
            errors.append("environment template still contains <card-id> placeholder")
    except Exception as exc:
        errors.append(f"environment-resolution template failed validation: {exc}")
    try:
        validate_model_card_path(
            template_dir / "model-card-draft.json",
            root / "schemas/model-card.schema.json",
        )
    except Exception as exc:
        errors.append(f"model-card draft template failed validation: {exc}")

    inspection_text = (root / "src/torch_dae/onboarding/inspection.py").read_text()
    if "expected_source_strategy" in inspection_text or "SCENARIO.json" in inspection_text:
        errors.append("production inspection code reads synthetic oracle fields")
    errors.extend(behavioral_smoke_errors(root))

    expectations = root / "tests/skills/scenario_expectations"
    golden = args.golden_root.resolve() if args.golden_root else root / "tests/skills/golden"
    synthetic_root = (
        args.synthetic_root.resolve()
        if args.synthetic_root
        else root / "tests/skills/fixtures/synthetic_onboarding"
    )
    for path in sorted(expectations.glob("*.json")):
        try:
            scenario = SkillEvaluationScenario.model_validate_json(path.read_text())
            report = AnalysisReport.model_validate_json(
                (golden / f"{scenario.scenario_id}.analysis.json").read_text()
            )
            scenario_fixture = synthetic_root / scenario.scenario_id.replace("-", "_")
            external_fixture = synthetic_root / "external_pytorch_implementation"
            observation = inspect_scenario_repository(
                scenario_fixture,
                scenario_id=scenario.scenario_id,
                external_pytorch_root=external_fixture
                if scenario.scenario_id == "non-pytorch-upstream"
                else None,
            )
            EnvironmentCandidateGenerationResult.model_validate(observation.environment_candidates)
            failures = evaluate_analysis_report(scenario, report, observation)
            if failures:
                errors.append(f"golden report failed {scenario.scenario_id}: {failures}")
        except Exception as exc:
            errors.append(f"golden scenario validation failed {path.name}: {exc}")

    for marker in sorted(synthetic_root.glob("*/SCENARIO.json")):
        try:
            payload = json.loads(marker.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"invalid synthetic marker {marker}: {exc}")
            continue
        if payload.get("synthetic") is not True:
            errors.append(f"synthetic marker missing synthetic=true: {marker.parent.name}")
        stale_keys = sorted(key for key in payload if key.startswith("expected_"))
        if stale_keys:
            errors.append(
                f"synthetic marker contains oracle keys {marker.parent.name}: {stale_keys}"
            )

    payload = {
        "valid": not (
            missing_modes or missing_refs or missing_templates or missing_scripts or errors
        ),
        "missing_modes": missing_modes,
        "missing_references": missing_refs,
        "missing_templates": missing_templates,
        "missing_scripts": missing_scripts,
        "errors": errors,
        "profile_reserved": "profiling is not implemented in Phase 02" in text,
    }
    if args.json:
        emit_json(payload)
    elif payload["valid"]:
        print("Phase 02 skill artifacts are present")
    else:
        print(payload)
    return 0 if payload["valid"] and payload["profile_reserved"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
