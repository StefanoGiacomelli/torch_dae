"""Repository validation for Phase 01 repository safety."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from copy import deepcopy
from importlib.metadata import distributions
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from pydantic import BaseModel

from torch_dae.cards.validation import load_json, validate_model_card_path
from torch_dae.core.checkpoint import CheckpointMaterializationRecord, CheckpointSpec
from torch_dae.core.embeddings import EmbeddingSpec
from torch_dae.core.registry import ModelCardRegistry
from torch_dae.environment.runtime import EnvironmentMaterializationRecord, RuntimeReportSink
from torch_dae.environment.specification import EnvironmentSourcesManifest, EnvironmentSpecification
from torch_dae.environment.verification import VerificationReport
from torch_dae.onboarding.contracts import (
    AnalysisReport,
    DependencyEvidenceRecord,
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

ROOT = Path(__file__).resolve().parents[1]
MODEL_DEPS = {"torch", "torchaudio", "torchvision", "transformers", "tensorflow", "jax", "librosa"}
REQUIRED = [
    "project_spec.md",
    "pyproject.toml",
    "uv.lock",
    "skills/audio-model-onboarding/SKILL.md",
    ".agents/skills/audio-model-onboarding",
    ".claude/skills/audio-model-onboarding",
    "schemas/model-card.schema.json",
    "schemas/checkpoint.schema.json",
    "schemas/environment.schema.json",
    "schemas/environment-sources.schema.json",
    "schemas/embedding.schema.json",
    "schemas/verification-report.schema.json",
    "schemas/analysis-report.schema.json",
    "schemas/environment-resolution-report.schema.json",
    "src/torch_dae",
    "tests/fixtures",
]


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)


def git_ignored(path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", path],
        cwd=ROOT,
        check=False,
    )
    return result.returncode == 0


def schema_validate(fixture: Path, schema: Path) -> None:
    data = load_json(fixture)
    Draft202012Validator(load_json(schema), format_checker=FormatChecker()).validate(data)


def pydantic_validate(fixture: Path) -> None:
    kind = fixture.name.split(".")[0]
    if kind == "model-card":
        validate_model_card_path(fixture)
        return
    model_by_kind: dict[str, type[BaseModel]] = {
        "checkpoint": CheckpointSpec,
        "embedding": EmbeddingSpec,
        "environment": EnvironmentSpecification,
        "environment-sources": EnvironmentSourcesManifest,
        "verification-report": VerificationReport,
        "analysis-report": AnalysisReport,
        "environment-resolution-report": EnvironmentResolutionReport,
    }
    model_by_kind[kind].model_validate(load_json(fixture))


def validate_fixture(path: Path, schema: Path) -> None:
    kind = path.name.split(".")[0]
    if kind == "model-card":
        validate_model_card_path(path, schema)
        return
    schema_validate(path, schema)
    pydantic_validate(path)


def semantic_invalid_fixture_fails(path: Path) -> bool:
    """Return whether a semantic-cross-reference fixture fails repository validation."""

    try:
        pydantic_validate(path)
    except Exception:
        return True
    name = path.name
    if name == "environment.model-card-id-mismatch.json":
        specification = EnvironmentSpecification.model_validate_json(path.read_text())
        return specification.model_card_id != "synthetic-family-variant-checkpoint"
    if name == "environment-sources.environment-id-mismatch.json":
        valid_spec = EnvironmentSpecification.model_validate_json(
            (ROOT / "tests/fixtures/valid/environment.synthetic.json").read_text()
        )
        manifest = EnvironmentSourcesManifest.model_validate_json(path.read_text())
        return manifest.environment_id != valid_spec.environment_id
    return False


def expect_model_failure(label: str, failures: list[str], func: Callable[[], object]) -> None:
    try:
        func()
    except Exception:
        return
    fail(f"behavioral smoke unexpectedly passed: {label}", failures)


def phase02_behavioral_smoke(failures: list[str]) -> None:
    analysis_data = load_json(ROOT / "tests/fixtures/valid/analysis-report.synthetic.json")
    env_data = load_json(ROOT / "tests/fixtures/valid/environment-resolution-report.synthetic.json")

    verified_agent = deepcopy(analysis_data)
    verified_agent["source_strategy_candidates"][0]["status"] = "verified_upstream_fact"
    verified_agent["evidence_items"][0]["kind"] = "agent_inference"
    verified_agent["evidence_items"][0]["claim_status"] = "reasoned_inference"
    verified_agent["evidence_items"][0]["rationale"] = "Agent inference is not proof."
    expect_model_failure(
        "verified source strategy citing agent inference",
        failures,
        lambda: AnalysisReport.model_validate(verified_agent),
    )

    verified_runtime = deepcopy(analysis_data)
    verified_runtime["source_strategy_candidates"][0]["status"] = "verified_upstream_fact"
    verified_runtime["evidence_items"][0]["kind"] = "runtime_observation"
    verified_runtime["evidence_items"][0]["claim_status"] = "verified_upstream_fact"
    verified_runtime["evidence_items"][0]["source_file"] = None
    expect_model_failure(
        "verified source strategy citing runtime observation",
        failures,
        lambda: AnalysisReport.model_validate(verified_runtime),
    )

    generated_upstream = deepcopy(analysis_data)
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
    expect_model_failure(
        "generated environment artifact proving verified upstream fact",
        failures,
        lambda: AnalysisReport.model_validate(generated_upstream),
    )

    generic_package = deepcopy(env_data)
    for item in generic_package["evidence_items"]:
        if item["evidence_id"] == "ev-package":
            item["package_name"] = None
            item["package_version"] = None
    expect_model_failure(
        "official package selected from generic package evidence",
        failures,
        lambda: EnvironmentResolutionReport.model_validate(generic_package),
    )

    inferred_package = deepcopy(env_data)
    for item in inferred_package["evidence_items"]:
        if item["evidence_id"] == "ev-package":
            item["kind"] = "agent_inference"
            item["claim_status"] = "reasoned_inference"
            item["source_file"] = None
            item["rationale"] = "Package identity remains inferred."
    expect_model_failure(
        "agent inference proving official package identity",
        failures,
        lambda: EnvironmentResolutionReport.model_validate(inferred_package),
    )

    url_only_package = deepcopy(env_data)
    for item in url_only_package["evidence_items"]:
        if item["evidence_id"] == "ev-package":
            item["claim_status"] = "verified_upstream_fact"
            item["source_file"] = None
            item["url"] = "https://example.invalid/arbitrary-package-metadata"
    expect_model_failure(
        "arbitrary URL-only metadata proving official package identity",
        failures,
        lambda: EnvironmentResolutionReport.model_validate(url_only_package),
    )

    verification_script_package = deepcopy(env_data)
    for item in verification_script_package["evidence_items"]:
        if item["evidence_id"] == "ev-package":
            item["source_file"] = "environments/synthetic/verify_environment.py"
    expect_model_failure(
        "verification script proving official package identity",
        failures,
        lambda: EnvironmentResolutionReport.model_validate(verification_script_package),
    )

    unresolved_gate = deepcopy(env_data)
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
    expect_model_failure(
        "arbitrary source-strategy gate blocks promotion",
        failures,
        lambda: EnvironmentResolutionReport.model_validate(unresolved_gate),
    )

    fingerprint = str(env_data["environment_fingerprint"])
    sink = RuntimeReportSink(
        ROOT / ".torch-dae",
        "reports",
        "environments",
        "synthetic",
        fingerprint,
    )
    diagnostic = sink.record_event(operation="verification-script", status="success")
    diagnostic_report = deepcopy(env_data)
    diagnostic_report["verification_report_or_diagnostic_reference"] = diagnostic
    try:
        EnvironmentResolutionReport.model_validate(diagnostic_report)
    except Exception as exc:
        fail(f"actual Phase 01 diagnostic reference rejected: {exc}", failures)

    checkpoint_reference = deepcopy(env_data)
    checkpoint_reference["verification_report_or_diagnostic_reference"] = (
        "reports/checkpoints/synthetic/check.json"
    )
    expect_model_failure(
        "checkpoint report reference rejected for environment promotion",
        failures,
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
        fail(f".github evidence path was not accepted: {exc}", failures)

    dependency_evidence = inspect_dependencies(
        ROOT / "tests/skills/fixtures/synthetic_onboarding/unpinned_dependencies",
        budget=InspectionBudget(),
    )
    records = dependency_evidence.get("dependency_records", ())
    if not any(
        record.get("raw_declaration") == "numpy<1.24"
        and record.get("constraint") == "<1.24"
        and record.get("dependency_kind") == "conda"
        for record in records
    ):
        fail("conda numpy range was not parsed as a version constraint", failures)
    if not any(
        record.get("source_file") == ".github/workflows/ci.yml"
        and record.get("source_section") == "matrix.python-version"
        and record.get("raw_declaration") == "python==3.10"
        for record in records
    ):
        fail("CI matrix list dependency was not parsed with preserved .github path", failures)

    hidden = load_json(ROOT / "tests/skills/golden/hidden-checkpoint-helper.analysis.json")
    candidates = hidden.get("checkpoint_candidates", ())
    if not candidates or candidates[0].get("helper_symbol") != "get_pretrained_checkpoint_url":
        fail("hidden checkpoint golden does not record helper symbol", failures)
    hidden_scenario = SkillEvaluationScenario.model_validate_json(
        (ROOT / "tests/skills/scenario_expectations/hidden-checkpoint-helper.json").read_text()
    )
    hidden_observation = inspect_scenario_repository(
        ROOT / "tests/skills/fixtures/synthetic_onboarding/hidden_checkpoint_helper",
        scenario_id="hidden-checkpoint-helper",
    )
    wrong_hash_report = deepcopy(hidden)
    wrong_hash_report["checkpoint_candidates"][0]["hash_evidence"] = "1" * 64
    checkpoint_failures = evaluate_analysis_report(
        hidden_scenario,
        AnalysisReport.model_validate(wrong_hash_report),
        hidden_observation,
    )
    if not any("hash was not associated" in failure for failure in checkpoint_failures):
        fail("checkpoint A accepted checkpoint B hash", failures)

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
            fail("GitHub Actions expression became a dependency record", failures)
        if ci_result["candidates"][0]["python_version"] != "3.11":
            fail("invalid dependency record erased valid exact CI Python version", failures)
        if any(item.startswith("python ") for item in ci_result["unresolved_constraints"]):
            fail("invalid dependency record created unresolved Python constraints", failures)
        if "dependency_conflict" in ci_result["candidates"][0]["predicted_failure_risks"]:
            fail("invalid dependency record created dependency conflict risk", failures)

    budget = InspectionBudget()
    inspect_scenario_repository(
        ROOT / "tests/skills/fixtures/synthetic_onboarding/official_package",
        scenario_id="official-package",
        budget=budget,
    )
    if budget.files_visited <= 0 or budget.bytes_read <= 0:
        fail("scenario inspection did not use the shared inspection budget", failures)
    if (
        "test_real_git_grounded_scenario"
        not in (ROOT / "tests/onboarding/test_phase02_evaluation.py").read_text()
    ):
        fail("real-Git grounded Phase 02 integration test is missing", failures)


def main() -> int:
    failures: list[str] = []

    if ROOT.name != "torch-dae":
        fail("repository root basename is not torch-dae", failures)
    for relative in REQUIRED:
        if not (ROOT / relative).exists():
            fail(f"missing required path: {relative}", failures)

    if list(ROOT.glob("**/*backbone*.json")):
        fail("legacy backbone JSON files are present", failures)
    if list((ROOT / "model_cards").glob("**/*.json")):
        fail("real model cards are present during Phase 00", failures)
    if (ROOT / "src/torch_dae/models").exists():
        fail("pilot model modules exist during Phase 00", failures)
    if subprocess.run(
        ["git", "ls-files", "._*"], cwd=ROOT, check=False, capture_output=True, text=True
    ).stdout:
        fail("tracked AppleDouble files are present", failures)
    if subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--", ".torch-dae"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    ).stdout:
        fail("runtime artifact is staged under .torch-dae", failures)
    if subprocess.run(
        ["git", "ls-files", ".torch-dae"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    ).stdout:
        fail("runtime artifact is tracked under .torch-dae", failures)
    if not git_ignored(".torch-dae/checkpoints/example/file.bin"):
        fail(".torch-dae/ paths are not ignored", failures)
    if list((ROOT / "environments").glob("*/.venv")):
        fail("committed environment directory contains .venv", failures)

    canonical = (ROOT / "skills/audio-model-onboarding").resolve()
    for relative in [
        ".agents/skills/audio-model-onboarding",
        ".claude/skills/audio-model-onboarding",
    ]:
        if (ROOT / relative).resolve() != canonical:
            fail(f"{relative} does not resolve to canonical skill", failures)

    for schema in (ROOT / "schemas").glob("*.json"):
        try:
            Draft202012Validator.check_schema(load_json(schema))
        except Exception as exc:
            fail(f"invalid schema {schema.name}: {exc}", failures)

    result = subprocess.run(
        [sys.executable, "scripts/generate_schemas.py", "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        fail("schemas are not synchronized with Pydantic generation", failures)

    valid_dir = ROOT / "tests/fixtures/valid"
    invalid_dir = ROOT / "tests/fixtures/invalid"
    manifest = ROOT / "tests/fixtures/invalid_manifest.json"
    classifications = load_json(manifest) if manifest.exists() else {}
    schema_map = {
        "model-card": ROOT / "schemas/model-card.schema.json",
        "environment": ROOT / "schemas/environment.schema.json",
        "environment-sources": ROOT / "schemas/environment-sources.schema.json",
        "checkpoint": ROOT / "schemas/checkpoint.schema.json",
        "embedding": ROOT / "schemas/embedding.schema.json",
        "verification-report": ROOT / "schemas/verification-report.schema.json",
        "analysis-report": ROOT / "schemas/analysis-report.schema.json",
        "environment-resolution-report": ROOT / "schemas/environment-resolution-report.schema.json",
    }
    for path in valid_dir.glob("*.json"):
        kind = path.name.split(".")[0]
        try:
            validate_fixture(path, schema_map[kind])
        except Exception as exc:
            fail(f"valid fixture failed {path.name}: {exc}", failures)
    for path in invalid_dir.glob("*.json"):
        relative = f"tests/fixtures/invalid/{path.name}"
        if classifications.get(relative) == "semantic_cross_reference":
            continue
        kind = path.name.split(".")[0]
        try:
            validate_fixture(path, schema_map[kind])
        except Exception:
            continue
        fail(f"invalid fixture passed: {path.name}", failures)

    if manifest.exists():
        for relative, classification in classifications.items():
            path = ROOT / relative
            kind = path.name.split(".")[0]
            schema = schema_map[kind]
            pydantic_failed = False
            schema_failed = False
            try:
                if kind == "model-card":
                    validate_model_card_path(path)
                else:
                    pydantic_validate(path)
            except Exception:
                pydantic_failed = True
            try:
                schema_validate(path, schema)
            except Exception:
                schema_failed = True
            if classification == "structural" and not (pydantic_failed and schema_failed):
                fail(
                    f"structural invalid fixture did not fail both validators: {relative}",
                    failures,
                )
            if classification == "semantic_cross_reference" and not pydantic_failed:
                if not semantic_invalid_fixture_fails(path):
                    fail(
                        f"semantic invalid fixture passed repository validation: {relative}",
                        failures,
                    )

    registry = ModelCardRegistry(ROOT)
    try:
        registry.list_cards()
    except Exception as exc:
        fail(f"empty production registry failed: {exc}", failures)

    installed = {dist.metadata["Name"].lower() for dist in distributions()}
    installed &= MODEL_DEPS
    if installed:
        fail(
            f"model-specific dependencies importable in root environment: {sorted(installed)}",
            failures,
        )
    for model in (EnvironmentMaterializationRecord, CheckpointMaterializationRecord):
        if not model.model_fields:
            fail(f"runtime metadata model did not import: {model.__name__}", failures)

    skill_root = ROOT / "skills/audio-model-onboarding"
    skill_text = (skill_root / "SKILL.md").read_text()
    required_modes = ("analyze", "resolve-environment", "integrate", "verify", "card", "profile")
    for mode in required_modes:
        if f"## `{mode}` Mode" not in skill_text:
            fail(f"Phase 02 skill mode is missing: {mode}", failures)
    if "profiling is not implemented in Phase 02" not in skill_text:
        fail("profile mode is not truthfully reserved in the skill", failures)
    stale_license_access_phrases = (
        "license_or_" + "access_blocker",
        "licensing/" + "access constraint",
        "license/" + "access implications",
        "ambiguous license " + "evidence",
    )
    for phrase in stale_license_access_phrases:
        if phrase in skill_text:
            fail(
                f"stale combined {'license/' + 'access'} wording remains in SKILL.md: {phrase}",
                failures,
            )
    required_references = {
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
    required_templates = {
        "technical-analysis-report.json",
        "technical-analysis-report.md",
        "environment-resolution-report.json",
        "integration-plan.md",
        "verification-plan.md",
        "decision-request.md",
        "model-card-draft.json",
    }
    required_scripts = {
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
    for name in sorted(required_references):
        path = skill_root / "references" / name
        if not path.exists() or not path.read_text().strip():
            fail(f"required Phase 02 reference is missing or empty: {name}", failures)
        if f"references/{name}" not in skill_text:
            fail(f"SKILL.md does not link reference: {name}", failures)
    for name in sorted(required_templates):
        if not (skill_root / "templates" / name).exists():
            fail(f"required Phase 02 template is missing: {name}", failures)
    for name in sorted(required_scripts):
        path = skill_root / "scripts" / name
        if not path.exists():
            fail(f"required Phase 02 script is missing: {name}", failures)
        elif "import torch" in path.read_text() or "urllib.request.urlopen" in path.read_text():
            fail(
                f"Phase 02 script imports a model runtime or public network primitive: {name}",
                failures,
            )
    result = subprocess.run(
        [
            sys.executable,
            "skills/audio-model-onboarding/scripts/validate_skill_artifacts.py",
            str(ROOT),
            "--json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        fail("Phase 02 skill artifact validation script failed", failures)
    phase02_behavioral_smoke(failures)
    synthetic_root = ROOT / "tests/skills/fixtures/synthetic_onboarding"
    required_scenarios = {
        "official_package": "official_package",
        "pinned_git": "pinned_official_git_repository",
        "minimal_vendoring": "minimal_vendored_adaptation",
        "ambiguous_embeddings": "official_package",
        "unpinned_dependencies": "official_package",
        "hidden_checkpoint_helper": "official_package",
        "non_pytorch_upstream": "external_pytorch_implementation",
        "unsupported": "unsupported_or_non_equivalent_implementation",
    }
    expectation_root = ROOT / "tests/skills/scenario_expectations"
    golden_root = ROOT / "tests/skills/golden"
    for scenario, expected_strategy in sorted(required_scenarios.items()):
        scenario_root = synthetic_root / scenario
        marker = scenario_root / "SCENARIO.json"
        readme = scenario_root / "README.md"
        if not marker.exists():
            fail(f"missing synthetic scenario marker: {scenario}", failures)
            continue
        if not readme.exists() or "synthetic" not in readme.read_text().lower():
            fail(f"synthetic fixture marker missing from README: {scenario}", failures)
        try:
            payload = json.loads(marker.read_text())
        except json.JSONDecodeError as exc:
            fail(f"invalid synthetic scenario JSON {scenario}: {exc}", failures)
            continue
        if payload.get("synthetic") is not True:
            fail(f"synthetic fixture is not explicitly marked synthetic: {scenario}", failures)
        stale_keys = sorted(key for key in payload if key.startswith("expected_"))
        if stale_keys:
            fail(f"synthetic fixture embeds oracle keys {scenario}: {stale_keys}", failures)
        expectation_id = payload.get("scenario_id")
        expectation_path = expectation_root / f"{expectation_id}.json"
        golden_path = golden_root / f"{expectation_id}.analysis.json"
        if not expectation_path.exists():
            fail(f"missing external scenario expectation: {expectation_id}", failures)
            continue
        expectation_payload = json.loads(expectation_path.read_text())
        if expectation_payload.get("expected_source_strategy") != expected_strategy:
            fail(f"external scenario expectation has wrong strategy: {expectation_id}", failures)
        if not golden_path.exists():
            fail(f"missing golden scenario analysis report: {expectation_id}", failures)
        else:
            try:
                scenario_contract = SkillEvaluationScenario.model_validate(expectation_payload)
                analysis_report = AnalysisReport.model_validate_json(golden_path.read_text())
                external_fixture = synthetic_root / "external_pytorch_implementation"
                observation = inspect_scenario_repository(
                    scenario_root,
                    scenario_id=scenario_contract.scenario_id,
                    external_pytorch_root=external_fixture
                    if scenario_contract.scenario_id == "non-pytorch-upstream"
                    else None,
                )
                failures_for_report = evaluate_analysis_report(
                    scenario_contract, analysis_report, observation
                )
                if failures_for_report:
                    fail(
                        f"golden scenario report failed {expectation_id}: {failures_for_report}",
                        failures,
                    )
            except Exception as exc:
                fail(f"golden scenario report is invalid {expectation_id}: {exc}", failures)
    inspection_text = (ROOT / "src/torch_dae/onboarding/inspection.py").read_text()
    if "expected_source_strategy" in inspection_text or "SCENARIO.json" in inspection_text:
        fail("production inspection code reads synthetic oracle fields", failures)
    for relative in (
        ".agents/skills/audio-model-onboarding",
        ".claude/skills/audio-model-onboarding",
    ):
        alias = ROOT / relative
        if not alias.is_symlink():
            fail(f"agent skill path is not a symlink: {relative}", failures)
        elif (alias.resolve() / "SKILL.md").read_bytes() != (canonical / "SKILL.md").read_bytes():
            fail(f"agent skill SKILL.md bytes diverge: {relative}", failures)
    stale_duplicates = [
        path
        for path in ROOT.glob("**/audio-model-onboarding/SKILL.md")
        if "skills/audio-model-onboarding/SKILL.md" not in path.as_posix()
    ]
    if stale_duplicates:
        fail(f"stale duplicate onboarding skills found: {stale_duplicates}", failures)

    env_cli = (ROOT / "src/torch_dae/cli/environment.py").read_text()
    checkpoint_cli = (ROOT / "src/torch_dae/cli/checkpoints.py").read_text()
    model_cli = (ROOT / "src/torch_dae/cli/models.py").read_text()
    env_manager = (ROOT / "src/torch_dae/environment/manager.py").read_text()
    source_manager = (ROOT / "src/torch_dae/environment/sources.py").read_text()
    runtime_module = (ROOT / "src/torch_dae/environment/runtime.py").read_text()
    if "belongs to Phase 01" in env_cli or "belongs to Phase 01" in checkpoint_cli:
        fail("environment or checkpoint CLI remains deferred to Phase 01", failures)
    if "belongs to Phase 03+" not in model_cli:
        fail("model CLI no longer truthfully defers model integration", failures)
    if "_write_local_wheel" in env_manager or "wheel_record_hash" in env_manager:
        fail("handwritten local wheel implementation remains present", failures)
    if '"uv",\n                    "build"' not in env_manager:
        fail("local torch-dae wheel is not built through uv/build backend", failures)
    if "source-builds/torch-dae/current" in source_manager:
        fail("stale hard-coded local wheel cache lookup remains", failures)
    if (
        "recommended.environment_id != model_card_id" in env_manager
        or "must match the requested card ID" in env_manager
    ):
        fail("environment manager still requires environment_id == card_id", failures)
    for required in (
        "recommended.lockfile != specification.lockfile",
        "_validate_environment_artifact_paths",
        "environments/{model_card_id}/uv.lock",
        "environments/{model_card_id}/pyproject.toml",
        "environments/{model_card_id}/sources.json",
        "environments/{model_card_id}/verify_environment.py",
    ):
        if required not in env_manager:
            fail(f"environment path/reference validation is missing: {required}", failures)
    fingerprint_module = (ROOT / "src/torch_dae/environment/fingerprint.py").read_text()
    for required in (
        "readme",
        "local_package_content_digest",
        "local_package_build_inputs",
        'src_root = repository_root / "src" / "torch_dae"',
        "git:{head.stdout.strip()}:content:",
    ):
        if required not in fingerprint_module:
            fail(f"local package identity coverage is missing: {required}", failures)
    if '["git", "clone", source.url, str(checkout)]' in source_manager:
        fail("Git source acquisition still clones directly into the final cache path", failures)
    for required in (
        ".clone-",
        "_ensure_git_checkout",
        "except SourceMaterializationError",
        "OfflineResourceUnavailableError",
        "replace_tree(checkout)",
    ):
        if required not in source_manager:
            fail(f"online Git checkout recovery is missing: {required}", failures)
    if (
        "GitSourceWheelCacheRecord" not in runtime_module
        or "_valid_cached_git_wheel" not in source_manager
    ):
        fail("Git source wheel cache metadata validation is missing", failures)
    compact_env_manager = env_manager.replace(" ", "").replace("\n", "")
    if (
        "env_remove=python_env_remove()" not in compact_env_manager
        or "PYTHONPATH" not in env_manager
    ):
        fail("model-environment subprocess sanitation is missing", failures)
    subprocess_module = (ROOT / "src/torch_dae/environment/subprocess.py").read_text()
    if "RuntimeReportSink" not in runtime_module:
        fail("shared runtime report sink is missing", failures)
    if "with_report_sink" not in subprocess_module:
        fail("CommandExecutor report-sink integration is missing", failures)
    if (
        "command_log_references" not in env_manager
        or '"reports"' not in env_manager
        or '"environments"' not in env_manager
    ):
        fail("environment command diagnostics are not wired into metadata", failures)
    for operation in (
        "python-resolution",
        "uv-venv",
        "uv-sync",
        "local-wheel-build",
        "local-wheel-install",
        "dependency-check",
        "verification-script",
    ):
        if operation not in env_manager:
            fail(f"environment operation report label is missing: {operation}", failures)
    for operation in (
        "git-clone",
        "git-checkout",
        "git-revision-check",
        "git-remote-check",
        "git-cleanliness-check",
        "git-archive",
        "git-wheel-build",
        "git-wheel-install",
    ):
        if operation not in source_manager:
            fail(f"source operation report label is missing: {operation}", failures)
    try:
        sink = RuntimeReportSink(ROOT / ".torch-dae", "reports", "validation-smoke")
        ref = sink.record_event(
            operation="validation",
            status="failed",
            arguments=("https://user:secret@example.invalid/file?token=secret",),
            stderr="Authorization: Bearer secret",
        )
        payload = json.loads((ROOT / ".torch-dae" / ref).read_text())
        serialized = json.dumps(payload)
        if (
            payload["status"] != "failed"
            or "user:secret" in serialized
            or "Bearer secret" in serialized
            or "token=secret" in serialized
        ):
            fail("runtime report sink did not redact or persist expected fields", failures)
    except Exception as exc:
        fail(f"runtime report sink behavioral smoke failed: {exc}", failures)
    if not git_ignored(".torch-dae/reports/environments/card/hash/log.json"):
        fail(".torch-dae diagnostic reports are not ignored", failures)
    phase01_readme = ROOT / "tests/fixtures/phase01/README.md"
    if not phase01_readme.exists() or "synthetic" not in phase01_readme.read_text().lower():
        fail("Phase 01 fixture area is missing a synthetic marker", failures)
    checkpoint_tests = (ROOT / "tests/core/test_phase01_checkpoint.py").read_text()
    source_tests = (ROOT / "tests/environment/test_phase01_sources.py").read_text()
    materialization_tests = (ROOT / "tests/environment/test_phase01_materialization.py").read_text()
    if "example.invalid" not in checkpoint_tests:
        fail("checkpoint tests do not document fake public-network endpoints", failures)
    if (
        "test_phase01_package_bundle_checkpoint_uses_owned_distribution_file_offline"
        not in checkpoint_tests
    ):
        fail("Phase 01 package-bundle checkpoint integration coverage is missing", failures)
    if "test_phase01_git_source_build_metadata_workspace_and_offline_reuse" not in source_tests:
        fail("Phase 01 Git source cache/reuse coverage is missing", failures)
    if (
        "@pytest.mark.integration\n"
        "def test_phase01_git_source_build_metadata_workspace_and_offline_reuse" in source_tests
    ):
        fail("simulated Git source state-machine test is incorrectly marked integration", failures)
    if (
        "test_phase01_git_source_real_local_repository_installs_and_reuses_offline"
        not in source_tests
    ):
        fail("real local-Git source integration coverage is missing", failures)
    real_git_body = source_tests.split(
        "def test_phase01_git_source_real_local_repository_installs_and_reuses_offline",
        1,
    )[1].split("def run_git_command", 1)[0]
    if "GitSourceRunner" in real_git_body:
        fail("real local-Git integration test still uses the fake Git runner", failures)
    for required in (
        "test_phase01_git_source_online_recovers_invalid_checkout",
        "test_phase01_git_source_offline_invalid_checkout_fails_without_mutation",
        "dirty-offline",
        'remote", "set-url"',
        "other_revision",
        'metadata.write_text("{bad json")',
    ):
        if required not in source_tests:
            fail(f"Git invalid-cache recovery coverage is missing: {required}", failures)
    for required in (
        "phase01-synthetic-shared-environment",
        "test_phase01_environment_load_rejects_cross_document_path_mismatch",
        "test_phase01_failed_uv_sync_metadata_references_reports",
        "test_phase01_failed_git_clone_metadata_references_reports",
        "test_phase01_failed_git_wheel_build_metadata_references_reports",
        "failed_materialization_metadata",
        "command_log_references",
        "remove-wheel-json",
        "malformed-wheel-json",
    ):
        if required not in materialization_tests:
            fail(f"environment identity/cache regression coverage is missing: {required}", failures)
    environment_tests = (ROOT / "tests/environment/test_environment.py").read_text()
    for required in ("README.md", "package-data.json", "vendor/source.txt", "new_module.py"):
        if required not in environment_tests:
            fail(f"local package identity input coverage is missing: {required}", failures)
    if "test_phase01_local_wheel_backend_build_is_reproducible" not in materialization_tests:
        fail("Phase 01 reproducible local wheel coverage is missing", failures)
    checkpoint_module = (ROOT / "src/torch_dae/core/checkpoint.py").read_text()
    for required in (
        "reports",
        "checkpoints",
        "remote-open",
        "remote-stream",
        "remote-finalize",
        "hash-validation",
        "offline-cache-lookup",
        "local-path-copy",
        "package-bundle-lookup",
        "package-bundle-copy",
        "cache-finalize",
        "metadata-write",
        "response-close",
        "failure-cleanup",
        "command_log_references",
    ):
        if required not in checkpoint_module:
            fail(f"checkpoint diagnostic implementation is missing: {required}", failures)
    for required in (
        "checkpoint download stream failed",
        "checkpoint_failure_classification",
        "expected_hash_mismatch",
        "observed_hash_mismatch",
        "offline_cache_miss",
        "checkpoint metadata write failed",
        "checkpoint copy failed",
        "checkpoint cache finalization failed",
        "checkpoint response close failed",
        "HTTPError",
        "body.close()",
    ):
        if required not in checkpoint_module:
            fail(f"checkpoint failure normalization is missing: {required}", failures)
    for required in (
        "test_phase01_package_bundle_rejects_file_owned_by_other_distribution",
        "test_phase01_remote_response_is_closed_for_success_and_http_failure",
        "test_phase01_remote_response_is_closed_for_hash_mismatch",
        "test_phase01_interrupted_remote_read_cleans_partial_state_and_reports_failure",
        "test_phase01_transport_open_oserror_is_typed_reported_and_redacted",
        "test_phase01_urllib_transport_urlerror_is_typed_and_sanitized",
        "test_phase01_urllib_transport_httperror_closes_body",
        "test_phase01_local_copy_failure_is_typed_reported_and_cleans_tmp",
        "test_phase01_cache_finalize_failure_is_typed_reported_and_cleans_tmp",
        "test_phase01_metadata_write_failure_removes_incomplete_cache_entry_and_reports",
        "test_phase01_package_bundle_malformed_lookup_is_typed_and_reported",
        "test_phase01_successful_response_close_failure_is_reported_and_typed",
        "test_phase01_failed_response_close_does_not_mask_stream_error",
        "closed_observed",
        "hash-validation",
        "offline-cache-lookup",
        "metadata-write",
        "response-close",
        "failure-cleanup",
        "secret-token",
        "redacted secret",
    ):
        if required not in checkpoint_tests:
            fail(f"checkpoint evidence regression coverage is missing: {required}", failures)
    if "with pytest.raises(OSError)" in checkpoint_tests:
        fail("interrupted checkpoint stream coverage still expects raw OSError", failures)
    if (
        "test_phase01_checkpoint_cli_expected_acquisition_failures_are_concise"
        not in (ROOT / "tests/cli/test_phase01_cli.py").read_text()
    ):
        fail("real checkpoint CLI operational-failure coverage is missing", failures)

    report: dict[str, Any] = {"ok": not failures, "failures": failures}
    report_dir = ROOT / ".torch-dae/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "phase02-validation.json").write_text(json.dumps(report, indent=2))

    if failures:
        for item in failures:
            print(f"FAIL: {item}")
        return 1
    print("Phase 02 repository validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
