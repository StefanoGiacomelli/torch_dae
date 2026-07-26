from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from torch_dae.onboarding.contracts import (
    AnalysisReport,
    EnvironmentCandidateGenerationResult,
    EnvironmentResolutionReport,
    EvidenceItem,
    SourceStrategy,
)
from torch_dae.onboarding.inspection import generate_environment_candidates


def environment_resolution_data(repo_root: Path) -> dict[str, object]:
    return json.loads(
        (
            repo_root / "tests/fixtures/valid/environment-resolution-report.synthetic.json"
        ).read_text()
    )


def test_analysis_report_rejects_unresolved_evidence_reference(repo_root: Path) -> None:
    path = repo_root / "tests/fixtures/invalid/analysis-report.unresolved-evidence-reference.json"
    with pytest.raises(ValidationError, match="evidence references are unresolved"):
        AnalysisReport.model_validate_json(path.read_text())


def test_analysis_report_rejects_agent_inference_as_verified_evidence(
    repo_root: Path,
) -> None:
    data = json.loads(
        (repo_root / "tests/fixtures/valid/analysis-report.synthetic.json").read_text()
    )
    data["evidence_items"].append(
        {
            "evidence_id": "ev-inference",
            "kind": "agent_inference",
            "claim_status": "reasoned_inference",
            "description": "Inferred only.",
            "source_file": None,
            "source_line_or_symbol": None,
            "url": None,
            "revision": None,
            "rationale": "Synthetic inference.",
        }
    )
    data["official_status"]["evidence_ids"] = ["ev-inference"]
    data["official_status"]["status"] = "verified_upstream_fact"
    data["repository"]["official_status"] = data["official_status"]
    with pytest.raises(ValidationError, match="verified claims cite incompatible evidence"):
        AnalysisReport.model_validate(data)


@pytest.mark.parametrize(
    ("kind", "claim_status", "source_file", "accepted"),
    [
        ("runtime_observation", "verified_upstream_fact", None, False),
        ("agent_inference", "reasoned_inference", None, False),
        ("user_decision", "user_provided_decision", None, False),
        ("source_file", "verified_upstream_fact", "pyproject.toml", True),
        ("official_documentation", "verified_upstream_fact", None, True),
    ],
)
def test_analysis_report_verified_upstream_provenance(
    repo_root: Path,
    kind: str,
    claim_status: str,
    source_file: str | None,
    accepted: bool,
) -> None:
    data = json.loads(
        (repo_root / "tests/fixtures/valid/analysis-report.synthetic.json").read_text()
    )
    evidence = {
        "evidence_id": "ev-authority",
        "kind": kind,
        "claim_status": claim_status,
        "description": "Synthetic provenance boundary evidence.",
        "source_file": source_file,
    }
    if kind == "agent_inference":
        evidence["rationale"] = "This remains an inference."
    data["evidence_items"].append(evidence)
    data["source_strategy_candidates"][0]["status"] = "verified_upstream_fact"
    data["source_strategy_candidates"][0]["evidence_ids"] = ["ev-authority"]
    if accepted:
        data["confidence_summary"]["verified_fact_count"] += 2
        data["confidence_summary"]["unresolved_count"] -= 1
        AnalysisReport.model_validate(data)
    else:
        with pytest.raises(ValidationError, match="verified source strategys"):
            AnalysisReport.model_validate(data)


@pytest.mark.parametrize(
    "source_file",
    [
        "environments/card/environment.json",
        "verification_reports/card/report.json",
        "reports/environments/card/fingerprint/report.json",
        ".torch-dae/reports/card/report.json",
        "model_cards/card.json",
    ],
)
def test_verified_upstream_fact_rejects_generated_project_evidence(
    repo_root: Path,
    source_file: str,
) -> None:
    data = json.loads(
        (repo_root / "tests/fixtures/valid/analysis-report.synthetic.json").read_text()
    )
    data["evidence_items"].append(
        {
            "evidence_id": "ev-generated",
            "kind": "source_file",
            "claim_status": "verified_upstream_fact",
            "description": "Generated project artifact.",
            "source_file": source_file,
        }
    )
    data["source_strategy_candidates"][0]["status"] = "verified_upstream_fact"
    data["source_strategy_candidates"][0]["evidence_ids"] = ["ev-generated"]
    data["confidence_summary"]["verified_fact_count"] += 2
    data["confidence_summary"]["unresolved_count"] -= 1
    with pytest.raises(ValidationError, match="verified source strategys"):
        AnalysisReport.model_validate(data)


def test_locally_observed_behavior_accepts_generated_environment_evidence(
    repo_root: Path,
) -> None:
    data = json.loads(
        (repo_root / "tests/fixtures/valid/analysis-report.synthetic.json").read_text()
    )
    data["evidence_items"].append(
        {
            "evidence_id": "ev-generated",
            "kind": "source_file",
            "claim_status": "locally_observed_behavior",
            "description": "Locally observed generated environment artifact.",
            "source_file": "environments/card/environment.json",
        }
    )
    data["source_strategy_candidates"][0]["status"] = "locally_observed_behavior"
    data["source_strategy_candidates"][0]["evidence_ids"] = ["ev-generated"]
    data["confidence_summary"]["locally_observed_count"] += 2
    data["confidence_summary"]["unresolved_count"] -= 1
    AnalysisReport.model_validate(data)


@pytest.mark.parametrize(
    ("kind", "claim_status", "source_file", "accepted"),
    [
        ("runtime_observation", "verified_upstream_fact", None, False),
        ("agent_inference", "reasoned_inference", None, False),
        ("user_decision", "user_provided_decision", None, False),
        ("source_file", "verified_upstream_fact", "pyproject.toml", True),
        ("official_documentation", "verified_upstream_fact", None, True),
    ],
)
def test_environment_source_strategy_verified_upstream_provenance(
    repo_root: Path,
    kind: str,
    claim_status: str,
    source_file: str | None,
    accepted: bool,
) -> None:
    data = generate_environment_candidates(
        repo_root / "tests/skills/fixtures/synthetic_onboarding/official_package"
    )
    evidence = {
        "evidence_id": "ev-authority",
        "kind": kind,
        "claim_status": claim_status,
        "description": "Synthetic provenance boundary evidence.",
        "source_file": source_file,
    }
    if kind == "agent_inference":
        evidence["rationale"] = "This remains an inference."
    data["evidence_items"].append(evidence)
    data["source_strategy_context"][0]["status"] = "verified_upstream_fact"
    data["source_strategy_context"][0]["evidence_ids"] = ["ev-authority"]
    if accepted:
        EnvironmentCandidateGenerationResult.model_validate(data)
    else:
        with pytest.raises(ValidationError, match="verified source strategys"):
            EnvironmentCandidateGenerationResult.model_validate(data)


def test_inference_evidence_requires_rationale() -> None:
    with pytest.raises(ValidationError, match="rationale"):
        EvidenceItem.model_validate(
            {
                "evidence_id": "ev-inference",
                "kind": "agent_inference",
                "claim_status": "reasoned_inference",
                "description": "Inference without rationale.",
            }
        )


def test_environment_resolution_failed_candidate_requires_classification(repo_root: Path) -> None:
    path = (
        repo_root
        / "tests/fixtures/invalid/environment-resolution-report.failed-without-classification.json"
    )
    with pytest.raises(ValidationError, match="failed candidates require failure_classification"):
        EnvironmentResolutionReport.model_validate_json(path.read_text())


def test_environment_candidate_generation_rejects_dangling_dependency_evidence(
    repo_root: Path,
) -> None:
    data = generate_environment_candidates(
        repo_root / "tests/skills/fixtures/synthetic_onboarding/pinned_git"
    )
    data["evidence_items"] = [
        item for item in data["evidence_items"] if item["evidence_id"] != "ev-pyproject-toml"
    ]
    with pytest.raises(ValidationError, match="evidence references are unresolved"):
        EnvironmentCandidateGenerationResult.model_validate(data)


def test_environment_candidate_generation_rejects_dangling_source_strategy_evidence(
    repo_root: Path,
) -> None:
    data = generate_environment_candidates(
        repo_root / "tests/skills/fixtures/synthetic_onboarding/pinned_git"
    )
    data["source_strategy_context"][0]["evidence_ids"] = ["ev-missing-source"]
    with pytest.raises(ValidationError, match="evidence references are unresolved"):
        EnvironmentCandidateGenerationResult.model_validate(data)


def test_environment_candidate_generation_rejects_dangling_candidate_evidence(
    repo_root: Path,
) -> None:
    data = generate_environment_candidates(
        repo_root / "tests/skills/fixtures/synthetic_onboarding/official_package"
    )
    data["candidates"][0]["expected_compatibility_evidence"] = ["ev-missing-candidate"]
    with pytest.raises(ValidationError, match="evidence references are unresolved"):
        EnvironmentCandidateGenerationResult.model_validate(data)


def test_environment_candidate_generation_rejects_dangling_decision_gate_evidence(
    repo_root: Path,
) -> None:
    data = generate_environment_candidates(
        repo_root / "tests/skills/fixtures/synthetic_onboarding/minimal_vendoring"
    )
    assert data["decision_gates"]
    data["decision_gates"][0]["evidence_ids"] = ["ev-missing-gate"]
    with pytest.raises(ValidationError, match="evidence references are unresolved"):
        EnvironmentCandidateGenerationResult.model_validate(data)


def test_environment_resolution_rejects_unsafe_promotion_states(repo_root: Path) -> None:
    data = environment_resolution_data(repo_root)

    unsupported = json.loads(json.dumps(data))
    unsupported["ordered_candidates"][0]["installation_strategy"] = (
        SourceStrategy.UNSUPPORTED_OR_NON_EQUIVALENT_IMPLEMENTATION.value
    )
    with pytest.raises(ValidationError, match="unsupported source strategies"):
        EnvironmentResolutionReport.model_validate(unsupported)

    missing_revision = json.loads(json.dumps(data))
    missing_revision["ordered_candidates"][0]["installation_strategy"] = (
        SourceStrategy.PINNED_OFFICIAL_GIT_REPOSITORY.value
    )
    with pytest.raises(ValidationError, match="source_revision"):
        EnvironmentResolutionReport.model_validate(missing_revision)

    non_sha_revision = json.loads(json.dumps(missing_revision))
    non_sha_revision["ordered_candidates"][0]["source_revision"] = "main"
    with pytest.raises(ValidationError, match="source_revision"):
        EnvironmentResolutionReport.model_validate(non_sha_revision)

    arbitrary_report = json.loads(json.dumps(data))
    arbitrary_report["verification_report_or_diagnostic_reference"] = "README.md"
    with pytest.raises(ValidationError, match="recognized verification or diagnostic"):
        EnvironmentResolutionReport.model_validate(arbitrary_report)

    open_source_gate = json.loads(json.dumps(data))
    open_source_gate["source_strategy_decision_gates"] = [
        {
            "question_id": "q-source-strategy",
            "classification": "needs_user_decision",
            "description": "Choose a source strategy.",
            "alternatives": ["official_package", "pinned_official_git_repository"],
            "evidence_ids": ["ev-package"],
            "default_if_deferred": "do not promote",
            "failure_classification": None,
        }
    ]
    with pytest.raises(ValidationError, match="source-strategy decision gates"):
        EnvironmentResolutionReport.model_validate(open_source_gate)


def test_environment_candidate_generation_rejects_semantically_incompatible_evidence(
    repo_root: Path,
) -> None:
    data = generate_environment_candidates(
        repo_root / "tests/skills/fixtures/synthetic_onboarding/official_package"
    )

    verified_agent = json.loads(json.dumps(data))
    verified_agent["evidence_items"].append(
        {
            "evidence_id": "ev-agent",
            "kind": "agent_inference",
            "claim_status": "reasoned_inference",
            "description": "Inferred source strategy.",
            "rationale": "Synthetic inference.",
        }
    )
    verified_agent["source_strategy_context"][0]["status"] = "verified_upstream_fact"
    verified_agent["source_strategy_context"][0]["evidence_ids"] = ["ev-agent"]
    with pytest.raises(ValidationError, match="verified source strategys"):
        EnvironmentCandidateGenerationResult.model_validate(verified_agent)

    dependency_user_decision = json.loads(json.dumps(data))
    first_record = dependency_user_decision["dependency_records"][0]
    dependency_user_decision["evidence_items"] = [
        {
            "evidence_id": first_record["evidence_id"],
            "kind": "user_decision",
            "claim_status": "user_provided_decision",
            "description": "User chose dependency evidence.",
        }
    ] + [
        item
        for item in dependency_user_decision["evidence_items"]
        if item["evidence_id"] != first_record["evidence_id"]
    ]
    with pytest.raises(ValidationError, match="locally observed dependency records"):
        EnvironmentCandidateGenerationResult.model_validate(dependency_user_decision)

    missing_rationale = json.loads(json.dumps(data))
    missing_rationale["candidates"][0]["reason_for_selection"] = " "
    with pytest.raises(ValidationError, match="reasoned inference environment candidates"):
        EnvironmentCandidateGenerationResult.model_validate(missing_rationale)


def test_analysis_report_rejects_user_decision_citing_source_file(repo_root: Path) -> None:
    data = json.loads(
        (repo_root / "tests/fixtures/valid/analysis-report.synthetic.json").read_text()
    )
    data["decisions"].append(
        {
            "decision_id": "decision-user",
            "decision": "Synthetic user decision.",
            "selected_option": "selected",
            "status": "user_provided_decision",
            "evidence_ids": ["ev-static"],
            "rationale": None,
        }
    )
    with pytest.raises(ValidationError, match="user-provided decisions"):
        AnalysisReport.model_validate(data)


def test_decision_gate_unsupported_evidence_rules(repo_root: Path) -> None:
    data = generate_environment_candidates(
        repo_root / "tests/skills/fixtures/synthetic_onboarding/minimal_vendoring"
    )
    data["evidence_items"].append(
        {
            "evidence_id": "ev-unsupported",
            "kind": "source_file",
            "claim_status": "unsupported_claim",
            "description": "Unsupported upstream evidence.",
            "source_file": "README.md",
        }
    )
    ordinary_gate = json.loads(json.dumps(data))
    ordinary_gate["decision_gates"][0]["evidence_ids"] = ["ev-unsupported"]
    with pytest.raises(ValidationError, match="cannot cite only unsupported evidence"):
        EnvironmentCandidateGenerationResult.model_validate(ordinary_gate)

    unsupported_gate = json.loads(json.dumps(data))
    unsupported_gate["decision_gates"] = [
        {
            "question_id": "q-unsupported",
            "classification": "unsupported_upstream_claim",
            "description": "Unsupported upstream claim.",
            "alternatives": [],
            "evidence_ids": ["ev-unsupported"],
            "default_if_deferred": None,
            "failure_classification": None,
        }
    ]
    EnvironmentCandidateGenerationResult.model_validate(unsupported_gate)


def test_environment_candidate_generation_valid_semantic_evidence(repo_root: Path) -> None:
    data = generate_environment_candidates(
        repo_root / "tests/skills/fixtures/synthetic_onboarding/pinned_git"
    )
    result = EnvironmentCandidateGenerationResult.model_validate(data)
    assert result.dependency_records
    assert result.source_strategy_context
    assert result.candidates
    assert result.decision_gates


def test_official_package_identity_validation(repo_root: Path) -> None:
    data = environment_resolution_data(repo_root)

    missing_name = json.loads(json.dumps(data))
    missing_name["ordered_candidates"][0]["source_package_name"] = None
    with pytest.raises(ValidationError, match="source_package_name"):
        EnvironmentResolutionReport.model_validate(missing_name)

    missing_version = json.loads(json.dumps(data))
    missing_version["ordered_candidates"][0]["source_package_version"] = None
    with pytest.raises(ValidationError, match="source_package_name"):
        EnvironmentResolutionReport.model_validate(missing_version)

    invalid_version = json.loads(json.dumps(data))
    invalid_version["ordered_candidates"][0]["source_package_version"] = "not a version !"
    with pytest.raises(ValidationError, match="source_package_version"):
        EnvironmentResolutionReport.model_validate(invalid_version)

    generic_evidence = json.loads(json.dumps(data))
    for item in generic_evidence["evidence_items"]:
        if item["evidence_id"] == "ev-package":
            item["package_name"] = None
            item["package_version"] = None
    with pytest.raises(ValidationError, match="package/version evidence"):
        EnvironmentResolutionReport.model_validate(generic_evidence)

    mismatched_name = json.loads(json.dumps(data))
    for item in mismatched_name["evidence_items"]:
        if item["evidence_id"] == "ev-package":
            item["package_name"] = "different-package"
    with pytest.raises(ValidationError, match="package/version evidence"):
        EnvironmentResolutionReport.model_validate(mismatched_name)

    mismatched_version = json.loads(json.dumps(data))
    for item in mismatched_version["evidence_items"]:
        if item["evidence_id"] == "ev-package":
            item["package_version"] = "9.9.9"
    with pytest.raises(ValidationError, match="package/version evidence"):
        EnvironmentResolutionReport.model_validate(mismatched_version)

    not_environment_artifact = json.loads(json.dumps(data))
    for item in not_environment_artifact["evidence_items"]:
        if item["evidence_id"] == "ev-package":
            item["source_file"] = "pyproject.toml"
    with pytest.raises(ValidationError, match="environment artifacts"):
        EnvironmentResolutionReport.model_validate(not_environment_artifact)

    EnvironmentResolutionReport.model_validate(data)


@pytest.mark.parametrize(
    ("kind", "claim_status", "source_file", "accepted"),
    [
        ("agent_inference", "reasoned_inference", None, False),
        ("runtime_observation", "locally_observed_behavior", None, False),
        (
            "package_metadata",
            "locally_observed_behavior",
            "environments/synthetic/verify_environment.py",
            False,
        ),
        (
            "package_metadata",
            "locally_observed_behavior",
            "environments/synthetic/sources.json",
            False,
        ),
        (
            "package_metadata",
            "locally_observed_behavior",
            "environments/wrong/pyproject.toml",
            False,
        ),
        ("package_metadata", "verified_upstream_fact", "pyproject.toml", True),
        (
            "package_metadata",
            "verified_upstream_fact",
            "upstream-package/synthetic.dist-info/METADATA",
            True,
        ),
        (
            "configuration_file",
            "locally_observed_behavior",
            "environments/synthetic/pyproject.toml",
            True,
        ),
        (
            "package_metadata",
            "locally_observed_behavior",
            "environments/synthetic/uv.lock",
            True,
        ),
        (
            "configuration_file",
            "locally_observed_behavior",
            "environments/synthetic/environment.json",
            True,
        ),
    ],
)
def test_official_package_identity_requires_semantic_provenance(
    repo_root: Path,
    kind: str,
    claim_status: str,
    source_file: str | None,
    accepted: bool,
) -> None:
    data = environment_resolution_data(repo_root)
    package_evidence = next(
        item for item in data["evidence_items"] if item["evidence_id"] == "ev-package"
    )
    package_evidence["kind"] = kind
    package_evidence["claim_status"] = claim_status
    package_evidence["source_file"] = source_file
    if kind == "agent_inference":
        package_evidence["rationale"] = "An inference cannot establish package identity."
    if accepted:
        EnvironmentResolutionReport.model_validate(data)
    else:
        with pytest.raises(ValidationError, match="package/version evidence"):
            EnvironmentResolutionReport.model_validate(data)


def test_official_package_identity_rejects_arbitrary_package_metadata_file(
    repo_root: Path,
) -> None:
    data = environment_resolution_data(repo_root)
    package_evidence = next(
        item for item in data["evidence_items"] if item["evidence_id"] == "ev-package"
    )
    package_evidence["claim_status"] = "verified_upstream_fact"
    package_evidence["source_file"] = "README.md"
    with pytest.raises(ValidationError, match="package/version evidence"):
        EnvironmentResolutionReport.model_validate(data)


def test_official_package_identity_rejects_url_only_metadata(repo_root: Path) -> None:
    data = environment_resolution_data(repo_root)
    package_evidence = next(
        item for item in data["evidence_items"] if item["evidence_id"] == "ev-package"
    )
    package_evidence["claim_status"] = "verified_upstream_fact"
    package_evidence["source_file"] = None
    package_evidence["url"] = "https://example.invalid/arbitrary-package-metadata"
    with pytest.raises(ValidationError, match="package/version evidence"):
        EnvironmentResolutionReport.model_validate(data)


@pytest.mark.parametrize(
    "question_id",
    [
        "q-source-strategy",
        "q-origin-choice",
        "q-implementation-selection",
        "q-repository-decision",
    ],
)
def test_any_source_strategy_gate_blocks_promotion(repo_root: Path, question_id: str) -> None:
    data = environment_resolution_data(repo_root)
    data["source_strategy_decision_gates"] = [
        {
            "question_id": question_id,
            "classification": "needs_user_decision",
            "description": "Choose a source strategy.",
            "alternatives": ["official_package", "pinned_official_git_repository"],
            "evidence_ids": ["ev-package"],
            "default_if_deferred": "do not promote",
            "failure_classification": None,
        }
    ]
    with pytest.raises(ValidationError, match="source-strategy decision gates"):
        EnvironmentResolutionReport.model_validate(data)


def test_environment_report_reference_formats(repo_root: Path, tmp_path: Path) -> None:
    from torch_dae.environment.runtime import RuntimeReportSink

    data = environment_resolution_data(repo_root)
    fingerprint = data["environment_fingerprint"]
    sink = RuntimeReportSink(
        tmp_path / ".torch-dae",
        "reports",
        "environments",
        "synthetic",
        fingerprint,
    )
    diagnostic = sink.record_event(operation="verification-script", status="success")

    diagnostic_data = json.loads(json.dumps(data))
    diagnostic_data["verification_report_or_diagnostic_reference"] = diagnostic
    EnvironmentResolutionReport.model_validate(diagnostic_data)

    committed_report = json.loads(json.dumps(data))
    committed_report["verification_report_or_diagnostic_reference"] = (
        "verification_reports/synthetic/environment-check.json"
    )
    EnvironmentResolutionReport.model_validate(committed_report)

    rejected = {
        "reports/checkpoints/synthetic/check.json": "recognized verification or diagnostic",
        "unrelated/report.json": "recognized verification or diagnostic",
        f"reports/environments/wrong/{fingerprint}/check.json": "recognized verification",
        "reports/environments/synthetic/not-a-fingerprint/check.json": "recognized verification",
    }
    for reference, message in rejected.items():
        bad = json.loads(json.dumps(data))
        bad["verification_report_or_diagnostic_reference"] = reference
        with pytest.raises(ValidationError, match=message):
            EnvironmentResolutionReport.model_validate(bad)
