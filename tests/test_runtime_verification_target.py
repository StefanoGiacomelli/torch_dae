from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from torch_dae.onboarding.handoff import skill_fingerprint
from torch_dae.runtime_verification import (
    RuntimeVerificationTarget,
    validate_runtime_verification_target,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_target(valid_fixture_dir: Path) -> RuntimeVerificationTarget:
    return RuntimeVerificationTarget.model_validate_json(
        (valid_fixture_dir / "runtime-verification-target.synthetic.json").read_text()
    )


def test_runtime_targets_preserve_distinct_tuple_identities(valid_fixture_dir: Path) -> None:
    first = load_target(valid_fixture_dir)
    second_data = first.model_dump(mode="json")
    second_data["target_id"] = "synthetic-runtime-target-two"
    second_data["checkpoint"]["checkpoint_id"] = "synthetic-checkpoint-two"
    second = RuntimeVerificationTarget.model_validate(second_data)

    assert first.environment_id == second.environment_id
    assert first.target_id != second.target_id
    assert first.checkpoint.checkpoint_id != second.checkpoint.checkpoint_id


def test_runtime_target_allows_future_card_to_be_deferred(valid_fixture_dir: Path) -> None:
    data = load_target(valid_fixture_dir).model_dump(mode="json")
    data["future_card_id"] = None
    target = RuntimeVerificationTarget.model_validate(data)
    assert target.future_card_id is None


def test_runtime_target_requires_at_least_one_required_check(valid_fixture_dir: Path) -> None:
    data = load_target(valid_fixture_dir).model_dump(mode="json")
    data["required_check_ids"] = []

    with pytest.raises(ValidationError, match="at least one required check"):
        RuntimeVerificationTarget.model_validate(data)


@pytest.mark.parametrize("field", ["required_check_ids", "optional_check_ids"])
def test_runtime_target_rejects_duplicate_checks(valid_fixture_dir: Path, field: str) -> None:
    data = load_target(valid_fixture_dir).model_dump(mode="json")
    data[field] = [data[field][0], data[field][0]]

    with pytest.raises(ValidationError, match="check IDs must be unique"):
        RuntimeVerificationTarget.model_validate(data)


def test_runtime_target_rejects_required_optional_overlap(valid_fixture_dir: Path) -> None:
    data = load_target(valid_fixture_dir).model_dump(mode="json")
    data["optional_check_ids"] = [data["required_check_ids"][0]]

    with pytest.raises(ValidationError, match="must be disjoint"):
        RuntimeVerificationTarget.model_validate(data)


def test_legacy_runtime_target_remains_readable_via_explicit_migration_path(
    valid_fixture_dir: Path,
) -> None:
    data = load_target(valid_fixture_dir).model_dump(mode="json")
    data["schema_version"] = "1.0.0"
    data.pop("required_check_ids")
    data.pop("optional_check_ids")

    target = RuntimeVerificationTarget.model_validate(data)

    assert target.schema_version == "1.0.0"
    assert target.required_check_ids == ()


def test_authority_complete_runtime_target_binds_all_integrity_requirements(
    valid_fixture_dir: Path,
) -> None:
    data = load_target(valid_fixture_dir).model_dump(mode="json")
    data["checkpoint"] = {
        "schema_version": "2.0.0",
        "checkpoint_id": "synthetic-checkpoint",
        "source_type": "https",
        "filename": "model_mAP=0.438.pth",
        "authority": {
            "provider": "zenodo",
            "record_id": "12345",
            "filename": "model_mAP=0.438.pth",
            "expected_size_bytes": 100,
            "published_checksums": [{"algorithm": "md5", "digest": "a" * 32}],
            "provenance_status": "authoritative_provider_declared",
        },
        "format": "binary",
        "loader": "manual",
        "license": {"status": "not_applicable"},
    }
    data["checkpoint_acquisition_policy"] = {
        "allow_network": True,
        "allow_authentication": False,
        "maximum_bytes": 100,
        "require_expected_sha256": False,
        "require_authority": True,
        "require_exact_size": True,
        "require_published_checksums": True,
        "require_observed_sha256": True,
    }

    target = RuntimeVerificationTarget.model_validate(data)

    assert target.checkpoint.authority is not None


def test_authority_complete_runtime_target_rejects_incomplete_policy(
    valid_fixture_dir: Path,
) -> None:
    data = load_target(valid_fixture_dir).model_dump(mode="json")
    data["checkpoint_acquisition_policy"]["require_authority"] = True

    with pytest.raises(ValidationError, match="requires checkpoint authority"):
        RuntimeVerificationTarget.model_validate(data)


def test_runtime_target_requires_accepted_integration_handoff(
    tmp_path: Path,
    valid_fixture_dir: Path,
) -> None:
    target = load_target(valid_fixture_dir)
    with pytest.raises(ValueError):
        validate_runtime_verification_target(target, tmp_path)


def test_runtime_target_validates_real_synthetic_associations(
    tmp_path: Path,
    repo_root: Path,
    valid_fixture_dir: Path,
) -> None:
    (tmp_path / "project_spec.md").write_text("synthetic specification\n")
    shutil.copytree(
        repo_root / "skills/audio-model-onboarding",
        tmp_path / "skills/audio-model-onboarding",
    )
    environment = tmp_path / "environments/synthetic-environment"
    environment.mkdir(parents=True)
    (environment / "pyproject.toml").write_text(
        "[project]\nname='synthetic-environment'\nversion='0.0.0'\n"
        "requires-python='==3.12.13'\ndependencies=[]\n"
    )
    (environment / "uv.lock").write_text("version = 1\nrevision = 3\n")
    (environment / "sources.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "environment_id": "synthetic-environment",
                "sources": [],
            }
        )
    )
    (environment / "verify_environment.py").write_text("print('synthetic')\n")
    (environment / "environment.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "environment_id": "synthetic-environment",
                "python": {"constraint": "==3.12.13", "resolved_version": "3.12.13"},
                "platforms": {
                    "resolved_on": ["synthetic-platform"],
                    "expected_compatible": [],
                    "verified": [],
                },
                "dependency_manager": "uv",
                "lockfile": "environments/synthetic-environment/uv.lock",
                "project_file": "environments/synthetic-environment/pyproject.toml",
                "sources_file": "environments/synthetic-environment/sources.json",
                "verification": {
                    "script": "environments/synthetic-environment/verify_environment.py"
                },
            }
        )
    )
    workflow_root = tmp_path / "onboarding_reports/synthetic-workflow"
    integrate = workflow_root / "integrate"
    integrate.mkdir(parents=True)
    integration_report = integrate / "integration-report.json"
    integration_report.write_text("{}\n")
    handoff_path = integrate / "handoff.json"
    handoff = {
        "schema_version": "1.0.0",
        "workflow_id": "synthetic-workflow",
        "phase": "integrate",
        "handoff_status": "accepted",
        "repository_commit": "a" * 40,
        "project_spec_sha256": sha(tmp_path / "project_spec.md"),
        "canonical_skill_fingerprint": skill_fingerprint(tmp_path),
        "input_artifacts": [],
        "output_artifacts": [
            {
                "path": "onboarding_reports/synthetic-workflow/integrate/integration-report.json",
                "sha256": sha(integration_report),
                "media_type": "application/json",
                "originating_phase": "integrate",
                "canonical_role": "integration-report",
                "external_label": None,
            }
        ],
        "artifact_supersessions": [],
        "target_variant_ids": ["synthetic-variant"],
        "target_checkpoint_ids": ["synthetic-checkpoint"],
        "target_card_ids": ["synthetic-card"],
        "user_decisions_consumed": [],
        "unresolved_items_carried_forward": [],
        "validation_summary": {"passed": True, "checks": ["synthetic"], "errors": []},
        "allowed_next_modes": ["verify"],
        "lifecycle_promotion": None,
        "superseded_handoff_sha256": None,
    }
    handoff_path.write_text(json.dumps(handoff, indent=2))
    (workflow_root / "workflow.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "workflow_id": "synthetic-workflow",
                "model_family": "Synthetic",
                "target_variant_ids": ["synthetic-variant"],
                "target_checkpoint_ids": ["synthetic-checkpoint"],
                "target_card_ids": ["synthetic-card"],
                "created_repository_commit": "a" * 40,
                "current_accepted_phase": "integrate",
                "accepted_phase_paths": [
                    {
                        "phase": "integrate",
                        "handoff_path": (
                            "onboarding_reports/synthetic-workflow/integrate/handoff.json"
                        ),
                    }
                ],
                "status": "active",
            },
            indent=2,
        )
    )
    target_data = load_target(valid_fixture_dir).model_dump(mode="json")
    target_data["accepted_integration_handoff"]["sha256"] = sha(handoff_path)
    target_data["environment_spec_sha256"] = sha(environment / "environment.json")
    target_data["source_manifest"]["sha256"] = sha(environment / "sources.json")
    target = RuntimeVerificationTarget.model_validate(target_data)

    assert validate_runtime_verification_target(target, tmp_path) == target
