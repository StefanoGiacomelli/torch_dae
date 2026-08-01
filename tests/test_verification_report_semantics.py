from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from torch_dae.environment.verification import VerificationReport


def load_legacy_report(valid_fixture_dir: Path) -> dict[str, object]:
    return json.loads((valid_fixture_dir / "verification-report.synthetic.json").read_text())


def target_aware_report(valid_fixture_dir: Path) -> dict[str, object]:
    data = load_legacy_report(valid_fixture_dir)
    data.update(
        {
            "schema_version": "2.0.0",
            "runtime_target_id": "synthetic-runtime-target",
            "workflow_id": "synthetic-workflow",
            "integrated_variant_id": "synthetic-variant",
            "checkpoint_id": "synthetic-checkpoint",
            "public_model_entry_point": "torch_dae.synthetic:AudioModel",
            "integration_handoff_sha256": "1" * 64,
            "environment_spec_sha256": "2" * 64,
            "source_manifest_sha256": "3" * 64,
            "required_check_ids": ["canonical-input", "checkpoint-loaded-forward"],
            "optional_check_ids": ["valid-lengths"],
            "verification_status": "passed",
        }
    )
    data["unsupported_capabilities"] = []
    data["checks"] = [
        {"name": "canonical-input", "status": "passed", "details": None},
        {"name": "checkpoint-loaded-forward", "status": "passed", "details": None},
    ]
    return data


def test_successful_target_aware_report_has_no_failed_checks(valid_fixture_dir: Path) -> None:
    report = VerificationReport.model_validate(target_aware_report(valid_fixture_dir))

    assert report.verification_status == "passed"
    assert report.successful


def test_passed_target_aware_report_rejects_failed_check(valid_fixture_dir: Path) -> None:
    data = target_aware_report(valid_fixture_dir)
    data["checks"][0]["status"] = "failed"

    with pytest.raises(ValidationError, match="cannot contain failed checks"):
        VerificationReport.model_validate(data)


def test_passed_target_aware_report_rejects_empty_checks(valid_fixture_dir: Path) -> None:
    data = target_aware_report(valid_fixture_dir)
    data["checks"] = []

    with pytest.raises(ValidationError, match="nonempty checks"):
        VerificationReport.model_validate(data)


def test_passed_target_aware_report_rejects_missing_required_check(
    valid_fixture_dir: Path,
) -> None:
    data = target_aware_report(valid_fixture_dir)
    data["checks"].pop()

    with pytest.raises(ValidationError, match="missing required checks"):
        VerificationReport.model_validate(data)


def test_passed_target_aware_report_rejects_unsupported_required_check(
    valid_fixture_dir: Path,
) -> None:
    data = target_aware_report(valid_fixture_dir)
    data["checks"][0] = {
        "name": "canonical-input",
        "status": "unsupported",
        "details": "The required operation was unavailable.",
    }
    data["unsupported_capabilities"] = ["canonical-input"]
    data["known_limitations"] = ["The required operation was unavailable."]

    with pytest.raises(ValidationError, match="cannot be unsupported"):
        VerificationReport.model_validate(data)


def test_passed_target_aware_report_rejects_undeclared_check(valid_fixture_dir: Path) -> None:
    data = target_aware_report(valid_fixture_dir)
    data["checks"].append({"name": "extra-check", "status": "passed", "details": None})

    with pytest.raises(ValidationError, match="undeclared checks"):
        VerificationReport.model_validate(data)


def test_passed_target_aware_report_rejects_duplicate_check_names(
    valid_fixture_dir: Path,
) -> None:
    data = target_aware_report(valid_fixture_dir)
    data["checks"].append({"name": "canonical-input", "status": "passed", "details": "duplicate"})

    with pytest.raises(ValidationError, match="names must be unique"):
        VerificationReport.model_validate(data)


def test_failed_target_aware_report_requires_failed_evidence(valid_fixture_dir: Path) -> None:
    data = target_aware_report(valid_fixture_dir)
    data["verification_status"] = "failed"

    with pytest.raises(ValidationError, match="requires a failed check"):
        VerificationReport.model_validate(data)


def test_failed_target_aware_report_is_valid_diagnostic_evidence(valid_fixture_dir: Path) -> None:
    data = target_aware_report(valid_fixture_dir)
    data["verification_status"] = "failed"
    data["checks"][0]["status"] = "failed"

    report = VerificationReport.model_validate(data)

    assert not report.successful


def test_unsupported_capability_does_not_create_false_failure(valid_fixture_dir: Path) -> None:
    data = target_aware_report(valid_fixture_dir)
    data["unsupported_capabilities"] = ["valid-lengths"]
    data["known_limitations"] = ["Padded variable-length batches are unsupported."]
    data["checks"].append(
        {
            "name": "valid-lengths",
            "status": "unsupported",
            "details": "The public adapter rejects padded variable-length batches.",
        }
    )

    report = VerificationReport.model_validate(data)

    assert report.successful


def test_unsupported_check_requires_declared_capability_and_limitation(
    valid_fixture_dir: Path,
) -> None:
    data = target_aware_report(valid_fixture_dir)
    data["checks"].append(
        {
            "name": "valid-lengths",
            "status": "unsupported",
            "details": "Unsupported by the adapter.",
        }
    )

    with pytest.raises(ValidationError, match="unsupported_capabilities"):
        VerificationReport.model_validate(data)


def test_legacy_reports_remain_readable_with_conservative_success(
    valid_fixture_dir: Path,
) -> None:
    report = VerificationReport.model_validate(load_legacy_report(valid_fixture_dir))
    assert report.successful

    failed = load_legacy_report(valid_fixture_dir)
    failed["checks"][0]["status"] = "failed"
    failed_report = VerificationReport.model_validate(failed)
    assert not failed_report.successful
