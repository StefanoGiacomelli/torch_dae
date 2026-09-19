from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from torch_dae.environment.results import EnvironmentVerificationResult


def load_result(valid_fixture_dir: Path) -> dict[str, object]:
    return json.loads(
        (valid_fixture_dir / "environment-verification-result.synthetic.json").read_text()
    )


def test_passed_environment_result_promotes_environment_lifecycle(
    valid_fixture_dir: Path,
) -> None:
    result = EnvironmentVerificationResult.model_validate(load_result(valid_fixture_dir))

    assert result.verification_status == "passed"
    assert result.lifecycle_state == "environment_verified"
    assert result.failure_classification is None


@pytest.mark.parametrize(
    ("collection", "status", "message"),
    [
        ("import_results", "failed", "every observation to pass"),
        ("environment_smoke_results", "failed", "every observation to pass"),
        ("import_results", "unsupported", "every observation to pass"),
    ],
)
def test_passed_environment_result_requires_passed_observations(
    valid_fixture_dir: Path,
    collection: str,
    status: str,
    message: str,
) -> None:
    data = load_result(valid_fixture_dir)
    data[collection][0]["status"] = status

    with pytest.raises(ValidationError, match=message):
        EnvironmentVerificationResult.model_validate(data)


@pytest.mark.parametrize(
    ("collection", "message"),
    [
        ("import_results", "requires import observations"),
        ("environment_smoke_results", "requires smoke observations"),
    ],
)
def test_passed_environment_result_requires_complete_observation_collections(
    valid_fixture_dir: Path,
    collection: str,
    message: str,
) -> None:
    data = load_result(valid_fixture_dir)
    data[collection] = []

    with pytest.raises(ValidationError, match=message):
        EnvironmentVerificationResult.model_validate(data)


def test_environment_observation_names_are_unique_across_collections(
    valid_fixture_dir: Path,
) -> None:
    data = load_result(valid_fixture_dir)
    data["environment_smoke_results"][0]["name"] = data["import_results"][0]["name"]

    with pytest.raises(ValidationError, match="names must be unique"):
        EnvironmentVerificationResult.model_validate(data)


def test_passed_environment_result_cannot_remain_materialized(
    valid_fixture_dir: Path,
) -> None:
    data = load_result(valid_fixture_dir)
    data["lifecycle_state"] = "materialized"

    with pytest.raises(ValidationError, match="environment_verified lifecycle state"):
        EnvironmentVerificationResult.model_validate(data)


def test_failed_environment_result_is_valid_diagnostic_evidence(
    valid_fixture_dir: Path,
) -> None:
    data = load_result(valid_fixture_dir)
    data["verification_status"] = "failed"
    data["lifecycle_state"] = "materialized"
    data["failure_classification"] = "verification_script"

    result = EnvironmentVerificationResult.model_validate(data)

    assert result.verification_status == "failed"
    assert result.lifecycle_state == "materialized"


def test_failed_environment_result_allows_absent_observations(
    valid_fixture_dir: Path,
) -> None:
    data = load_result(valid_fixture_dir)
    data.update(
        {
            "verification_status": "failed",
            "lifecycle_state": "materialized",
            "failure_classification": "verification_script",
            "import_results": [],
            "environment_smoke_results": [],
        }
    )

    result = EnvironmentVerificationResult.model_validate(data)

    assert result.import_results == ()
    assert result.environment_smoke_results == ()


def test_failed_environment_result_cannot_promote_environment_lifecycle(
    valid_fixture_dir: Path,
) -> None:
    data = load_result(valid_fixture_dir)
    data["verification_status"] = "failed"
    data["failure_classification"] = "verification_script"

    with pytest.raises(ValidationError, match="materialized lifecycle state"):
        EnvironmentVerificationResult.model_validate(data)


@pytest.mark.parametrize(
    ("verification_status", "failure_classification", "message"),
    [
        ("passed", "verification_script", "cannot carry failure_classification"),
        ("failed", None, "requires failure_classification"),
    ],
)
def test_environment_result_failure_classification_matches_status(
    valid_fixture_dir: Path,
    verification_status: str,
    failure_classification: str | None,
    message: str,
) -> None:
    data = load_result(valid_fixture_dir)
    data["verification_status"] = verification_status
    data["lifecycle_state"] = (
        "environment_verified" if verification_status == "passed" else "materialized"
    )
    data["failure_classification"] = failure_classification

    with pytest.raises(ValidationError, match=message):
        EnvironmentVerificationResult.model_validate(data)
