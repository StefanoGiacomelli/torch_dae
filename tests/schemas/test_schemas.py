from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import ValidationError as PydanticValidationError

from scripts.validate_repository import (
    load_json,
    pydantic_validate,
    schema_validate,
    validate_fixture,
)

SCHEMA_MAP = {
    "model-card": "model-card.schema.json",
    "checkpoint": "checkpoint.schema.json",
    "embedding": "embedding.schema.json",
    "environment": "environment.schema.json",
    "environment-sources": "environment-sources.schema.json",
    "verification-report": "verification-report.schema.json",
    "analysis-report": "analysis-report.schema.json",
    "environment-resolution-report": "environment-resolution-report.schema.json",
}


def test_valid_fixtures_pass_dual_validation(repo_root: Path, valid_fixture_dir: Path) -> None:
    for path in valid_fixture_dir.glob("*.json"):
        schema = repo_root / "schemas" / SCHEMA_MAP[path.name.split(".")[0]]
        validate_fixture(path, schema)


def test_invalid_fixtures_fail(repo_root: Path, invalid_fixture_dir: Path) -> None:
    manifest = load_json(repo_root / "tests/fixtures/invalid_manifest.json")
    for path in invalid_fixture_dir.glob("*.json"):
        if manifest.get(f"tests/fixtures/invalid/{path.name}") == "semantic_cross_reference":
            continue
        schema = repo_root / "schemas" / SCHEMA_MAP[path.name.split(".")[0]]
        with pytest.raises((PydanticValidationError, JsonSchemaValidationError)):
            validate_fixture(path, schema)


def test_invalid_fixture_manifest_declares_validation_responsibility(repo_root: Path) -> None:
    manifest = load_json(repo_root / "tests/fixtures/invalid_manifest.json")
    structural = semantic = 0
    for relative, classification in manifest.items():
        path = repo_root / relative
        schema = repo_root / "schemas" / SCHEMA_MAP[path.name.split(".")[0]]
        if classification == "structural":
            structural += 1
            with pytest.raises((PydanticValidationError, JsonSchemaValidationError)):
                pydantic_validate(path)
            with pytest.raises(JsonSchemaValidationError):
                schema_validate(path, schema)
        else:
            semantic += 1
            try:
                pydantic_validate(path)
            except PydanticValidationError:
                pass
    assert structural >= 20
    assert semantic >= 3


def test_generated_model_card_conditions_are_effective(
    repo_root: Path, invalid_fixture_dir: Path
) -> None:
    schema = repo_root / "schemas/model-card.schema.json"
    for name in [
        "model-card.runtime-without-report.json",
        "model-card.zero-default-embedding.json",
        "model-card.duplicate-default-embedding.json",
        "model-card.unsupported-capability-without-reason.json",
        "model-card.inferred-evidence-no-rationale.json",
        "model-card.profiled-section-without-report.json",
        "model-card.probability-capability-no-output.json",
    ]:
        with pytest.raises(JsonSchemaValidationError):
            schema_validate(invalid_fixture_dir / name, schema)


def test_checkpoint_and_embedding_schema_version_required(
    repo_root: Path, invalid_fixture_dir: Path
) -> None:
    for name, schema_name in [
        ("checkpoint.missing-schema-version.json", "checkpoint.schema.json"),
        ("embedding.missing-schema-version.json", "embedding.schema.json"),
    ]:
        with pytest.raises(JsonSchemaValidationError):
            schema_validate(invalid_fixture_dir / name, repo_root / "schemas" / schema_name)


def test_invalid_canonical_id_fixtures_fail_json_schema(
    repo_root: Path, invalid_fixture_dir: Path
) -> None:
    schema = repo_root / "schemas/checkpoint.schema.json"
    for path in invalid_fixture_dir.glob("checkpoint.invalid-id-*.json"):
        with pytest.raises(JsonSchemaValidationError):
            schema_validate(path, schema)


def test_schema_generation_idempotent(repo_root: Path) -> None:
    result = subprocess.run(
        [sys.executable, "scripts/generate_schemas.py", "--check"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
