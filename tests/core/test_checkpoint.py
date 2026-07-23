from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from torch_dae.core.checkpoint import (
    CheckpointManager,
    CheckpointSourceType,
    CheckpointSpec,
    checkpoint_cache_path,
)
from torch_dae.core.errors import NotImplementedInPhaseError


def test_checkpoint_source_variants(valid_fixture_dir: Path) -> None:
    expected = {item.value for item in CheckpointSourceType}
    observed = set()
    for path in valid_fixture_dir.glob("checkpoint.*.json"):
        spec = CheckpointSpec.model_validate_json(path.read_text())
        observed.add(spec.source_type.value)
    assert observed == expected


def test_malformed_sha_rejected(invalid_fixture_dir: Path) -> None:
    with pytest.raises(ValidationError):
        CheckpointSpec.model_validate_json(
            (invalid_fixture_dir / "checkpoint.malformed-sha.json").read_text()
        )


def test_source_missing_required_fields_rejected(invalid_fixture_dir: Path) -> None:
    with pytest.raises(ValidationError, match="https checkpoints require url"):
        CheckpointSpec.model_validate_json(
            (invalid_fixture_dir / "checkpoint.missing-source-field.json").read_text()
        )


def test_checkpoint_cache_path() -> None:
    sha = "a" * 64
    assert (
        checkpoint_cache_path(Path(".torch-dae"), "card", sha)
        == Path(".torch-dae/checkpoints/card/" + sha).resolve()
    )


@pytest.mark.parametrize("checkpoint_id", ["../escape", "a/b", "a\\b", "has space"])
def test_checkpoint_cache_path_rejects_escaping_ids(checkpoint_id: str) -> None:
    with pytest.raises(ValueError):
        checkpoint_cache_path(Path(".torch-dae"), checkpoint_id, "a" * 64)


def test_checkpoint_schema_version_required(valid_fixture_dir: Path) -> None:
    data = json.loads((valid_fixture_dir / "checkpoint.https.json").read_text())
    del data["schema_version"]
    with pytest.raises(ValidationError):
        CheckpointSpec.model_validate(data)


@pytest.mark.parametrize(
    "name",
    [
        "checkpoint.https-contradictory-fields.json",
        "checkpoint.github_release-contradictory-fields.json",
        "checkpoint.huggingface-contradictory-fields.json",
        "checkpoint.package_bundle-contradictory-fields.json",
        "checkpoint.local_path-contradictory-fields.json",
    ],
)
def test_checkpoint_source_exclusivity(invalid_fixture_dir: Path, name: str) -> None:
    with pytest.raises(ValidationError):
        CheckpointSpec.model_validate_json((invalid_fixture_dir / name).read_text())


def test_checkpoint_manager_phase00(repo_root: Path) -> None:
    manager = CheckpointManager(repo_root)
    assert manager.info("card")["runtime_root"].endswith(".torch-dae/checkpoints")
    with pytest.raises(NotImplementedInPhaseError):
        manager.ensure("card")
