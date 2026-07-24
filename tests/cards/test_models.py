from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from torch_dae.cards.models import (
    EvidenceStatus,
    IssueStatus,
    ModelCard,
    ModelCardLifecycle,
    ProfilingStatus,
)


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def test_required_enums() -> None:
    assert {item.value for item in ModelCardLifecycle} == {
        "draft",
        "analyzed",
        "environment_resolved",
        "checkpoint_verified",
        "runtime_verified",
        "profiled",
    }
    assert "officially_reported" in {item.value for item in EvidenceStatus}
    assert {item.value for item in IssueStatus} == {
        "open",
        "resolved",
        "accepted",
        "not_applicable",
    }
    assert {item.value for item in ProfilingStatus} == {"not_profiled", "profiled"}


def test_valid_model_card_fixture(valid_fixture_dir: Path) -> None:
    card = ModelCard.model_validate(load(valid_fixture_dir / "model-card.analyzed.json"))
    assert card.card_id == "synthetic-family-variant-checkpoint"
    assert card.input.shape == "B,C,T"
    assert card.input.valid_lengths_shape == "B"
    assert card.embeddings.default_embedding_id == "synthetic.global"


def test_runtime_verified_requires_report(valid_fixture_dir: Path) -> None:
    data = load(valid_fixture_dir / "model-card.analyzed.json")
    data["card_status"] = "runtime_verified"
    data["usage"]["recommended_environment"]["verified"] = True
    data["checkpoint"]["observed_sha256"] = "9" * 64
    data["checkpoint"]["expected_sha256"] = "9" * 64
    with pytest.raises(ValidationError, match="verification_report"):
        ModelCard.model_validate(data)


def test_default_embedding_must_exist(valid_fixture_dir: Path) -> None:
    data = load(valid_fixture_dir / "model-card.analyzed.json")
    embeddings = data["embeddings"]
    assert isinstance(embeddings, dict)
    embeddings["default_embedding_id"] = "missing"
    with pytest.raises(ValidationError, match="default_embedding_id"):
        ModelCard.model_validate(data)


def test_duplicate_default_embeddings_fail(valid_fixture_dir: Path) -> None:
    data = load(valid_fixture_dir / "model-card.analyzed.json")
    embeddings = data["embeddings"]
    assert isinstance(embeddings, dict)
    items = embeddings["items"]
    assert isinstance(items, list)
    items[1]["default"] = True
    with pytest.raises(ValidationError, match="exactly one"):
        ModelCard.model_validate(data)


def test_profiled_card_requires_profiled_sections(valid_fixture_dir: Path) -> None:
    data = load(valid_fixture_dir / "model-card.runtime.json")
    data["card_status"] = "profiled"
    with pytest.raises(ValidationError, match="profiling"):
        ModelCard.model_validate(data)


def test_empty_strings_are_rejected(valid_fixture_dir: Path) -> None:
    data = load(valid_fixture_dir / "model-card.analyzed.json")
    data["card_id"] = ""
    with pytest.raises(ValidationError, match="empty strings"):
        ModelCard.model_validate(data)
