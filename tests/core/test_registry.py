from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from torch_dae.core.errors import DuplicateModelCardError
from torch_dae.core.registry import ModelCardRegistry


def test_registry_recursive_discovery(
    tmp_path: Path, valid_fixture_dir: Path, repo_root: Path
) -> None:
    root = tmp_path
    (root / "schemas").mkdir()
    shutil.copy(
        repo_root / "schemas/model-card.schema.json", root / "schemas/model-card.schema.json"
    )
    card_dir = root / "model_cards/synthetic/nested"
    card_dir.mkdir(parents=True)
    shutil.copy(valid_fixture_dir / "model-card.analyzed.json", card_dir / "card.json")
    cards = ModelCardRegistry(root).list_cards()
    assert [card.card_id for card in cards] == ["synthetic-family-variant-checkpoint"]


def test_registry_duplicate_detection(
    tmp_path: Path, valid_fixture_dir: Path, repo_root: Path
) -> None:
    root = tmp_path
    (root / "schemas").mkdir()
    shutil.copy(
        repo_root / "schemas/model-card.schema.json", root / "schemas/model-card.schema.json"
    )
    for index in (1, 2):
        card_dir = root / f"model_cards/{index}"
        card_dir.mkdir(parents=True)
        shutil.copy(valid_fixture_dir / "model-card.analyzed.json", card_dir / "card.json")
    with pytest.raises(DuplicateModelCardError):
        ModelCardRegistry(root).list_cards()


def test_registry_get_card_does_not_import_wrapper(
    tmp_path: Path, valid_fixture_dir: Path, repo_root: Path
) -> None:
    root = tmp_path
    (root / "schemas").mkdir()
    shutil.copy(
        repo_root / "schemas/model-card.schema.json", root / "schemas/model-card.schema.json"
    )
    card_dir = root / "model_cards/synthetic"
    card_dir.mkdir(parents=True)
    shutil.copy(valid_fixture_dir / "model-card.analyzed.json", card_dir / "card.json")
    card = ModelCardRegistry(root).get_card("synthetic-family-variant-checkpoint")
    assert card.identity.wrapper_entry_point == "torch_dae.synthetic:AudioModel"


def test_registry_returns_exact_path_when_filename_differs(
    tmp_path: Path, valid_fixture_dir: Path, repo_root: Path
) -> None:
    root = tmp_path
    (root / "schemas").mkdir()
    shutil.copy(
        repo_root / "schemas/model-card.schema.json", root / "schemas/model-card.schema.json"
    )
    card_dir = root / "model_cards/synthetic"
    card_dir.mkdir(parents=True)
    expected = card_dir / "not-the-card-id.json"
    shutil.copy(valid_fixture_dir / "model-card.analyzed.json", expected)
    assert ModelCardRegistry(root).get_card_path("synthetic-family-variant-checkpoint") == expected
