from __future__ import annotations

import json
import shutil
from pathlib import Path

from scripts.validate_repository import validate_canonical_technical_cards


def _copy_canonical_evidence(repo_root: Path, tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    shutil.copytree(repo_root / "model_cards", root / "model_cards")
    shutil.copytree(repo_root / "technical_cards", root / "technical_cards")
    return root


def test_current_canonical_technical_cards_pass_repository_validation(repo_root: Path) -> None:
    failures: list[str] = []
    validate_canonical_technical_cards(repo_root, failures)
    assert failures == []


def test_missing_canonical_raw_asset_fails(repo_root: Path, tmp_path: Path) -> None:
    root = _copy_canonical_evidence(repo_root, tmp_path)
    raw = next((root / "technical_cards").glob("*/*.npz"))
    raw.unlink()

    failures: list[str] = []
    validate_canonical_technical_cards(root, failures)

    assert any("raw asset is missing" in item or "raw asset missing" in item for item in failures)


def test_tampered_canonical_raw_asset_fails(repo_root: Path, tmp_path: Path) -> None:
    root = _copy_canonical_evidence(repo_root, tmp_path)
    raw = next((root / "technical_cards").glob("*/*.npz"))
    raw.write_bytes(raw.read_bytes() + b"tampered")

    failures: list[str] = []
    validate_canonical_technical_cards(root, failures)

    assert any("SHA-256" in item or "hash" in item for item in failures)


def test_orphan_canonical_raw_asset_fails(repo_root: Path, tmp_path: Path) -> None:
    root = _copy_canonical_evidence(repo_root, tmp_path)
    model_dir = next(path for path in (root / "technical_cards").iterdir() if path.is_dir())
    orphan = model_dir / "tc-orphan.npz"
    orphan.write_bytes(b"not-a-card")

    failures: list[str] = []
    validate_canonical_technical_cards(root, failures)

    assert any("orphan canonical Technical Card raw asset" in item for item in failures)


def test_model_directory_must_match_card_model_id(repo_root: Path, tmp_path: Path) -> None:
    root = _copy_canonical_evidence(repo_root, tmp_path)
    card_path = next((root / "technical_cards").glob("*/*.json"))
    card = json.loads(card_path.read_text(encoding="utf-8"))
    wrong_dir = root / "technical_cards" / "wrong-model-id"
    wrong_dir.mkdir()
    relocated_json = wrong_dir / card_path.name
    relocated_npz = wrong_dir / card_path.with_suffix(".npz").name
    shutil.copy2(card_path, relocated_json)
    shutil.copy2(card_path.with_suffix(".npz"), relocated_npz)
    card_path.unlink()
    card_path.with_suffix(".npz").unlink()

    failures: list[str] = []
    validate_canonical_technical_cards(root, failures)

    assert card["model"]["model_card_id"] != "wrong-model-id"
    assert any("model directory disagrees" in item for item in failures)
