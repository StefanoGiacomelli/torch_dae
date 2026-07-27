from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

from scripts.validate_repository import validate_integration_artifacts


def test_runtime_state_is_ignored(repo_root: Path) -> None:
    result = subprocess.run(
        ["git", "check-ignore", "-q", ".torch-dae/checkpoints/card/hash/file.pt"],
        cwd=repo_root,
        check=False,
    )
    assert result.returncode == 0


def test_no_runtime_files_staged(repo_root: Path) -> None:
    result = subprocess.run(
        ["git", "status", "--short", ".torch-dae"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout == ""


def test_no_legacy_backbone_json(repo_root: Path) -> None:
    assert not list(repo_root.glob("**/*backbone*.json"))


def test_root_environment_has_no_torch() -> None:
    assert importlib.util.find_spec("torch") is None


def prepare_integration_root(repo_root: Path, root: Path) -> None:
    shutil.copytree(repo_root / "schemas", root / "schemas")
    (root / "model_cards").mkdir()
    (root / "environments").mkdir()
    (root / "verification_reports").mkdir()
    wrapper = root / "src/torch_dae/synthetic.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("class AudioModel:\n    pass\n")


def add_valid_synthetic_integration(repo_root: Path, root: Path) -> None:
    card = json.loads((repo_root / "tests/fixtures/valid/model-card.analyzed.json").read_text())
    card["card_status"] = "environment_resolved"
    card["usage"]["recommended_environment"]["verified"] = True
    card_id = card["card_id"]
    (root / "model_cards" / f"{card_id}.json").write_text(json.dumps(card))

    environment_dir = root / "environments" / card_id
    environment_dir.mkdir()
    shutil.copy(
        repo_root / "tests/fixtures/valid/environment.synthetic.json",
        environment_dir / "environment.json",
    )
    shutil.copy(
        repo_root / "tests/fixtures/valid/environment-sources.synthetic.json",
        environment_dir / "sources.json",
    )
    (environment_dir / "pyproject.toml").write_text(
        "[project]\nname = 'synthetic-model-environment'\nversion = '0.1.0'\n"
    )
    (environment_dir / "uv.lock").write_text("version = 1\n")
    (environment_dir / "verify_environment.py").write_text("raise SystemExit(0)\n")


def test_current_empty_integration_layout_passes(repo_root: Path) -> None:
    failures: list[str] = []
    validate_integration_artifacts(repo_root, failures)
    assert failures == []


def test_structurally_valid_synthetic_integration_passes(repo_root: Path, tmp_path: Path) -> None:
    prepare_integration_root(repo_root, tmp_path)
    add_valid_synthetic_integration(repo_root, tmp_path)
    failures: list[str] = []
    validate_integration_artifacts(tmp_path, failures)
    assert failures == []


def test_invalid_model_card_fails(repo_root: Path, tmp_path: Path) -> None:
    prepare_integration_root(repo_root, tmp_path)
    (tmp_path / "model_cards/invalid.json").write_text("{}")
    failures: list[str] = []
    validate_integration_artifacts(tmp_path, failures)
    assert any("invalid model card" in failure for failure in failures)


def test_missing_wrapper_symbol_fails(repo_root: Path, tmp_path: Path) -> None:
    prepare_integration_root(repo_root, tmp_path)
    card = repo_root / "tests/fixtures/valid/model-card.analyzed.json"
    shutil.copy(card, tmp_path / "model_cards/card.json")
    (tmp_path / "src/torch_dae/synthetic.py").write_text("class OtherModel:\n    pass\n")
    failures: list[str] = []
    validate_integration_artifacts(tmp_path, failures)
    assert any("wrapper symbol" in failure for failure in failures)


def test_checkpoint_binary_fails(repo_root: Path, tmp_path: Path) -> None:
    prepare_integration_root(repo_root, tmp_path)
    binary = tmp_path / "tests/fixture.ckpt"
    binary.parent.mkdir()
    binary.write_bytes(b"\x00\x01checkpoint")
    failures: list[str] = []
    validate_integration_artifacts(tmp_path, failures)
    assert any("binary is forbidden" in failure for failure in failures)


def test_runtime_verified_card_without_report_fails(repo_root: Path, tmp_path: Path) -> None:
    prepare_integration_root(repo_root, tmp_path)
    card = json.loads((repo_root / "tests/fixtures/valid/model-card.runtime.json").read_text())
    (tmp_path / "model_cards/runtime.json").write_text(json.dumps(card))
    failures: list[str] = []
    validate_integration_artifacts(tmp_path, failures)
    assert any("lacks its verification report" in failure for failure in failures)
