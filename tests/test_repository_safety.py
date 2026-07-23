from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


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
