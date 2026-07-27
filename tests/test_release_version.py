from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_check(
    repo_root: Path,
    *arguments: str,
    pyproject: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(repo_root / "scripts/check_release_version.py"), *arguments]
    if pyproject is not None:
        command.extend(["--pyproject", str(pyproject)])
    return subprocess.run(command, check=False, capture_output=True, text=True)


def test_matching_release_tag(repo_root: Path) -> None:
    result = run_check(repo_root, "v0.1.0")
    assert result.returncode == 0
    assert result.stdout == "release version validated: v0.1.0\n"


def test_current_project_version(repo_root: Path) -> None:
    result = run_check(repo_root, "--current")
    assert result.returncode == 0
    assert result.stdout == "project version validated: 0.1.0\n"


def test_missing_v_is_rejected(repo_root: Path) -> None:
    result = run_check(repo_root, "0.1.0")
    assert result.returncode != 0
    assert "exact format" in result.stderr
    assert "Traceback" not in result.stderr


def test_malformed_version_is_rejected(repo_root: Path) -> None:
    result = run_check(repo_root, "v0.1")
    assert result.returncode != 0
    assert "exact format" in result.stderr


def test_mismatched_version_is_rejected(repo_root: Path) -> None:
    result = run_check(repo_root, "v0.2.0")
    assert result.returncode != 0
    assert "does not match" in result.stderr


def test_empty_tag_is_rejected(repo_root: Path) -> None:
    result = run_check(repo_root, "")
    assert result.returncode != 0
    assert "exact format" in result.stderr


def test_malformed_pyproject_is_concise(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[project\n")
    result = run_check(repo_root, "v0.1.0", pyproject=pyproject)
    assert result.returncode != 0
    assert "cannot read valid pyproject.toml" in result.stderr
    assert "Traceback" not in result.stderr


def test_missing_project_version_is_concise(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[project]\nname = 'missing-version'\n")
    result = run_check(repo_root, "v0.1.0", pyproject=pyproject)
    assert result.returncode != 0
    assert "no valid project.version" in result.stderr
    assert "Traceback" not in result.stderr
