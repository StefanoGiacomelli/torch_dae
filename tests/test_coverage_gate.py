from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_gate(
    repo_root: Path, path: Path, line: float = 85, branch: float = 70
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/check_coverage.py"),
            str(path),
            "--min-line",
            str(line),
            "--min-branch",
            str(branch),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def write_coverage(path: Path, *, lines: tuple[int, int], branches: tuple[int, int]) -> None:
    path.write_text(
        json.dumps(
            {
                "totals": {
                    "covered_lines": lines[0],
                    "num_statements": lines[1],
                    "covered_branches": branches[0],
                    "num_branches": branches[1],
                }
            }
        )
    )


def test_both_thresholds_pass(repo_root: Path, tmp_path: Path) -> None:
    path = tmp_path / "coverage.json"
    write_coverage(path, lines=(90, 100), branches=(75, 100))
    result = run_gate(repo_root, path)
    assert result.returncode == 0
    assert "Line coverage: 90.00% (required 85.00%)" in result.stdout
    assert "Branch coverage: 75.00% (required 70.00%)" in result.stdout


def test_line_threshold_fails(repo_root: Path, tmp_path: Path) -> None:
    path = tmp_path / "coverage.json"
    write_coverage(path, lines=(84, 100), branches=(80, 100))
    assert run_gate(repo_root, path).returncode == 1


def test_branch_threshold_fails(repo_root: Path, tmp_path: Path) -> None:
    path = tmp_path / "coverage.json"
    write_coverage(path, lines=(90, 100), branches=(69, 100))
    assert run_gate(repo_root, path).returncode == 1


def test_malformed_json_is_concise(repo_root: Path, tmp_path: Path) -> None:
    path = tmp_path / "coverage.json"
    path.write_text("{bad")
    result = run_gate(repo_root, path)
    assert result.returncode == 2
    assert "ERROR:" in result.stderr
    assert "Traceback" not in result.stderr


def test_missing_totals_is_concise(repo_root: Path, tmp_path: Path) -> None:
    path = tmp_path / "coverage.json"
    path.write_text("{}")
    result = run_gate(repo_root, path)
    assert result.returncode == 2
    assert "missing totals" in result.stderr
    assert "Traceback" not in result.stderr


def test_zero_branch_denominator_is_full_coverage(repo_root: Path, tmp_path: Path) -> None:
    path = tmp_path / "coverage.json"
    write_coverage(path, lines=(90, 100), branches=(0, 0))
    result = run_gate(repo_root, path)
    assert result.returncode == 0
    assert "Branch coverage: 100.00%" in result.stdout
