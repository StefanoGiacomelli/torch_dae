from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.check_worktree_patch import validate_worktree_patch

ROOT = Path(__file__).resolve().parents[1]
EXEMPT_PATHS = (
    "src/torch_dae/models/panns/_vendor/LICENSE.MIT",
    "src/torch_dae/models/panns/_vendor/models.py",
    "src/torch_dae/models/panns/_vendor/pytorch_utils.py",
    "onboarding_reports/panns-audioset-three-tuple/integrate/source-reduction.diff",
)


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def git_repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    assert _git(root, "init", "-q").returncode == 0
    assert _git(root, "config", "user.name", "Validator Test").returncode == 0
    assert _git(root, "config", "user.email", "validator@example.invalid").returncode == 0
    (root / ".gitignore").write_text(".torch-dae/\n")
    (root / "README.md").write_text("baseline\n")
    assert _git(root, "add", "-A").returncode == 0
    assert _git(root, "commit", "-qm", "baseline").returncode == 0
    return root


def test_standard_diff_check_misses_untracked_whitespace(git_repository: Path) -> None:
    (git_repository / "untracked.py").write_text("value = 1  \n")

    result = _git(git_repository, "diff", "--check")

    assert result.returncode == 0
    assert result.stdout == ""


def test_staged_equivalent_validator_detects_untracked_whitespace(
    git_repository: Path,
) -> None:
    (git_repository / "untracked.py").write_text("value = 1  \n")

    result = validate_worktree_patch(git_repository)

    assert result["valid"] is False
    assert result["classification"] == "whitespace_error"
    assert "untracked.py:1: trailing whitespace" in result["diagnostics"]


def test_validator_does_not_change_real_index(git_repository: Path) -> None:
    staged = git_repository / "staged.txt"
    staged.write_text("staged content\n")
    assert _git(git_repository, "add", "staged.txt").returncode == 0
    before = (git_repository / ".git/index").read_bytes()
    (git_repository / "working.txt").write_text("working content\n")

    result = validate_worktree_patch(git_repository)

    assert result["valid"] is True
    assert result["real_index_unchanged"] is True
    assert (git_repository / ".git/index").read_bytes() == before
    assert _git(git_repository, "diff", "--cached", "--name-only").stdout == "staged.txt\n"


def test_validator_includes_tracked_modifications(git_repository: Path) -> None:
    (git_repository / "README.md").write_text("changed\n")

    result = validate_worktree_patch(git_repository)

    assert result["valid"] is True
    assert "README.md" in result["staged_equivalent_paths"]


def test_validator_includes_untracked_non_ignored_files(git_repository: Path) -> None:
    (git_repository / "new.txt").write_text("new\n")

    result = validate_worktree_patch(git_repository)

    assert result["valid"] is True
    assert "new.txt" in result["staged_equivalent_paths"]


def test_validator_excludes_ignored_runtime_files(git_repository: Path) -> None:
    ignored = git_repository / ".torch-dae/reports/ignored.txt"
    ignored.parent.mkdir(parents=True)
    ignored.write_text("ignored trailing whitespace  \n")

    result = validate_worktree_patch(git_repository)

    assert result["valid"] is True
    assert ".torch-dae/reports/ignored.txt" not in result["staged_equivalent_paths"]


def test_normal_source_trailing_whitespace_fails(git_repository: Path) -> None:
    source = git_repository / "src/package/model.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1  \n")

    result = validate_worktree_patch(git_repository)

    assert result["valid"] is False
    assert "src/package/model.py:1: trailing whitespace" in result["diagnostics"]


@pytest.mark.parametrize("relative", EXEMPT_PATHS)
def test_each_exact_panns_path_is_exempt(git_repository: Path, relative: str) -> None:
    attributes = "\n".join(f"{path} -whitespace" for path in EXEMPT_PATHS) + "\n"
    (git_repository / ".gitattributes").write_text(attributes)
    target = git_repository / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("preserved trailing whitespace  \n")

    result = validate_worktree_patch(git_repository)

    assert result["valid"] is True
    assert relative in result["staged_equivalent_paths"]


def test_unrelated_vendored_source_is_not_exempt(git_repository: Path) -> None:
    attributes = "\n".join(f"{path} -whitespace" for path in EXEMPT_PATHS) + "\n"
    (git_repository / ".gitattributes").write_text(attributes)
    target = git_repository / "src/torch_dae/models/other/_vendor/models.py"
    target.parent.mkdir(parents=True)
    target.write_text("not preserved  \n")

    result = validate_worktree_patch(git_repository)

    assert result["valid"] is False
    assert "src/torch_dae/models/other/_vendor/models.py" in result["diagnostics"]


@pytest.mark.parametrize("invalid", [False, True])
def test_temporary_index_is_removed_after_result(
    git_repository: Path,
    invalid: bool,
) -> None:
    suffix = "  \n" if invalid else "\n"
    (git_repository / "candidate.py").write_text(f"value = 1{suffix}")

    result = validate_worktree_patch(git_repository)

    assert result["valid"] is not invalid
    assert result["temporary_index_removed"] is True
    assert not Path(result["temporary_index_path"]).exists()


def test_validator_represents_deletions_and_detected_renames(git_repository: Path) -> None:
    deleted = git_repository / "delete-me.txt"
    renamed = git_repository / "rename-me.txt"
    deleted.write_text("delete\n")
    renamed.write_text("rename\n")
    assert _git(git_repository, "add", "-A").returncode == 0
    assert _git(git_repository, "commit", "-qm", "add rename fixtures").returncode == 0
    deleted.unlink()
    renamed.rename(git_repository / "renamed.txt")

    result = validate_worktree_patch(git_repository)

    assert result["valid"] is True
    assert "delete-me.txt" in result["staged_equivalent_paths"]
    assert "renamed.txt" in result["staged_equivalent_paths"]
    assert any(line.startswith("R") for line in result["staged_equivalent_name_status"])


def test_complete_panns_worktree_passes_staged_equivalent_validation() -> None:
    result = validate_worktree_patch(ROOT)

    assert result["valid"] is True, result["diagnostics"]
    assert result["real_index_unchanged"] is True
    assert result["temporary_index_removed"] is True
