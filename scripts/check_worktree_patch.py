"""Check the complete working-tree patch with a disposable Git index."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


class WorktreePatchValidationError(RuntimeError):
    """Raised when Git cannot construct the staged-equivalent patch."""


def _run_git(
    repository_root: Path,
    arguments: list[str],
    *,
    index_path: Path | None = None,
) -> subprocess.CompletedProcess[bytes]:
    environment = os.environ.copy()
    if index_path is not None:
        environment["GIT_INDEX_FILE"] = str(index_path)
    return subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
    )


def _index_path(repository_root: Path) -> Path:
    result = _run_git(repository_root, ["rev-parse", "--git-path", "index"])
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise WorktreePatchValidationError(f"cannot locate the real Git index: {detail}")
    path = Path(result.stdout.decode().strip())
    return path if path.is_absolute() else repository_root / path


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_git_success(
    repository_root: Path,
    arguments: list[str],
    *,
    index_path: Path,
) -> subprocess.CompletedProcess[bytes]:
    result = _run_git(repository_root, arguments, index_path=index_path)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode("utf-8", errors="replace").strip()
        raise WorktreePatchValidationError(
            f"git {' '.join(arguments)} failed with exit code {result.returncode}: {detail}"
        )
    return result


def validate_worktree_patch(
    repository_root: Path,
    *,
    temporary_directory: Path | None = None,
) -> dict[str, Any]:
    """Validate exactly what ``git add -A`` would stage without changing the real index."""

    root = repository_root.resolve()
    real_index = _index_path(root)
    real_index_before = _file_sha256(real_index)
    descriptor, raw_temp_path = tempfile.mkstemp(
        prefix="torch-dae-index-",
        dir=temporary_directory,
    )
    os.close(descriptor)
    temporary_index = Path(raw_temp_path)
    temporary_index.unlink()
    payload: dict[str, Any]
    try:
        _require_git_success(root, ["read-tree", "HEAD"], index_path=temporary_index)
        _require_git_success(root, ["add", "-A"], index_path=temporary_index)
        names = _require_git_success(
            root,
            ["diff", "--cached", "--name-only", "-z"],
            index_path=temporary_index,
        )
        staged_paths = sorted(
            item.decode("utf-8", errors="surrogateescape")
            for item in names.stdout.split(b"\0")
            if item
        )
        status = _require_git_success(
            root,
            ["diff", "--cached", "--name-status", "-M"],
            index_path=temporary_index,
        )
        whitespace = _run_git(
            root,
            ["diff", "--cached", "--check"],
            index_path=temporary_index,
        )
        diagnostics = (whitespace.stdout + whitespace.stderr).decode("utf-8", errors="replace")
        if whitespace.returncode not in {0, 1, 2}:
            raise WorktreePatchValidationError(
                "git diff --cached --check failed with exit code "
                f"{whitespace.returncode}: {diagnostics.strip()}"
            )
        valid = whitespace.returncode == 0
        payload = {
            "classification": "passed" if valid else "whitespace_error",
            "diagnostics": diagnostics,
            "real_index_path": str(real_index),
            "repository_root": str(root),
            "staged_equivalent_file_count": len(staged_paths),
            "staged_equivalent_name_status": status.stdout.decode(
                "utf-8", errors="replace"
            ).splitlines(),
            "staged_equivalent_paths": staged_paths,
            "temporary_index_path": str(temporary_index),
            "valid": valid,
        }
    except WorktreePatchValidationError as exc:
        payload = {
            "classification": "git_error",
            "diagnostics": str(exc),
            "real_index_path": str(real_index),
            "repository_root": str(root),
            "staged_equivalent_file_count": 0,
            "staged_equivalent_name_status": [],
            "staged_equivalent_paths": [],
            "temporary_index_path": str(temporary_index),
            "valid": False,
        }
    finally:
        temporary_index.unlink(missing_ok=True)
    real_index_after = _file_sha256(real_index)
    payload.update(
        {
            "real_index_after_sha256": real_index_after,
            "real_index_before_sha256": real_index_before,
            "real_index_unchanged": real_index_before == real_index_after,
            "temporary_index_removed": not temporary_index.exists(),
        }
    )
    if not payload["real_index_unchanged"]:
        payload["classification"] = "real_index_changed"
        payload["valid"] = False
    if not payload["temporary_index_removed"]:
        payload["classification"] = "temporary_index_cleanup_error"
        payload["valid"] = False
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        payload = validate_worktree_patch(args.repository)
    except (OSError, WorktreePatchValidationError) as exc:
        payload = {
            "classification": "git_error",
            "diagnostics": str(exc),
            "repository_root": str(args.repository.resolve()),
            "valid": False,
        }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif payload["valid"]:
        print(
            "staged-equivalent whitespace validation passed "
            f"({payload['staged_equivalent_file_count']} files)"
        )
    else:
        print(f"{payload['classification']}: {payload['diagnostics']}", file=sys.stderr)
    if payload["valid"]:
        return 0
    return 1 if payload["classification"] == "whitespace_error" else 2


if __name__ == "__main__":
    raise SystemExit(main())
