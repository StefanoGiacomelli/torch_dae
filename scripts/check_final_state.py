#!/usr/bin/env python3
"""Record and recheck the immutable final repository state for a validation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA-256 for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repository_root: Path, *arguments: str) -> bytes:
    """Run one read-only Git command and return raw stdout."""

    return subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=True,
        capture_output=True,
    ).stdout


def staged_equivalent_inventory(repository_root: Path) -> tuple[str, ...]:
    """Return every tracked change and untracked non-ignored file without staging."""

    tracked = git_output(repository_root, "diff", "--name-only", "-z").split(b"\0")
    untracked = git_output(
        repository_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    ).split(b"\0")
    return tuple(sorted({item.decode("utf-8") for item in (*tracked, *untracked) if item}))


def capture_state(repository_root: Path, workflow_id: str, phase: str) -> dict[str, Any]:
    """Capture the exact final state relevant to immutable validation and packaging."""

    inventory = staged_equivalent_inventory(repository_root)
    handoff = repository_root / "onboarding_reports" / workflow_id / phase / "handoff.json"
    if not handoff.is_file():
        raise FileNotFoundError(f"current handoff is missing: {workflow_id}/{phase}")
    file_hashes = {relative: sha256_file(repository_root / relative) for relative in inventory}
    cached_diff = git_output(repository_root, "diff", "--cached", "--binary")
    return {
        "schema_version": "1.0.0",
        "workflow_id": workflow_id,
        "phase": phase,
        "branch": git_output(repository_root, "branch", "--show-current").decode().strip(),
        "head": git_output(repository_root, "rev-parse", "HEAD").decode().strip(),
        "real_index_empty": not cached_diff,
        "real_index_diff_sha256": hashlib.sha256(cached_diff).hexdigest(),
        "staged_equivalent_count": len(inventory),
        "staged_equivalent_inventory": list(inventory),
        "staged_equivalent_file_sha256": file_hashes,
        "current_handoff_path": handoff.relative_to(repository_root).as_posix(),
        "current_handoff_sha256": sha256_file(handoff),
    }


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write deterministic JSON atomically outside the immutable repository change set."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument("--phase", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record", type=Path)
    mode.add_argument("--check", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    repository_root = args.repository_root.resolve()
    current = capture_state(repository_root, args.workflow_id, args.phase)
    if args.record is not None:
        result = {**current, "status": "recorded", "final_state_equal": True}
        write_json_atomic(args.record, result)
    else:
        recorded = json.loads(args.check.read_text(encoding="utf-8"))
        comparable_keys = tuple(current)
        differences = [key for key in comparable_keys if recorded.get(key) != current[key]]
        result = {
            **current,
            "status": "passed" if not differences else "failed",
            "final_state_equal": not differences,
            "differences": differences,
        }
        write_json_atomic(args.check, result)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    if not result["final_state_equal"] or not result["real_index_empty"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
