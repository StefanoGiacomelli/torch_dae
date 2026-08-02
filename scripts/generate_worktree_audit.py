#!/usr/bin/env python3
"""Generate a deterministic, portable, read-only working-tree audit archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

from scripts.check_final_state import capture_state, sha256_file


def canonical_json_bytes(data: object) -> bytes:
    """Return deterministic human-readable JSON bytes."""

    return (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8")


def normalized_tar_info(name: str, size: int) -> tarfile.TarInfo:
    """Create normalized metadata for one regular archive member."""

    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def add_member(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    """Add one normalized regular file."""

    archive.addfile(normalized_tar_info(name, len(data)), io.BytesIO(data))


def runtime_log_members(
    repository_root: Path,
    workflow_id: str,
    fingerprints: dict[str, str],
) -> dict[str, bytes]:
    """Collect only curated command logs, cleanup receipts, and no managed payloads."""

    members: dict[str, bytes] = {}
    runtime_root = repository_root / ".torch-dae"
    for environment_id, fingerprint in sorted(fingerprints.items()):
        report_root = runtime_root / "reports" / "environments" / environment_id / fingerprint
        if not report_root.is_dir():
            raise FileNotFoundError(f"current environment report root is missing: {environment_id}")
        for path in sorted(report_root.rglob("*")):
            if path.is_file() and not path.is_symlink():
                relative = path.relative_to(runtime_root).as_posix()
                members[f"runtime-command-logs/{relative}"] = path.read_bytes()
    cleanup_root = runtime_root / "reports" / "onboarding" / workflow_id / "cleanup"
    if cleanup_root.is_dir():
        for path in sorted(cleanup_root.glob("*.json")):
            members[f"cleanup-receipts/{path.name}"] = path.read_bytes()
    return members


def generate_audit(
    repository_root: Path,
    output_dir: Path,
    artifact_stem: str,
    validation_result: Path,
) -> dict[str, Any]:
    """Build the archive only when the live repository exactly matches final validation."""

    recorded = json.loads(validation_result.read_text(encoding="utf-8"))
    current = capture_state(repository_root, recorded["workflow_id"], recorded["phase"])
    differences = [key for key, value in current.items() if recorded.get(key) != value]
    if differences or not recorded.get("final_state_equal") or not current["real_index_empty"]:
        raise RuntimeError(f"final repository state drifted before audit generation: {differences}")
    fingerprints: dict[str, str] = {}
    results_root = (
        repository_root
        / "onboarding_reports"
        / recorded["workflow_id"]
        / recorded["phase"]
        / "environment-results"
    )
    for path in sorted(results_root.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        fingerprints[result["environment_id"]] = result["environment_fingerprint"]

    members: dict[str, bytes] = {
        "validation/final-state.json": validation_result.read_bytes(),
        "repository/state.json": canonical_json_bytes(current),
    }
    for relative in current["staged_equivalent_inventory"]:
        members[f"working-tree/{relative}"] = (repository_root / relative).read_bytes()
    members.update(runtime_log_members(repository_root, recorded["workflow_id"], fingerprints))
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(members.items())}
    manifest = "".join(f"{digest}  {name}\n" for name, digest in hashes.items()).encode("utf-8")

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"{artifact_stem}.tar.gz"
    with archive_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.GNU_FORMAT) as archive:
                for name, data in sorted(members.items()):
                    add_member(archive, name, data)
                add_member(archive, "MANIFEST.sha256", manifest)

    archive_sha256 = sha256_file(archive_path)
    archive_sidecar = output_dir / f"{artifact_stem}.tar.gz.sha256"
    archive_sidecar.write_text(f"{archive_sha256}  {archive_path.name}\n", encoding="utf-8")
    result = {
        "schema_version": "1.0.0",
        "status": "passed",
        "archive": str(archive_path),
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_path.stat().st_size,
        "member_count": len(members) + 1,
        "manifest_member_count": len(hashes),
        "manifest_covers_every_member_except_itself": len(hashes) + 1 == len(members) + 1,
        "normalized_gzip_mtime": int.from_bytes(archive_path.read_bytes()[4:8], "little") == 0,
        "normalized_tar_metadata": True,
        "staged_equivalent_count": current["staged_equivalent_count"],
        "final_state_equal": True,
        "current_handoff_sha256": current["current_handoff_sha256"],
        "environment_fingerprints": fingerprints,
        "included_environments_or_caches": False,
        "checkpoint_payloads_included": False,
    }
    result_path = output_dir / f"{artifact_stem}.result.json"
    result_path.write_bytes(canonical_json_bytes(result))
    result_sha256 = sha256_file(result_path)
    result_sidecar = output_dir / f"{artifact_stem}.result.json.sha256"
    result_sidecar.write_text(f"{result_sha256}  {result_path.name}\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-stem", required=True)
    parser.add_argument("--validation-result", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = generate_audit(
        args.repository_root.resolve(),
        args.output_dir.resolve(),
        args.artifact_stem,
        args.validation_result.resolve(),
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
