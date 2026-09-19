"""Assemble the deterministic, self-contained Profiling v1 implementation review package.

Produces a reproducible `.tar.gz` (normalized ownership/timestamps/modes) plus a `.sha256`
sidecar and a top-level `.result.json`, gathering: repository identity, the implementation patch,
gate/test outputs, the profiling protocol identity, a privacy-safe hardware/software summary, the
real PANNs profiling campaign results, every generated candidate Technical Card and `.npz`, and
their validation outputs. This is on-demand transport/audit tooling generated outside the
repository; it is not itself part of the committed tree.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def run(*cmd: str, check: bool = False) -> tuple[int, str, str]:
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {result.stderr}")
    return result.returncode, result.stdout, result.stderr


def _run_git_bytes(arguments: list[str], *, index_path: Path | None = None) -> bytes:
    environment = os.environ.copy()
    if index_path is not None:
        environment["GIT_INDEX_FILE"] = str(index_path)
    result = subprocess.run(
        ["git", *arguments], cwd=ROOT, env=environment, check=False, capture_output=True
    )
    if result.returncode not in (0, 1):  # `diff --check` uses 1 for "found issues"
        raise RuntimeError(
            f"git {' '.join(arguments)} failed: {result.stderr.decode(errors='replace')}"
        )
    return result.stdout


def complete_worktree_patch() -> tuple[bytes, list[str]]:
    """Build the full `git diff`-format patch of the working tree against HEAD, with every
    untracked (new, previously-uncommitted) file included as a `/dev/null -> <path>` addition.

    A plain `git diff` never includes untracked files, which is exactly what made the prior
    review package's `implementation.patch` silently omit the entire new Profiling v1
    implementation (every file under `src/torch_dae/profiling/`, `tests/profiling/`, etc. was
    untracked). This uses a disposable temporary index -- the same technique as
    `scripts/check_worktree_patch.py` -- so the real Git index is never touched.
    """

    descriptor, raw_temp_path = tempfile.mkstemp(prefix="torch-dae-review-index-")
    os.close(descriptor)
    temporary_index = Path(raw_temp_path)
    temporary_index.unlink()
    try:
        _run_git_bytes(["read-tree", "HEAD"], index_path=temporary_index)
        _run_git_bytes(["add", "-A"], index_path=temporary_index)
        patch = _run_git_bytes(["diff", "--cached", "--no-color"], index_path=temporary_index)
        names = _run_git_bytes(
            ["diff", "--cached", "--name-only", "-z"], index_path=temporary_index
        )
        staged_paths = sorted(
            item.decode("utf-8", errors="surrogateescape") for item in names.split(b"\0") if item
        )
    finally:
        temporary_index.unlink(missing_ok=True)
    return patch, staged_paths


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_tarinfo(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    if info.isfile():
        info.mode = 0o644
    elif info.isdir():
        info.mode = 0o755
    return info


def main() -> int:
    output_root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "torch-dae-review"
    staging = output_root / "staging"
    staging.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "repository_root": str(ROOT),
    }

    # --- Git identity and status -------------------------------------------------------------
    _, head, _ = run("git", "rev-parse", "HEAD")
    _, branch, _ = run("git", "rev-parse", "--abbrev-ref", "HEAD")
    _, initial_status, _ = run("git", "status", "--short")
    _, diff_stat, _ = run("git", "diff", "--stat")
    _, diff_name_status, _ = run("git", "diff", "--name-status")
    _, untracked, _ = run("git", "ls-files", "--others", "--exclude-standard")
    diff_check_returncode, diff_check_stdout, diff_check_stderr = run("git", "diff", "--check")

    # Complete working-tree patch (tracked diffs + every untracked file as a `/dev/null -> path`
    # addition) via a disposable temporary index -- never a plain `git diff`, which silently
    # omits untracked files. This is what actually makes new files inspectable from the package.
    complete_patch_bytes, staged_equivalent_paths = complete_worktree_patch()
    (staging / "implementation.patch").write_bytes(complete_patch_bytes)

    result["git"] = {
        "head": head.strip(),
        "branch": branch.strip(),
        "initial_status_short": initial_status,
        "diff_stat": diff_stat,
        "diff_name_status": diff_name_status,
        "untracked_files": untracked.splitlines(),
        "complete_worktree_patch_file_count": len(staged_equivalent_paths),
        "complete_worktree_patch_paths": staged_equivalent_paths,
        "diff_check": {
            "returncode": diff_check_returncode,
            "stdout": diff_check_stdout,
            "stderr": diff_check_stderr,
        },
    }
    (staging / "git-status-initial.txt").write_text(initial_status)
    (staging / "git-diff-stat.txt").write_text(diff_stat)
    (staging / "git-diff-name-status.txt").write_text(diff_name_status)
    (staging / "untracked-files.txt").write_text(untracked)
    (staging / "complete-worktree-paths.txt").write_text("\n".join(staged_equivalent_paths) + "\n")

    # --- Normalized full-tree source snapshot ---------------------------------------------------
    # A second, redundant representation of every changed/new file's content -- not just a patch
    # to apply, but the actual files a reviewer can open directly. Covers exactly the same path
    # set as `complete_worktree_patch()` above (tracked diffs + untracked additions).
    source_snapshot_dir = staging / "source"
    if source_snapshot_dir.exists():
        shutil.rmtree(source_snapshot_dir)
    for relative in staged_equivalent_paths:
        source_path = ROOT / relative
        if not source_path.is_file():
            continue  # deletions appear in the patch but have no content to snapshot
        destination = source_snapshot_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)

    # --- Gates ---------------------------------------------------------------------------------
    gates: dict[str, object] = {}

    global_quality_commands = {
        "ruff_format_global": ("ruff", "format", "--check"),
        "ruff_lint_global": ("ruff", "check"),
        "mypy_global": ("mypy", "src", "scripts"),
    }
    for gate_name, command in global_quality_commands.items():
        rc, out, err = run(
            "uv",
            "run",
            "--python",
            "3.11",
            "--all-groups",
            "--extra",
            "profiling",
            "--frozen",
            *command,
        )
        gate_result = {"returncode": rc, "stdout": out, "stderr": err}
        gates[gate_name] = gate_result
        (staging / f"{gate_name}.log").write_text(
            f"returncode: {rc}\n\nstdout:\n{out}\n\nstderr:\n{err}"
        )

    rc, out, err = run(
        "uv", "run", "--extra", "profiling", "python", "scripts/validate_repository.py"
    )
    gates["repository_validation"] = {"returncode": rc, "stdout": out, "stderr": err}

    rc, out, err = run(
        "uv",
        "run",
        "--extra",
        "docs",
        "--extra",
        "profiling",
        "sphinx-build",
        "-b",
        "html",
        "docs",
        "docs/_build/html-review",
        "-W",
    )
    gates["docs_build"] = {"returncode": rc, "stdout": out[-8000:], "stderr": err[-8000:]}

    rc, out, err = run(
        "uv",
        "run",
        "--extra",
        "profiling",
        "python",
        "scripts/generate_schemas.py",
        "--check",
    )
    gates["generated_schemas_current"] = {"returncode": rc, "stdout": out, "stderr": err}

    rc, out, err = run(
        "uv",
        "run",
        "--extra",
        "profiling",
        "python",
        "scripts/generate_profiling_schema.py",
        "--check",
    )
    gates["profiling_schema_current"] = {"returncode": rc, "stdout": out, "stderr": err}

    rc, out, err = run(
        "uv",
        "run",
        "--extra",
        "profiling",
        "python",
        "-m",
        "ruff",
        "check",
        "src/torch_dae/profiling",
        "src/torch_dae/profiling_worker.py",
        "src/torch_dae/profiling_executor.py",
        "src/torch_dae/cli/technical_cards.py",
        "src/torch_dae/cli/models.py",
        "tests/profiling",
    )
    gates["ruff_profiling"] = {"returncode": rc, "stdout": out, "stderr": err}

    started = time.time()
    rc, out, err = run("uv", "run", "--extra", "profiling", "pytest", "-q")
    gates["full_test_suite"] = {
        "returncode": rc,
        "stdout": out[-20000:],
        "stderr": err[-4000:],
        "duration_seconds": time.time() - started,
    }

    rc, out, err = run("uv", "run", "--extra", "profiling", "pytest", "tests/profiling", "-q")
    gates["profiling_test_suite"] = {"returncode": rc, "stdout": out, "stderr": err}

    rc, out, err = run(
        "uv", "run", "--extra", "profiling", "torch-dae", "model", "profile", "--help"
    )
    gates["cli_model_profile_help"] = {"returncode": rc, "stdout": out, "stderr": err}
    rc, out, err = run("uv", "run", "--extra", "profiling", "torch-dae", "technical-card", "--help")
    gates["cli_technical_card_help"] = {"returncode": rc, "stdout": out, "stderr": err}

    result["gates"] = gates
    (staging / "gates.json").write_text(json.dumps(gates, indent=2))

    # --- Hardware/software summary (privacy-safe) -----------------------------------------------
    rc, out, err = run(
        "uv",
        "run",
        "--extra",
        "profiling",
        "python",
        "-c",
        "import json; from torch_dae.profiling.fingerprint import gather_hardware_metadata, "
        "hardware_fingerprint; hw = gather_hardware_metadata(); "
        "print(json.dumps({'hardware': hw.model_dump(), "
        "'hardware_fingerprint': hardware_fingerprint(hw)}, indent=2))",
    )
    result["local_hardware_summary"] = json.loads(out) if rc == 0 else {"error": err}
    (staging / "hardware-summary.json").write_text(
        json.dumps(result["local_hardware_summary"], indent=2)
    )

    # --- Profiling campaign / candidate evidence -----------------------------------------------
    dogfood_dir = ROOT / ".torch-dae/reports/profiling-dogfood"
    if (dogfood_dir / "summary.json").is_file():
        summary = json.loads((dogfood_dir / "summary.json").read_text())
        result["panns_profiling_summary"] = summary
        (staging / "panns-profiling-summary.json").write_text(json.dumps(summary, indent=2))

    candidates_dir = ROOT / "candidate_technical_cards"
    manifest: list[dict[str, object]] = []
    if candidates_dir.is_dir():
        evidence_dir = staging / "candidate_technical_cards"
        evidence_dir.mkdir(exist_ok=True)
        for path in sorted(candidates_dir.iterdir()):
            if path.suffix not in (".json", ".npz"):
                continue
            data = path.read_bytes()
            (evidence_dir / path.name).write_bytes(data)
            manifest.append(
                {
                    "path": f"candidate_technical_cards/{path.name}",
                    "sha256": sha256(data).hexdigest(),
                    "size": len(data),
                }
            )
    result["candidate_evidence_manifest"] = manifest

    # Campaign result JSONs from every profiling workspace this session created.
    profiling_workspace_root = ROOT / ".torch-dae/profiling"
    campaign_results = []
    if profiling_workspace_root.is_dir():
        for campaign_dir in sorted(profiling_workspace_root.glob("profiling-*")):
            campaign_json = campaign_dir / "campaign-result.json"
            if campaign_json.is_file():
                campaign_results.append(json.loads(campaign_json.read_text()))
    result["campaign_results"] = campaign_results
    (staging / "campaign-results.json").write_text(json.dumps(campaign_results, indent=2))

    # --- Changed/new file inventory --------------------------------------------------------------
    _, changed_files, _ = run("git", "diff", "--name-only")
    inventory = sorted(set(changed_files.splitlines()) | set(untracked.splitlines()))
    result["changed_and_new_files"] = inventory
    (staging / "changed-and-new-files.txt").write_text("\n".join(inventory) + "\n")

    (staging / "result.json").write_text(json.dumps(result, indent=2, default=str))

    # --- Declared file manifest (SHA-256 inventory of every staged file) -----------------------
    # Built before the archive so it is itself an archive member; verified against the archive's
    # actual bytes below, after writing.
    manifest_entries: list[dict[str, str | int]] = [
        {
            "path": str(entry.relative_to(staging)),
            "sha256": sha256_file(entry),
            "size": entry.stat().st_size,
        }
        for entry in staging.rglob("*")
        if entry.is_file()
    ]
    file_manifest = sorted(manifest_entries, key=lambda item: str(item["path"]))
    (staging / "MANIFEST.json").write_text(json.dumps(file_manifest, indent=2))

    # --- Deterministic archive --------------------------------------------------------------------
    output_root.mkdir(parents=True, exist_ok=True)
    archive_path = output_root / "torch-dae-profiling-review.tar.gz"
    entries = sorted(staging.rglob("*"))
    with tarfile.open(archive_path, "w:gz", compresslevel=9) as tar:
        for entry in entries:
            arcname = "review/" + str(entry.relative_to(staging))
            tar.add(entry, arcname=arcname, filter=normalize_tarinfo, recursive=False)

    digest = sha256_file(archive_path)
    (output_root / "torch-dae-profiling-review.tar.gz.sha256").write_text(
        f"{digest}  torch-dae-profiling-review.tar.gz\n"
    )

    # --- Verify the archive against its own declared manifest -----------------------------------
    manifest_by_path: dict[str, dict[str, str | int]] = {
        str(item["path"]): item for item in file_manifest
    }
    verification_errors: list[str] = []
    with tarfile.open(archive_path, "r:gz") as tar:
        archive_paths = set()
        for member in tar.getmembers():
            if not member.isfile():
                continue
            relative = member.name.removeprefix("review/")
            if relative == "MANIFEST.json":
                continue  # self-referential: the manifest cannot declare its own hash
            archive_paths.add(relative)
            expected = manifest_by_path.get(relative)
            if expected is None:
                verification_errors.append(f"archive member not in declared manifest: {relative}")
                continue
            extracted = tar.extractfile(member)
            assert extracted is not None
            actual_sha256 = sha256(extracted.read()).hexdigest()
            if actual_sha256 != expected["sha256"]:
                verification_errors.append(
                    f"archive member content disagrees with declared manifest: {relative} "
                    f"(manifest={expected['sha256']} archive={actual_sha256})"
                )
        missing = set(manifest_by_path) - archive_paths
        for relative in sorted(missing):
            verification_errors.append(f"manifest entry missing from archive: {relative}")
    archive_verified = not verification_errors
    if not archive_verified:
        for error in verification_errors:
            print(f"ARCHIVE VERIFICATION FAILED: {error}", file=sys.stderr)

    top_level_result = {
        "archive_path": str(archive_path),
        "archive_sha256": digest,
        "archive_size_bytes": archive_path.stat().st_size,
        "staging_result_path": str(staging / "result.json"),
        "manifest_file_count": len(file_manifest),
        "archive_verified_against_manifest": archive_verified,
        "archive_verification_errors": verification_errors,
        "generated_at": result["generated_at"],
        "git_head": result["git"]["head"],
    }
    (output_root / "torch-dae-profiling-review.result.json").write_text(
        json.dumps(top_level_result, indent=2)
    )

    print(json.dumps(top_level_result, indent=2))
    return 0 if archive_verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
