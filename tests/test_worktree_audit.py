from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path


def run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)


def test_final_state_barrier_and_audit_archive_are_read_only_and_normalized(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / ".gitignore").write_text(".torch-dae/\n")
    (repository / "tracked.txt").write_text("before\n")
    phase = repository / "onboarding_reports/workflow/integrate"
    phase.mkdir(parents=True)
    (phase / "handoff.json").write_text("{}\n")
    results = phase / "environment-results"
    results.mkdir()
    result_path = results / "environment.json"
    fingerprint = "a" * 64
    result_path.write_text(
        json.dumps(
            {
                "environment_id": "synthetic-environment",
                "environment_fingerprint": fingerprint,
            }
        )
    )
    run(["git", "init"], cwd=repository)
    run(["git", "add", "."], cwd=repository)
    run(
        [
            "git",
            "-c",
            "user.name=Test User",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "baseline",
        ],
        cwd=repository,
    )
    (repository / "tracked.txt").write_text("after\n")
    (repository / "untracked.txt").write_text("new\n")
    report = (
        repository
        / ".torch-dae/reports/environments/synthetic-environment"
        / fingerprint
        / "verification-commands/command.json"
    )
    report.parent.mkdir(parents=True)
    report.write_text('{"status":"success"}\n')
    validation = repository / ".torch-dae/reports/final-state.json"
    check_script = repo_root / "scripts/check_final_state.py"
    common = [
        "python",
        str(check_script),
        "--repository-root",
        str(repository),
        "--workflow-id",
        "workflow",
        "--phase",
        "integrate",
    ]
    run([*common, "--record", str(validation)], cwd=repo_root)
    run([*common, "--check", str(validation)], cwd=repo_root)
    recorded = json.loads(validation.read_text())
    assert recorded["final_state_equal"] is True
    assert recorded["real_index_empty"] is True
    assert recorded["staged_equivalent_inventory"] == ["tracked.txt", "untracked.txt"]

    before = run(["git", "status", "--porcelain=v1", "-uall"], cwd=repository).stdout
    output_dir = tmp_path / "audit"
    run(
        [
            "python",
            "-m",
            "scripts.generate_worktree_audit",
            "--repository-root",
            str(repository),
            "--output-dir",
            str(output_dir),
            "--artifact-stem",
            "closure-audit",
            "--validation-result",
            str(validation),
        ],
        cwd=repo_root,
    )
    after = run(["git", "status", "--porcelain=v1", "-uall"], cwd=repository).stdout
    assert after == before

    archive = output_dir / "closure-audit.tar.gz"
    assert int.from_bytes(archive.read_bytes()[4:8], "little") == 0
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        names = {member.name for member in members}
        assert all(member.uid == member.gid == member.mtime == 0 for member in members)
        assert all(not member.uname and not member.gname for member in members)
        assert "working-tree/tracked.txt" in names
        assert "working-tree/untracked.txt" in names
        assert "runtime-command-logs/reports/environments/synthetic-environment/" in "\n".join(
            names
        )
        assert "MANIFEST.sha256" in names
        manifest = bundle.extractfile("MANIFEST.sha256")
        assert manifest is not None
        lines = manifest.read().decode().splitlines()
        assert len(lines) == len(names) - 1
        assert {line.split("  ", 1)[1] for line in lines} == names - {"MANIFEST.sha256"}
        assert not any(
            "__MACOSX" in name or "/._" in name or name.endswith(".DS_Store") for name in names
        )
        assert not any(name.startswith("environments/") for name in names)
        assert not any("checkpoints" in name for name in names)

    sidecar_digest = (output_dir / "closure-audit.tar.gz.sha256").read_text().split()[0]
    assert sidecar_digest == hashlib.sha256(archive.read_bytes()).hexdigest()
    audit_result = json.loads((output_dir / "closure-audit.result.json").read_text())
    assert audit_result["status"] == "passed"
    assert audit_result["final_state_equal"] is True
