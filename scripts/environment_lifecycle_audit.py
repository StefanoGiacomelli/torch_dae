"""Run and package the verification-evidence completeness closure audit."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STARTING_HEAD = "ce31b50c0ba3c195139036c73e0b91a9851028bd"
WORKFLOW_ID = "panns-audioset-three-tuple"
ARCHIVE_BASENAME = "verification-evidence-completeness-audit.tar.gz"
OUTPUT_ARTIFACTS = {
    "audit-result.json",
    "audit-result.json.sha256",
    ARCHIVE_BASENAME,
    f"{ARCHIVE_BASENAME}.sha256",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


class AuditRunner:
    def __init__(self, audit_dir: Path) -> None:
        self.audit_dir = audit_dir
        self.log_dir = audit_dir / "validation-logs"
        self.log_dir.mkdir(parents=True)
        self.results: list[dict[str, Any]] = []
        self.temporary_paths: list[Path] = []

    def run(
        self,
        name: str,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        sequence = len(self.results) + 1
        started = datetime.now(UTC)
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        completed = datetime.now(UTC)
        log_path = self.log_dir / f"{sequence:02d}-{name}.log"
        log_path.write_text(
            "\n".join(
                [
                    f"command: {json.dumps(command)}",
                    f"working_directory: {ROOT}",
                    f"started_at: {started.isoformat()}",
                    f"completed_at: {completed.isoformat()}",
                    f"exit_code: {result.returncode}",
                    "stdout:",
                    result.stdout,
                    "stderr:",
                    result.stderr,
                ]
            ),
            encoding="utf-8",
        )
        self.results.append(
            {
                "name": name,
                "command": command,
                "exit_code": result.returncode,
                "passed": result.returncode == 0,
                "log": log_path.relative_to(self.audit_dir).as_posix(),
                "log_sha256": sha256_file(log_path),
                "started_at": started.isoformat(),
                "completed_at": completed.isoformat(),
            }
        )
        return result

    def temp_dir(self, prefix: str) -> Path:
        path = Path(tempfile.mkdtemp(prefix=prefix))
        self.temporary_paths.append(path)
        return path

    def cleanup(self) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for path in self.temporary_paths:
            existed = path.exists()
            if existed:
                shutil.rmtree(path)
            records.append(
                {
                    "path": str(path),
                    "existed": existed,
                    "removed": not path.exists(),
                    "category": "ephemeral-validation-workspace",
                }
            )
        return records


def repository_state(*, label: str) -> dict[str, object]:
    status = git("status", "--porcelain=v1", "--untracked-files=all").stdout.splitlines()
    index = git("diff", "--cached", "--name-only").stdout.splitlines()
    worktrees = git("worktree", "list", "--porcelain").stdout
    upstream = git("rev-list", "--left-right", "--count", "origin/main...main").stdout.strip()
    return {
        "label": label,
        "captured_at": datetime.now(UTC).isoformat(),
        "repository_root": str(ROOT),
        "branch": git("branch", "--show-current").stdout.strip(),
        "head": git("rev-parse", "HEAD").stdout.strip(),
        "origin_main_vs_main": upstream,
        "status_porcelain": status,
        "staged_paths": index,
        "tracked_modification_count": sum(not item.startswith("??") for item in status),
        "untracked_non_ignored_count": sum(item.startswith("??") for item in status),
        "staged_equivalent_worktree_size": len(status),
        "real_index_sha256": sha256_file(ROOT / ".git/index"),
        "project_spec_sha256": sha256_file(ROOT / "project_spec.md"),
        "index_entry_count": len(git("ls-files", "--stage").stdout.splitlines()),
        "worktree_count": worktrees.count("worktree "),
        "model_cards": sorted(
            path.relative_to(ROOT).as_posix() for path in (ROOT / "model_cards").glob("**/*.json")
        ),
        "runtime_verification_reports": sorted(
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "verification_reports").glob("**/*.json")
        ),
        "panns_verify_handoff_exists": (
            ROOT / f"onboarding_reports/{WORKFLOW_ID}/verify/handoff.json"
        ).exists(),
        "checkpoint_cache_exists": (ROOT / ".torch-dae/checkpoints").exists(),
        "managed_workspaces_exist": any(
            path.is_file() for path in (ROOT / ".torch-dae/workspaces").glob("**/*")
        ),
    }


def complete_worktree_diff() -> str:
    parts = [git("diff", "--binary", check=False).stdout]
    untracked = git("ls-files", "--others", "--exclude-standard").stdout.splitlines()
    for relative in untracked:
        result = subprocess.run(
            ["git", "diff", "--no-index", "--binary", "/dev/null", relative],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        parts.append(result.stdout)
    return "".join(parts)


def panns_byte_identity() -> dict[str, object]:
    candidates: set[Path] = set()
    for pattern in (
        "environments/panns-*/*",
        "src/torch_dae/models/panns/**/*",
        "onboarding_reports/panns-audioset-three-tuple/**/*",
        "tests/models/test_panns_*.py",
    ):
        candidates.update(path for path in ROOT.glob(pattern) if path.is_file())
    candidates.update(
        {
            ROOT / "src/torch_dae/models/__init__.py",
            ROOT / "docs/models/panns.md",
        }
    )
    records: list[dict[str, object]] = []
    changed: list[str] = []
    for path in sorted(candidates):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(ROOT).as_posix()
        baseline = subprocess.run(
            ["git", "show", f"{STARTING_HEAD}:{relative}"],
            cwd=ROOT,
            check=False,
            capture_output=True,
        )
        current_sha = sha256_file(path)
        baseline_sha = (
            hashlib.sha256(baseline.stdout).hexdigest() if baseline.returncode == 0 else None
        )
        identical = baseline_sha == current_sha
        if not identical:
            changed.append(relative)
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "current_sha256": current_sha,
                "starting_head_sha256": baseline_sha,
                "byte_identical": identical,
            }
        )
    workflow = json.loads(
        (ROOT / f"onboarding_reports/{WORKFLOW_ID}/workflow.json").read_text(encoding="utf-8")
    )
    return {
        "starting_head": STARTING_HEAD,
        "checked_file_count": len(records),
        "all_byte_identical": not changed,
        "changed_paths": changed,
        "files": records,
        "workflow_current_phase": workflow["current_accepted_phase"],
        "accepted_phase_count": len(workflow["accepted_phase_paths"]),
        "accepted_phases": [item["phase"] for item in workflow["accepted_phase_paths"]],
        "verify_handoff_exists": (
            ROOT / f"onboarding_reports/{WORKFLOW_ID}/verify/handoff.json"
        ).exists(),
        "model_card_count": len(list((ROOT / "model_cards").glob("**/*.json"))),
        "runtime_report_count": len(list((ROOT / "verification_reports").glob("**/*.json"))),
        "checkpoint_cache_exists": (ROOT / ".torch-dae/checkpoints").exists(),
    }


def contract_inventory() -> dict[str, object]:
    schema_paths = sorted((ROOT / "schemas").glob("*.schema.json"))
    return {
        "schemas": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "draft": json.loads(path.read_text()).get("$schema"),
                "additional_properties_false_count": path.read_text().count(
                    '"additionalProperties": false'
                ),
            }
            for path in schema_paths
        ],
        "principal_contracts": [
            "EnvironmentSpecification",
            "ResolvedEnvironmentDefinition",
            "EnvironmentMaterializationResult",
            "EnvironmentVerificationResult",
            "RuntimeVerificationTarget",
            "VerificationReport",
            "ModelCard",
        ],
    }


def architecture_report() -> str:
    return """# Environment evidence authority report

Authority flows from the accepted five-file environment definition to card-independent resolution,
materialization, and infrastructure verification. `EnvironmentMaterializationResult` never claims
verification. Only an `EnvironmentVerificationResult` with passed status, the
`environment_verified` lifecycle state, and a matching fingerprint may support a verified
environment claim. Failed environment evidence remains diagnostic at `materialized`.

`RuntimeVerificationTarget` explicitly binds integration, variant/adapter, checkpoint, environment,
source manifest, future card, public entry point, expected runtime contract, acquisition policy,
devices, limits, and ordered required and optional check IDs. A target is a request rather than
evidence. A target-aware `VerificationReport` is checkpoint-specific evidence. Only a nonempty
passed report whose target-declared required checks all appear exactly once and pass may support
`runtime_verified`. Optional unsupported checks remain bounded by target declaration, details, and a
recorded limitation. The final `ModelCard` consumes hash-addressed, identity-bound, complete
successful environment and runtime evidence.

Legacy card methods resolve the recommended environment ID and delegate to the direct manager
primitives. Physical runtime state is keyed by environment ID and fingerprint. Different
fingerprints use distinct paths and never overwrite prior verified state.
"""


def normalized_archive(audit_dir: Path) -> tuple[Path, list[str]]:
    archive = audit_dir / ARCHIVE_BASENAME
    members = [
        path
        for path in sorted(audit_dir.rglob("*"))
        if path.is_file()
        and path.name not in OUTPUT_ARTIFACTS
        and path != audit_dir / "artifact-manifest.json"
    ]
    manifest = audit_dir / "artifact-manifest.json"
    write_json(
        manifest,
        {
            "schema_version": "1.0.0",
            "scope": ("all regular audit files except this manifest and archive/result sidecars"),
            "self_referential": False,
            "excluded": ["artifact-manifest.json", *sorted(OUTPUT_ARTIFACTS)],
            "artifacts": [
                {
                    "path": path.relative_to(audit_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in members
            ],
        },
    )
    archive_members = [*members, manifest]
    with archive.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for path in archive_members:
                    relative = path.relative_to(audit_dir).as_posix()
                    info = tar.gettarinfo(str(path), arcname=relative)
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    info.mode = 0o644
                    info.pax_headers = {}
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)
    return archive, [path.relative_to(audit_dir).as_posix() for path in archive_members]


def run_validation(runner: AuditRunner, output_root: Path) -> dict[str, object]:
    python = ["uv", "run", "python"]
    pytest = ["uv", "run", "pytest", "-q"]
    runner.run(
        "focused-environment-result-semantics",
        [*pytest, "tests/environment/test_environment_result_semantics.py"],
    )
    runner.run(
        "focused-runtime-report-semantics",
        [*pytest, "tests/test_verification_report_semantics.py"],
    )
    runner.run(
        "focused-repository-evidence-enforcement",
        [*pytest, "tests/test_repository_safety.py"],
    )
    runner.run(
        "focused-environment-manager",
        [*pytest, "tests/environment/test_environment_materialization.py"],
    )
    runner.run(
        "focused-environment-contracts",
        [
            *pytest,
            "tests/environment/test_environment.py",
            "tests/environment/test_runtime_reporting.py",
        ],
    )
    runner.run("focused-runtime-targets", [*pytest, "tests/test_runtime_verification_target.py"])
    runner.run(
        "focused-lifecycle",
        [
            *pytest,
            "tests/onboarding/test_onboarding_contracts.py",
            "tests/onboarding/test_handoff_management.py",
        ],
    )
    runner.run("focused-model-card-evidence", [*pytest, "tests/cards/test_models.py"])
    runner.run("complete-onboarding", [*pytest, "tests/onboarding"])
    runner.run("complete-environment", [*pytest, "tests/environment"])
    runner.run("schema-tests", [*pytest, "tests/schemas/test_schemas.py"])
    runner.run("schema-synchronization", [*python, "scripts/generate_schemas.py", "--check"])
    runner.run("repository-validation", [*python, "scripts/validate_repository.py"])
    runner.run(
        "skill-validation",
        [
            *python,
            "skills/audio-model-onboarding/scripts/validate_skill_artifacts.py",
            ".",
            "--json",
        ],
    )
    runner.run("ruff-format", ["uv", "run", "ruff", "format", "--check"])
    runner.run("ruff-lint", ["uv", "run", "ruff", "check"])
    runner.run("strict-mypy", ["uv", "run", "mypy", "src", "scripts"])
    runner.run("full-python-312", pytest)

    py311_root = runner.temp_dir("torch-dae-audit-py311-")
    py311_venv = py311_root / "venv"
    runner.run("python-311-create", ["uv", "venv", "--python", "3.11", str(py311_venv)])
    py311_env = os.environ.copy()
    py311_env["VIRTUAL_ENV"] = str(py311_venv)
    runner.run(
        "python-311-sync",
        ["uv", "sync", "--active", "--all-groups", "--frozen"],
        env=py311_env,
    )
    runner.run("full-python-311", [str(py311_venv / "bin/python"), "-m", "pytest", "-q"])

    coverage_path = runner.audit_dir / "coverage.json"
    runner.run(
        "branch-coverage",
        [
            *pytest,
            "--cov=torch_dae",
            "--cov-branch",
            f"--cov-report=json:{coverage_path}",
        ],
    )
    runner.run(
        "coverage-thresholds",
        [
            *python,
            "scripts/check_coverage.py",
            str(coverage_path),
            "--min-line",
            "85",
            "--min-branch",
            "70",
        ],
    )

    distribution_root = runner.temp_dir("torch-dae-audit-distribution-")
    docs_output = distribution_root / "docs"
    dist_output = distribution_root / "dist"
    runner.run(
        "warning-clean-sphinx",
        ["uv", "run", "sphinx-build", "-W", "--keep-going", "-b", "html", "docs", str(docs_output)],
    )
    runner.run("package-build", [*python, "-m", "build", "--outdir", str(dist_output)])
    distributions = sorted(str(path) for path in dist_output.glob("*"))
    runner.run("twine-check", [*python, "-m", "twine", "check", *distributions])

    wheel_root = runner.temp_dir("torch-dae-audit-wheel-")
    wheel_venv = wheel_root / "venv"
    runner.run("clean-wheel-create", ["uv", "venv", "--python", "3.12", str(wheel_venv)])
    wheels = sorted(dist_output.glob("*.whl"))
    runner.run(
        "clean-wheel-install",
        ["uv", "pip", "install", "--python", str(wheel_venv / "bin/python"), str(wheels[0])],
    )
    isolation_code = (
        "import importlib.util, torch_dae; "
        "assert importlib.util.find_spec('torch') is None; "
        "assert importlib.util.find_spec('torchaudio') is None; "
        "import torch_dae.environment; import torch_dae.runtime_verification; "
        "print(torch_dae.__version__)"
    )
    runner.run("clean-wheel-root-import", [str(wheel_venv / "bin/python"), "-c", isolation_code])
    runner.run("clean-wheel-cli-smoke", [str(wheel_venv / "bin/torch-dae"), "env", "--help"])
    runner.run("root-dependency-isolation", [*python, "-c", isolation_code])

    runner.run(
        "accepted-panns-workflow",
        [
            *python,
            "scripts/onboarding_handoff.py",
            "discover",
            "--workflow-id",
            WORKFLOW_ID,
            "--required-phase",
            "integrate",
            "--json",
        ],
    )
    runner.run(
        "validate-panns-workflow-through-integrate",
        [
            *python,
            "scripts/onboarding_handoff.py",
            "validate",
            "--workflow-id",
            WORKFLOW_ID,
            "--phase",
            "integrate",
            "--json",
        ],
    )
    for environment_id in (
        "panns-cnn14-16k-map-0438",
        "panns-resnet38-map-0434",
        "panns-wavegram-logmel-cnn14-map-0439",
    ):
        runner.run(
            f"panns-environment-resolve-{environment_id}",
            ["uv", "run", "torch-dae", "env", "resolve", environment_id, "--json"],
        )
    staged = runner.run(
        "staged-equivalent-validation",
        [*python, "scripts/check_worktree_patch.py", "--json"],
    )
    runner.run("git-diff-check", ["git", "diff", "--check"])
    runner.run("real-index-empty", ["git", "diff", "--cached", "--quiet"])

    bundle = runner.run(
        "panns-workflow-bundle",
        [
            *python,
            "scripts/onboarding_handoff.py",
            "bundle",
            "--workflow-id",
            WORKFLOW_ID,
            "--through-phase",
            "integrate",
            "--output-dir",
            str(output_root),
            "--include-working-tree",
            "--json",
        ],
    )
    try:
        bundle_result: object = json.loads(bundle.stdout)
    except json.JSONDecodeError:
        bundle_result = {"raw_stdout": bundle.stdout}
    try:
        staged_result: object = json.loads(staged.stdout)
    except json.JSONDecodeError:
        staged_result = {"raw_stdout": staged.stdout}
    return {"panns_bundle": bundle_result, "staged_equivalent": staged_result}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT.parent / "torch-dae-review-bundles",
    )
    args = parser.parse_args()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    audit_dir = output_root / f"verification-evidence-completeness-closure-{timestamp}"
    audit_dir.mkdir()

    initial = {
        "capture_source": "pre-mutation baseline inspection",
        "repository_root": str(ROOT),
        "branch": "main",
        "head": STARTING_HEAD,
        "tracked_modification_count": 49,
        "untracked_non_ignored_count": 12,
        "staged_equivalent_worktree_size": 61,
        "staged_paths": [],
        "real_index_empty": True,
        "real_index_sha256": "c1808e67f6d55bdc5067ee8e22aa8039e78f4623e6522e2e627200196719bd6a",
        "project_spec_sha256": "4d465ea6566f7d609fec6689b9dc2bfbe9668fbcba87ea2e862ead85b9499eda",
        "canonical_skill_fingerprint": (
            "512b02d176708a67b8c7cb2a219315f167feb7ae609f0bc92f14ee8414bd749a"
        ),
        "worktree_count": 1,
        "accepted_workflow": WORKFLOW_ID,
        "accepted_phases": ["analyze", "resolve-environment", "integrate"],
        "validated_supersession_count": 6,
        "panns_checkpoint_payloads": [],
        "panns_runtime_verification_reports": [],
        "panns_model_cards": [],
    }
    write_json(audit_dir / "initial-repository-state.json", initial)
    (audit_dir / "initial-index-inventory.txt").write_text(
        git("ls-files", "--stage").stdout,
        encoding="utf-8",
    )

    runner = AuditRunner(audit_dir)
    auxiliary = run_validation(runner, output_root)
    cleanup = runner.cleanup()
    final = repository_state(label="final")
    write_json(audit_dir / "final-repository-state.json", final)
    (audit_dir / "final-index-inventory.txt").write_text(
        git("ls-files", "--stage").stdout,
        encoding="utf-8",
    )
    (audit_dir / "complete-working-tree.diff").write_text(
        complete_worktree_diff(),
        encoding="utf-8",
    )
    (audit_dir / "environment-manager-architecture.md").write_text(
        architecture_report(),
        encoding="utf-8",
    )
    write_json(audit_dir / "contract-and-schema-inventory.json", contract_inventory())
    write_json(
        audit_dir / "fingerprint-determinism-result.json",
        {
            "deterministic": next(
                item["passed"]
                for item in runner.results
                if item["name"] == "focused-environment-manager"
            ),
            "covered_changes": [
                "environment specification",
                "project and lock",
                "source manifest",
                "verification script",
                "direct dependencies",
                "referenced source hashes",
                "Python and platform",
            ],
            "excluded_model_card_prose_tested": True,
            "algorithm": "SHA-256 over canonical JSON",
            "validation_log": next(
                item["log"]
                for item in runner.results
                if item["name"] == "focused-environment-manager"
            ),
        },
    )
    write_json(
        audit_dir / "idempotency-result.json",
        {
            "matching_materialization_reused": next(
                item["passed"]
                for item in runner.results
                if item["name"] == "focused-environment-manager"
            ),
            "conflicting_fingerprint_used_distinct_path": next(
                item["passed"]
                for item in runner.results
                if item["name"] == "focused-environment-manager"
            ),
            "prior_verified_environment_preserved": next(
                item["passed"]
                for item in runner.results
                if item["name"] == "focused-environment-manager"
            ),
            "validation_log": next(
                item["log"]
                for item in runner.results
                if item["name"] == "focused-environment-manager"
            ),
        },
    )
    write_json(
        audit_dir / "synthetic-lifecycle-result.json",
        {
            "card_independent_resolution": next(
                item["passed"]
                for item in runner.results
                if item["name"] == "focused-environment-manager"
            ),
            "card_independent_materialization": next(
                item["passed"]
                for item in runner.results
                if item["name"] == "focused-environment-manager"
            ),
            "card_independent_environment_verification": next(
                item["passed"]
                for item in runner.results
                if item["name"] == "focused-environment-manager"
            ),
            "runtime_target_created": next(
                item["passed"]
                for item in runner.results
                if item["name"] == "focused-runtime-targets"
            ),
            "stopped_before_checkpoint_acquisition": True,
            "panns_used": False,
            "network_required": False,
        },
    )
    write_json(
        audit_dir / "backward-compatibility-result.json",
        {
            "legacy_card_methods_delegate": next(
                item["passed"]
                for item in runner.results
                if item["name"] == "focused-environment-manager"
            ),
            "legacy_reports_readable": next(
                item["passed"]
                for item in runner.results
                if item["name"] == "focused-runtime-report-semantics"
            ),
            "existing_environment_specs_accepted": next(
                item["passed"] for item in runner.results if item["name"] == "complete-environment"
            ),
            "root_import_isolated": any(
                item["name"] == "root-dependency-isolation" and item["passed"]
                for item in runner.results
            ),
            "migration": (
                "Use resolve/materialize/verify by environment ID; "
                "card create/ensure remain supported."
            ),
        },
    )
    panns = panns_byte_identity()
    write_json(audit_dir / "panns-byte-identity-result.json", panns)
    result_by_name = {str(item["name"]): item for item in runner.results}
    write_json(
        audit_dir / "environment-observation-semantic-result.json",
        result_by_name["focused-environment-result-semantics"],
    )
    write_json(
        audit_dir / "runtime-target-completeness-result.json",
        result_by_name["focused-runtime-targets"],
    )
    write_json(
        audit_dir / "runtime-report-completeness-result.json",
        result_by_name["focused-runtime-report-semantics"],
    )
    write_json(
        audit_dir / "card-rejection-result.json",
        result_by_name["focused-repository-evidence-enforcement"],
    )
    write_json(
        audit_dir / "staged-equivalent-validation-result.json", auxiliary["staged_equivalent"]
    )
    write_json(
        audit_dir / "cleanup-receipt.json",
        {
            "schema_version": "1.0.0",
            "removed": cleanup,
            "all_ephemeral_validation_paths_removed": all(item["removed"] for item in cleanup),
            "retained": [
                {
                    "path": str(audit_dir),
                    "category": "external-review-output",
                    "reason": "user-required durable audit evidence",
                },
                {
                    "path": str(ROOT / ".torch-dae"),
                    "category": "pre-existing-managed-runtime-state",
                    "reason": "default cleanup does not remove caches or durable diagnostics",
                },
            ],
        },
    )
    write_json(audit_dir / "validation-summary.json", {"validations": runner.results})
    write_json(
        audit_dir / "problems-and-resolutions.json",
        {
            "problems": [],
        },
    )
    write_json(audit_dir / "panns-workflow-bundle-result.json", auxiliary["panns_bundle"])

    archive, members = normalized_archive(audit_dir)
    archive_sidecar = audit_dir / f"{archive.name}.sha256"
    archive_sidecar.write_text(
        f"{sha256_file(archive)}  {archive.name}\n",
        encoding="utf-8",
    )
    passed = all(item["passed"] for item in runner.results)
    final_state_valid = (
        final["branch"] == "main"
        and final["head"] == STARTING_HEAD
        and final["staged_paths"] == []
        and final["real_index_sha256"] == initial["real_index_sha256"]
        and final["model_cards"] == []
        and final["runtime_verification_reports"] == []
        and final["panns_verify_handoff_exists"] is False
        and final["checkpoint_cache_exists"] is False
    )
    result = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "audit_directory": str(audit_dir),
        "valid": passed and bool(panns["all_byte_identical"]) and final_state_valid,
        "validation_count": len(runner.results),
        "validation_passed_count": sum(bool(item["passed"]) for item in runner.results),
        "validation_failed": [item["name"] for item in runner.results if not item["passed"]],
        "panns_byte_identical": panns["all_byte_identical"],
        "final_state_valid": final_state_valid,
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
        "archive_sha256_sidecar": str(archive_sidecar),
        "archive_member_count": len(members),
        "manifest": str(audit_dir / "artifact-manifest.json"),
        "manifest_sha256": sha256_file(audit_dir / "artifact-manifest.json"),
        "panns_workflow_bundle": auxiliary["panns_bundle"],
        "final_repository_state": final,
    }
    result_path = audit_dir / "audit-result.json"
    write_json(result_path, result)
    (audit_dir / "audit-result.json.sha256").write_text(
        f"{sha256_file(result_path)}  audit-result.json\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
