"""Environment materialization and execution manager."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tomllib
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

from torch_dae.contracts import contained_path, ensure_canonical_id, ensure_repository_relative
from torch_dae.core.errors import (
    EnvironmentAlreadyExistsError,
    EnvironmentIdentityMismatchError,
    EnvironmentMaterializationError,
    EnvironmentNotFoundError,
    EnvironmentVerificationError,
    PythonInterpreterUnavailableError,
)
from torch_dae.core.registry import ModelCardRegistry
from torch_dae.environment.fingerprint import (
    FingerprintInputs,
    calculate_environment_fingerprint,
    canonical_platform_tag,
    local_package_identity,
)
from torch_dae.environment.materialization import materialization_path
from torch_dae.environment.policy import ExecutionPolicy
from torch_dae.environment.runtime import (
    EnvironmentMaterializationRecord,
    LocalWheelCacheRecord,
    RuntimeReportSink,
    VerificationExecutionRecord,
    result_to_verification,
    utc_now,
    write_json_atomic,
)
from torch_dae.environment.sources import (
    SourceContext,
    SourceManager,
    installed_distributions,
    replace_tree,
    sha256_file,
    valid_single_wheel,
    wheel_distribution_metadata,
)
from torch_dae.environment.specification import (
    EnvironmentSourcesManifest,
    EnvironmentSpecification,
    SourceInstallationType,
)
from torch_dae.environment.subprocess import CommandExecutor, ManagedProcessResult

COMPLETE_MARKER = ".torch-dae-complete"
MATERIALIZATION_JSON = "torch-dae-materialization.json"


@dataclass(frozen=True)
class InstalledSource:
    """Installed source metadata for a resolved environment."""

    source_id: str
    location: str
    revision: str | None = None


@dataclass(frozen=True)
class ResolvedEnvironment:
    """Materialized environment metadata."""

    environment_id: str
    model_card_id: str
    root: Path
    python_executable: Path
    fingerprint: str
    python_version: str
    platform: str
    installed_packages: Mapping[str, str]
    installed_sources: tuple[InstalledSource, ...]
    valid: bool


@dataclass(frozen=True)
class EnvironmentInfo:
    """Read-only local-state inspection."""

    model_card_id: str
    specification_path: Path
    specification_exists: bool
    fingerprint: str | None
    expected_path: Path | None
    materialized: bool
    status: str
    stale_fingerprints: tuple[str, ...] = ()
    verification_status: str | None = None


@dataclass(frozen=True)
class EnvironmentVerification:
    """Environment verification result."""

    model_card_id: str
    passed: bool
    details: str
    status: str = "invalid"


@dataclass(frozen=True)
class EnvironmentInputs:
    """Loaded committed inputs for one model card environment."""

    card_id: str
    specification: EnvironmentSpecification
    sources_manifest: EnvironmentSourcesManifest
    specification_path: Path
    project_path: Path
    lock_path: Path
    sources_path: Path
    verification_script: Path
    fingerprint: str
    package_identity: str
    platform: str


def discover_repository_root(start: Path | None = None) -> Path:
    """Find the repository root containing `project_spec.md`."""

    current = (start or Path.cwd()).resolve()
    for path in (current, *current.parents):
        if (path / "project_spec.md").is_file():
            return path
    raise FileNotFoundError("could not discover repository root containing project_spec.md")


class EnvironmentManager:
    """Create, verify, inspect, remove, and run model-specific environments."""

    def __init__(
        self,
        repository_root: Path,
        *,
        policy: ExecutionPolicy | None = None,
        executor: CommandExecutor | None = None,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.runtime_root = self.repository_root / ".torch-dae"
        self.policy = policy or ExecutionPolicy()
        self.executor = executor or CommandExecutor()

    @classmethod
    def from_repository_root(
        cls,
        start: Path | None = None,
        *,
        policy: ExecutionPolicy | None = None,
    ) -> EnvironmentManager:
        return cls(discover_repository_root(start), policy=policy)

    def specification_path(self, model_card_id: str) -> Path:
        """Return the committed environment specification path for `model_card_id`."""

        ensure_canonical_id(model_card_id)
        return contained_path(
            self.repository_root / "environments", model_card_id, "environment.json"
        )

    def load_specification(self, model_card_id: str) -> EnvironmentSpecification:
        """Load and validate the committed environment specification."""

        path = self.specification_path(model_card_id)
        specification = EnvironmentSpecification.model_validate_json(path.read_text())
        if specification.model_card_id != model_card_id:
            raise EnvironmentIdentityMismatchError(
                "environment specification model_card_id does not match request"
            )
        return specification

    def load_sources_manifest(
        self, specification: EnvironmentSpecification
    ) -> EnvironmentSourcesManifest:
        """Load and validate the source manifest referenced by `specification`."""

        path = contained_path(self.repository_root, specification.sources_file)
        if not path.exists():
            raise FileNotFoundError(f"missing environment source manifest: {path}")
        manifest = EnvironmentSourcesManifest.model_validate_json(path.read_text())
        if manifest.environment_id != specification.environment_id:
            raise EnvironmentIdentityMismatchError(
                "environment source manifest environment_id does not match specification"
            )
        return manifest

    def fingerprint_for(self, specification: EnvironmentSpecification) -> str:
        """Calculate the expected fingerprint for `specification`."""

        lock_path = contained_path(self.repository_root, specification.lockfile)
        if not lock_path.exists():
            raise FileNotFoundError(f"missing environment lock file: {lock_path}")
        sources_manifest = self.load_sources_manifest(specification)
        inputs = FingerprintInputs(
            specification=specification,
            lockfile_bytes=lock_path.read_bytes(),
            sources_manifest=sources_manifest,
            target_platform=canonical_platform_tag(),
            local_package_identity=local_package_identity(self.repository_root),
        )
        return calculate_environment_fingerprint(inputs)

    def create(self, model_card_id: str) -> ResolvedEnvironment:
        """Create the current fingerprint target and fail if any target state exists."""

        inputs = self._load_inputs_for_card(model_card_id)
        target = materialization_path(self.runtime_root, model_card_id, inputs.fingerprint)
        if target.exists():
            raise EnvironmentAlreadyExistsError(f"environment already exists: {target}")
        return self._materialize(inputs, target)

    def ensure(self, model_card_id: str) -> ResolvedEnvironment:
        """Reuse a valid environment or create/rebuild the current fingerprint target."""

        inputs = self._load_inputs_for_card(model_card_id)
        target = materialization_path(self.runtime_root, model_card_id, inputs.fingerprint)
        verification = self._verify_target(inputs, target)
        if verification.passed:
            return self._resolved_from_record(inputs, target)
        if target.exists():
            replace_tree(target)
        return self._materialize(inputs, target)

    def verify(self, model_card_id: str) -> EnvironmentVerification:
        """Verify the current expected materialization without mutation."""

        inputs = self._load_inputs_for_card(model_card_id)
        target = materialization_path(self.runtime_root, model_card_id, inputs.fingerprint)
        verification = self._verify_target(inputs, target)
        if not verification.passed:
            raise EnvironmentVerificationError(verification.details)
        return verification

    def remove(self, model_card_id: str) -> None:
        """Remove all local environment materializations for one card."""

        try:
            ensure_canonical_id(model_card_id)
        except ValueError as exc:
            raise EnvironmentMaterializationError(
                f"invalid environment ID: {model_card_id}"
            ) from exc
        root = contained_path(self.runtime_root / "environments", model_card_id)
        if root.exists():
            shutil.rmtree(root)

    def info(self, model_card_id: str) -> EnvironmentInfo:
        """Inspect committed and local environment state without mutation."""

        try:
            specification_path = self.specification_path(model_card_id)
        except ValueError as exc:
            raise EnvironmentMaterializationError(
                f"invalid environment ID: {model_card_id}"
            ) from exc
        if not specification_path.exists():
            return EnvironmentInfo(
                model_card_id,
                specification_path,
                False,
                None,
                None,
                False,
                "missing-specification",
            )
        inputs = self._load_inputs_for_card(model_card_id)
        expected = materialization_path(self.runtime_root, model_card_id, inputs.fingerprint)
        verification = self._verify_target(inputs, expected)
        card_runtime_root = contained_path(self.runtime_root / "environments", model_card_id)
        stale = (
            tuple(
                sorted(
                    path.name
                    for path in card_runtime_root.iterdir()
                    if path.is_dir() and path.name != inputs.fingerprint and path.name != ".failed"
                )
            )
            if card_runtime_root.exists()
            else ()
        )
        return EnvironmentInfo(
            model_card_id=model_card_id,
            specification_path=specification_path,
            specification_exists=True,
            fingerprint=inputs.fingerprint,
            expected_path=expected,
            materialized=expected.exists(),
            status=verification.status,
            stale_fingerprints=stale,
            verification_status=verification.details,
        )

    def run(self, model_card_id: str, command: list[str]) -> ManagedProcessResult:
        """Ensure the environment and execute `command` inside it without shell activation."""

        if not command:
            raise ValueError("command must not be empty")
        resolved = self.ensure(model_card_id)
        env = {
            "VIRTUAL_ENV": str(resolved.root),
            "PATH": str(resolved.python_executable.parent)
            + os.pathsep
            + os.environ.get("PATH", ""),
        }
        return self.executor.run(
            command,
            cwd=self.repository_root,
            env=env,
            env_remove=python_env_remove(),
            timeout=self.policy.command_timeout_seconds,
            check=False,
        )

    def info_json(self, model_card_id: str) -> str:
        """Return JSON for CLI output."""

        info = self.info(model_card_id)
        return json.dumps(
            {
                "model_card_id": info.model_card_id,
                "specification_path": str(info.specification_path),
                "specification_exists": info.specification_exists,
                "fingerprint": info.fingerprint,
                "expected_path": str(info.expected_path) if info.expected_path else None,
                "materialized": info.materialized,
                "status": info.status,
                "stale_fingerprints": info.stale_fingerprints,
                "verification_status": info.verification_status,
            },
            indent=2,
            sort_keys=True,
        )

    def _load_inputs_for_card(self, model_card_id: str) -> EnvironmentInputs:
        registry = ModelCardRegistry(self.repository_root)
        try:
            card = registry.get_card(model_card_id)
        except KeyError as exc:
            raise EnvironmentMaterializationError(f"model card not found: {model_card_id}") from exc
        recommended = card.usage.recommended_environment
        if card.card_id != model_card_id:
            raise EnvironmentIdentityMismatchError("requested card ID does not match model card")
        if not recommended.verified:
            raise EnvironmentMaterializationError(
                "environment recreation requires a verified recommended environment"
            )
        specification_path = contained_path(self.repository_root, recommended.specification)
        if specification_path != self.specification_path(model_card_id):
            raise EnvironmentIdentityMismatchError(
                "recommended environment specification path does not match card ID"
            )
        specification = self.load_specification(model_card_id)
        if specification.environment_id != recommended.environment_id:
            raise EnvironmentIdentityMismatchError("environment ID mismatch between card and spec")
        if recommended.lockfile != specification.lockfile:
            raise EnvironmentIdentityMismatchError(
                "environment lockfile mismatch between card and spec"
            )
        self._validate_environment_artifact_paths(model_card_id, specification)
        lock_path = contained_path(self.repository_root, specification.lockfile)
        project_path = contained_path(self.repository_root, specification.project_file)
        sources_path = contained_path(self.repository_root, specification.sources_file)
        verification_script = contained_path(
            self.repository_root, specification.verification.script
        )
        for path, label in (
            (lock_path, "lock file"),
            (project_path, "project file"),
            (sources_path, "source manifest"),
            (verification_script, "verification script"),
        ):
            if not path.exists():
                raise EnvironmentMaterializationError(f"missing {label}: {path}")
        sources_manifest = self.load_sources_manifest(specification)
        package_identity = local_package_identity(self.repository_root)
        platform = canonical_platform_tag()
        inputs = FingerprintInputs(
            specification=specification,
            lockfile_bytes=lock_path.read_bytes(),
            sources_manifest=sources_manifest,
            target_platform=platform,
            local_package_identity=package_identity,
        )
        return EnvironmentInputs(
            card_id=model_card_id,
            specification=specification,
            sources_manifest=sources_manifest,
            specification_path=specification_path,
            project_path=project_path,
            lock_path=lock_path,
            sources_path=sources_path,
            verification_script=verification_script,
            fingerprint=calculate_environment_fingerprint(inputs),
            package_identity=package_identity,
            platform=platform,
        )

    def _validate_environment_artifact_paths(
        self,
        model_card_id: str,
        specification: EnvironmentSpecification,
    ) -> None:
        expected = {
            "lockfile": f"environments/{model_card_id}/uv.lock",
            "project_file": f"environments/{model_card_id}/pyproject.toml",
            "sources_file": f"environments/{model_card_id}/sources.json",
            "verification script": f"environments/{model_card_id}/verify_environment.py",
        }
        actual = {
            "lockfile": specification.lockfile,
            "project_file": specification.project_file,
            "sources_file": specification.sources_file,
            "verification script": specification.verification.script,
        }
        for label, value in actual.items():
            ensure_repository_relative(value)
            contained_path(self.repository_root, value)
            if value != expected[label]:
                raise EnvironmentIdentityMismatchError(f"{label} path must be {expected[label]}")

    def _materialize(self, inputs: EnvironmentInputs, target: Path) -> ResolvedEnvironment:
        created_at = utc_now()
        target.mkdir(parents=True, exist_ok=False)
        record = self._base_record(inputs, "building", created_at)
        metadata_path = target / MATERIALIZATION_JSON
        report_sink = RuntimeReportSink(
            self.runtime_root,
            "reports",
            "environments",
            inputs.card_id,
            inputs.fingerprint,
        )
        original_executor = self.executor
        self.executor = original_executor.with_report_sink(report_sink)
        write_json_atomic(metadata_path, record)
        try:
            interpreter = self._resolve_python(inputs.specification)
            self._create_virtual_environment(target, interpreter)
            self._sync_locked_environment(inputs, target, interpreter)
            python_executable = environment_python(target)
            actual_python = self._inspect_python(python_executable)
            local_wheel, local_wheel_sha = self._build_local_wheel(inputs.package_identity)
            self.executor.run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(python_executable),
                    "--no-deps",
                    *self.policy.uv_flags(),
                    str(local_wheel),
                ],
                operation="local-wheel-install",
                cwd=self.repository_root,
                env_remove=python_env_remove(),
                timeout=self.policy.command_timeout_seconds,
                check=True,
            )
            source_records = SourceManager(executor=self.executor, policy=self.policy).materialize(
                inputs.sources_manifest,
                SourceContext(
                    repository_root=self.repository_root,
                    runtime_root=self.runtime_root,
                    environment_root=target,
                    python_executable=python_executable,
                    lockfile_path=inputs.lock_path,
                    lockfile_sha256=sha256_file(inputs.lock_path),
                    python_version=actual_python,
                    platform=inputs.platform,
                    local_package_wheel_path=local_wheel,
                    local_package_wheel_sha256=local_wheel_sha,
                    local_package_identity=inputs.package_identity,
                ),
            )
            dependency_check = self.executor.run(
                [
                    "uv",
                    "pip",
                    "check",
                    "--python",
                    str(python_executable),
                    *self.policy.uv_flags(),
                ],
                operation="dependency-check",
                cwd=self.repository_root,
                env_remove=python_env_remove(),
                timeout=self.policy.command_timeout_seconds,
                check=False,
            )
            if dependency_check.returncode != 0:
                raise EnvironmentMaterializationError("dependency check failed")
            packages = tuple(installed_distributions(python_executable, self.executor).values())
            verification_result = self._execute_verification_script(
                inputs,
                target,
                python_executable,
            )
            verification = verification_from_result(verification_result)
            if verification.status != "valid":
                raise EnvironmentVerificationError(verification.details)
            complete_data = record.model_dump()
            complete_data.update(
                {
                    "status": "complete",
                    "python_actual_version": actual_python,
                    "python_executable": str(python_executable),
                    "python_provider": "uv-or-system",
                    "completed_at": utc_now(),
                    "local_package_wheel_sha256": local_wheel_sha,
                    "installed_packages": packages,
                    "installed_sources": source_records,
                    "verification_result": verification,
                    "command_log_references": tuple(report_sink.references),
                }
            )
            complete = EnvironmentMaterializationRecord.model_validate(complete_data)
            write_json_atomic(metadata_path, complete)
            marker = target / COMPLETE_MARKER
            tmp_marker = target / f".{COMPLETE_MARKER}.tmp"
            tmp_marker.write_text("complete\n", encoding="utf-8")
            os.replace(tmp_marker, marker)
            return self._resolved_from_record(inputs, target)
        except Exception:
            if target.exists():
                failed_data = record.model_dump()
                failed_data.update(
                    {
                        "status": "failed",
                        "completed_at": utc_now(),
                        "command_log_references": tuple(report_sink.references),
                    }
                )
                write_json_atomic(
                    metadata_path,
                    EnvironmentMaterializationRecord.model_validate(failed_data),
                )
            failed_root = contained_path(
                self.runtime_root / "environments",
                inputs.card_id,
                ".failed",
                inputs.fingerprint,
            )
            if failed_root.exists():
                replace_tree(failed_root)
            failed_root.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                shutil.move(str(target), str(failed_root))
            raise
        finally:
            self.executor = original_executor

    def _base_record(
        self,
        inputs: EnvironmentInputs,
        status: Literal["building", "complete", "failed"],
        created_at: str,
    ) -> EnvironmentMaterializationRecord:
        return EnvironmentMaterializationRecord(
            schema_version="1.0.0",
            status=status,
            card_id=inputs.card_id,
            environment_id=inputs.specification.environment_id,
            fingerprint=inputs.fingerprint,
            platform=inputs.platform,
            python_requested_version=inputs.specification.python.resolved_version,
            created_at=created_at,
            environment_spec_sha256=sha256_file(inputs.specification_path),
            lockfile_sha256=sha256_file(inputs.lock_path),
            source_manifest_sha256=sha256_file(inputs.sources_path),
            local_package_identity=inputs.package_identity,
        )

    def _resolve_python(self, specification: EnvironmentSpecification) -> Path:
        requested = specification.python.resolved_version
        if Version(requested) not in SpecifierSet(specification.python.constraint):
            raise PythonInterpreterUnavailableError("resolved Python violates declared constraint")
        command = ["uv", "python", "find", requested, *self.policy.uv_flags()]
        result = self.executor.run(
            command,
            operation="python-resolution",
            cwd=self.repository_root,
            env_remove=python_env_remove(),
            timeout=self.policy.command_timeout_seconds,
            check=False,
        )
        if result.returncode != 0:
            raise PythonInterpreterUnavailableError(result.stderr.strip() or result.stdout.strip())
        interpreter = Path(result.stdout.strip()).resolve()
        if not interpreter.exists():
            raise PythonInterpreterUnavailableError(f"uv returned missing Python: {interpreter}")
        actual = self._inspect_python(interpreter)
        if actual != requested:
            raise PythonInterpreterUnavailableError(
                f"Python interpreter mismatch: requested {requested}, got {actual}"
            )
        return interpreter

    def _inspect_python(self, python_executable: Path) -> str:
        code = """
import platform
import sys
if platform.python_implementation() != "CPython":
    raise SystemExit("only CPython is supported in Phase 01")
print(".".join(str(part) for part in sys.version_info[:3]))
""".strip()
        result = self.executor.run(
            [str(python_executable), "-c", code],
            operation="python-inspection",
            env_remove=python_env_remove(),
            timeout=self.policy.command_timeout_seconds,
            check=True,
        )
        return result.stdout.strip()

    def _sync_locked_environment(
        self,
        inputs: EnvironmentInputs,
        target: Path,
        interpreter: Path,
    ) -> None:
        env = {"UV_PROJECT_ENVIRONMENT": str(target), "VIRTUAL_ENV": str(target)}
        command = [
            "uv",
            "sync",
            "--project",
            str(inputs.project_path.parent),
            "--locked",
            "--no-default-groups",
            "--no-install-project",
            "--python",
            str(interpreter),
            *self.policy.uv_flags(),
        ]
        self.executor.run(
            command,
            operation="uv-sync",
            cwd=self.repository_root,
            env=env,
            env_remove=python_env_remove(),
            timeout=self.policy.command_timeout_seconds,
            check=True,
        )
        if (inputs.project_path.parent / ".venv").exists():
            raise EnvironmentMaterializationError(
                f"uv created forbidden committed .venv: {inputs.project_path.parent / '.venv'}"
            )

    def _create_virtual_environment(self, target: Path, interpreter: Path) -> None:
        self.executor.run(
            [
                "uv",
                "venv",
                str(target),
                "--allow-existing",
                "--python",
                str(interpreter),
                *self.policy.uv_flags(),
            ],
            operation="uv-venv",
            cwd=self.repository_root,
            env={"VIRTUAL_ENV": str(target)},
            env_remove=python_env_remove(),
            timeout=self.policy.command_timeout_seconds,
            check=True,
        )

    def _build_local_wheel(self, package_identity: str) -> tuple[Path, str]:
        path_identity = hashlib.sha256(package_identity.encode()).hexdigest()
        wheel_dir = contained_path(self.runtime_root / "source-builds/torch-dae", path_identity)
        metadata = wheel_dir / "wheel.json"
        wheel, record = self._valid_local_wheel_cache(wheel_dir, package_identity)
        if wheel is not None and record is not None:
            return wheel, record.wheel_sha256
        if wheel_dir.exists():
            replace_tree(wheel_dir)
        wheel_dir.mkdir(parents=True, exist_ok=True)
        self._build_local_wheel_with_backend(wheel_dir)
        wheel = valid_single_wheel(wheel_dir)
        if wheel is None:
            raise EnvironmentMaterializationError("local torch-dae wheel build produced no wheel")
        distribution, version = wheel_distribution_metadata(wheel)
        if canonicalize_name(distribution) != "torch-dae":
            raise EnvironmentMaterializationError("local wheel distribution is not torch-dae")
        sha = sha256_file(wheel)
        source_date_epoch = self._source_date_epoch()
        record = LocalWheelCacheRecord(
            schema_version="1.0.0",
            package_identity=package_identity,
            distribution_name=distribution,
            distribution_version=version,
            wheel_filename=wheel.name,
            wheel_sha256=sha,
            source_date_epoch=source_date_epoch,
            build_command=(
                "uv",
                "build",
                "--wheel",
                "--out-dir",
                str(wheel_dir),
                "--no-create-gitignore",
                "--no-build-isolation",
                str(self.repository_root),
            ),
            created_at=utc_now(),
        )
        write_json_atomic(metadata, record)
        return wheel, sha

    def _valid_local_wheel_cache(
        self,
        wheel_dir: Path,
        package_identity: str,
    ) -> tuple[Path | None, LocalWheelCacheRecord | None]:
        wheel = valid_single_wheel(wheel_dir)
        metadata = wheel_dir / "wheel.json"
        if wheel is None or not metadata.exists():
            return None, None
        try:
            record = LocalWheelCacheRecord.model_validate_json(metadata.read_text())
            distribution, version = wheel_distribution_metadata(wheel)
        except Exception:
            return None, None
        if (
            record.package_identity != package_identity
            or canonicalize_name(record.distribution_name) != "torch-dae"
            or canonicalize_name(distribution) != "torch-dae"
            or record.distribution_name != distribution
            or record.distribution_version != version
            or record.distribution_version != self._project_version()
            or record.wheel_filename != wheel.name
            or record.wheel_sha256 != sha256_file(wheel)
        ):
            return None, None
        return wheel, record

    def _project_version(self) -> str:
        data = tomllib.loads((self.repository_root / "pyproject.toml").read_text(encoding="utf-8"))
        version = data.get("project", {}).get("version")
        if not isinstance(version, str):
            raise EnvironmentMaterializationError("project.version is missing")
        return version

    def _build_local_wheel_with_backend(self, wheel_dir: Path) -> None:
        tmp = wheel_dir.with_name(f".{wheel_dir.name}.build")
        if tmp.exists():
            replace_tree(tmp)
        tmp.mkdir(parents=True)
        try:
            self.executor.run(
                [
                    "uv",
                    "build",
                    "--wheel",
                    "--out-dir",
                    str(tmp),
                    "--no-create-gitignore",
                    "--no-build-isolation",
                    str(self.repository_root),
                ],
                operation="local-wheel-build",
                cwd=self.repository_root,
                env={"SOURCE_DATE_EPOCH": self._source_date_epoch()},
                env_remove=python_env_remove(),
                timeout=self.policy.command_timeout_seconds,
                check=True,
            )
            wheel = valid_single_wheel(tmp)
            if wheel is None:
                raise EnvironmentMaterializationError("uv build produced no local wheel")
            shutil.move(str(wheel), str(wheel_dir / wheel.name))
        finally:
            if tmp.exists():
                replace_tree(tmp)

    def _source_date_epoch(self) -> str:
        result = self.executor.run(
            ["git", "show", "-s", "--format=%ct", "HEAD"],
            operation="source-date-epoch",
            cwd=self.repository_root,
            env_remove=python_env_remove(),
            timeout=self.policy.command_timeout_seconds,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip().isdigit():
            return result.stdout.strip()
        return "0"

    def _verify_target(
        self,
        inputs: EnvironmentInputs,
        target: Path,
    ) -> EnvironmentVerification:
        if not target.exists():
            return EnvironmentVerification(
                inputs.card_id, False, "environment is missing", "missing"
            )
        metadata_path = target / MATERIALIZATION_JSON
        marker = target / COMPLETE_MARKER
        if not metadata_path.exists() or not marker.exists():
            return EnvironmentVerification(
                inputs.card_id,
                False,
                "environment is incomplete",
                "incomplete",
            )
        try:
            record = EnvironmentMaterializationRecord.model_validate_json(metadata_path.read_text())
        except Exception as exc:
            return EnvironmentVerification(
                inputs.card_id,
                False,
                f"materialization metadata is invalid: {exc}",
                "invalid",
            )
        expected_hashes = {
            "fingerprint": inputs.fingerprint,
            "environment_spec_sha256": sha256_file(inputs.specification_path),
            "lockfile_sha256": sha256_file(inputs.lock_path),
            "source_manifest_sha256": sha256_file(inputs.sources_path),
            "local_package_identity": inputs.package_identity,
        }
        for field, expected in expected_hashes.items():
            if getattr(record, field) != expected:
                return EnvironmentVerification(inputs.card_id, False, f"{field} is stale", "stale")
        if (
            record.card_id != inputs.card_id
            or record.environment_id != inputs.specification.environment_id
        ):
            return EnvironmentVerification(
                inputs.card_id, False, "environment identity mismatch", "invalid"
            )
        python_executable = Path(record.python_executable or "")
        if not python_executable.exists():
            return EnvironmentVerification(
                inputs.card_id, False, "Python executable is missing", "invalid"
            )
        actual_python = self._inspect_python(python_executable)
        if actual_python != inputs.specification.python.resolved_version:
            return EnvironmentVerification(
                inputs.card_id, False, "Python version mismatch", "invalid"
            )
        packages = installed_distributions(python_executable, self.executor)
        local = packages.get("torch-dae")
        if local is None:
            return EnvironmentVerification(
                inputs.card_id, False, "torch-dae is not installed", "invalid"
            )
        check = self.executor.run(
            ["uv", "pip", "check", "--python", str(python_executable), *self.policy.uv_flags()],
            operation="dependency-check",
            cwd=self.repository_root,
            env_remove=python_env_remove(),
            timeout=self.policy.command_timeout_seconds,
            check=False,
        )
        if check.returncode != 0:
            return EnvironmentVerification(
                inputs.card_id, False, "dependency check failed", "invalid"
            )
        integrity_error = self._verify_installed_integrity(
            inputs,
            target,
            python_executable,
            record,
        )
        if integrity_error is not None:
            return EnvironmentVerification(inputs.card_id, False, integrity_error, "invalid")
        script = self._run_verification_script(inputs, target, python_executable)
        if script.status != "valid":
            return EnvironmentVerification(inputs.card_id, False, script.details, script.status)
        return EnvironmentVerification(inputs.card_id, True, "environment is valid", "valid")

    def _verify_installed_integrity(
        self,
        inputs: EnvironmentInputs,
        target: Path,
        python_executable: Path,
        record: EnvironmentMaterializationRecord,
    ) -> str | None:
        wheel_dir = contained_path(
            self.runtime_root / "source-builds/torch-dae",
            hashlib.sha256(inputs.package_identity.encode()).hexdigest(),
        )
        local_wheel, local_record = self._valid_local_wheel_cache(
            wheel_dir,
            inputs.package_identity,
        )
        if local_wheel is None or local_record is None:
            return "local torch-dae wheel cache metadata is missing or invalid"
        if (
            record.local_package_wheel_sha256 is None
            or sha256_file(local_wheel) != record.local_package_wheel_sha256
            or local_record.wheel_sha256 != record.local_package_wheel_sha256
        ):
            return "local torch-dae wheel cache hash mismatch"
        error = verify_installed_wheel(
            local_wheel,
            target,
            python_executable,
            self.executor,
            expected_distribution="torch-dae",
        )
        if error is not None:
            return error

        installed = {
            item.normalized_name: item.version
            for item in installed_distributions(
                python_executable,
                self.executor,
            ).values()
        }
        recorded = {item.normalized_name: item.version for item in record.installed_packages}
        if installed.get("torch-dae") != recorded.get("torch-dae"):
            return "installed torch-dae package inventory drifted"
        source_records = {item.source_id: item for item in record.installed_sources}
        for source in inputs.sources_manifest.sources:
            source_record = source_records.get(source.source_id)
            if source_record is None:
                return f"source materialization metadata is missing: {source.source_id}"
            if source.installation == SourceInstallationType.PACKAGE:
                normalized = canonicalize_name(source.package)
                if installed.get(normalized) != source.version:
                    return f"installed package source drifted: {source.package}"
                if recorded.get(normalized) != source.version:
                    return f"recorded package source drifted: {source.package}"
            elif source.installation == SourceInstallationType.GIT:
                source_context = SourceContext(
                    repository_root=self.repository_root,
                    runtime_root=self.runtime_root,
                    environment_root=target,
                    python_executable=python_executable,
                    lockfile_path=inputs.lock_path,
                    lockfile_sha256=sha256_file(inputs.lock_path),
                    python_version=record.python_actual_version
                    or inputs.specification.python.resolved_version,
                    platform=inputs.platform,
                    local_package_wheel_path=local_wheel,
                    local_package_wheel_sha256=record.local_package_wheel_sha256,
                    local_package_identity=inputs.package_identity,
                )
                wheel_dir = contained_path(
                    self.runtime_root / "source-builds",
                    source.source_id,
                    SourceManager()._git_build_fingerprint(source, source_context),
                )
                source_manager = SourceManager(executor=self.executor, policy=self.policy)
                wheel = source_manager._valid_cached_git_wheel(
                    source,
                    source_context,
                    wheel_dir,
                    wheel_dir / "source-wheel.json",
                    source_manager._git_build_fingerprint(source, source_context),
                )
                if wheel is None:
                    return (
                        f"Git source wheel cache metadata is missing or invalid: {source.source_id}"
                    )
                if source_record.wheel_sha256 != sha256_file(wheel):
                    return f"Git source wheel hash mismatch: {source.source_id}"
                git_error = verify_installed_wheel(wheel, target, python_executable, self.executor)
                if git_error is not None:
                    return f"{source.source_id}: {git_error}"
                try:
                    SourceManager(executor=self.executor, policy=self.policy)._verify_git_checkout(
                        source,
                        Path(source_record.location),
                    )
                except Exception as exc:
                    return f"Git source checkout drifted: {source.source_id}: {exc}"
            elif source.installation == SourceInstallationType.VENDORED:
                distribution_info = installed_distribution_info(
                    python_executable,
                    self.executor,
                    "torch-dae",
                )
                for relative, expected_hash in source_record.file_hashes.items():
                    path = contained_path(self.repository_root, relative)
                    if not path.is_file() or sha256_file(path) != expected_hash:
                        return f"vendored repository file drifted: {relative}"
                for relative, member in source_record.wheel_members.items():
                    installed_file = contained_path(distribution_info.root, member)
                    if not installed_file.is_file():
                        return f"vendored installed file is missing: {relative}"
                    repository_file = contained_path(self.repository_root, relative)
                    if installed_file.read_bytes() != repository_file.read_bytes():
                        return f"vendored installed file drifted: {relative}"
        return None

    def _run_verification_script(
        self,
        inputs: EnvironmentInputs,
        target: Path,
        python_executable: Path,
    ) -> VerificationExecutionRecord:
        result = self._execute_verification_script(inputs, target, python_executable)
        return verification_from_result(result)

    def _execute_verification_script(
        self,
        inputs: EnvironmentInputs,
        target: Path,
        python_executable: Path,
    ) -> ManagedProcessResult:
        return self.executor.run(
            [str(python_executable), str(inputs.verification_script)],
            operation="verification-script",
            cwd=self.repository_root,
            env={
                "TORCH_DAE_ENVIRONMENT_ROOT": str(target),
                "TORCH_DAE_CARD_ID": inputs.card_id,
                "TORCH_DAE_ENVIRONMENT_ID": inputs.specification.environment_id,
                "TORCH_DAE_ENVIRONMENT_FINGERPRINT": inputs.fingerprint,
            },
            env_remove=python_env_remove(),
            timeout=self.policy.command_timeout_seconds,
            check=False,
        )

    def _resolved_from_record(self, inputs: EnvironmentInputs, target: Path) -> ResolvedEnvironment:
        metadata_path = target / MATERIALIZATION_JSON
        if not metadata_path.exists():
            raise EnvironmentNotFoundError(f"missing materialization metadata: {metadata_path}")
        record = EnvironmentMaterializationRecord.model_validate_json(metadata_path.read_text())
        if record.python_executable is None or record.python_actual_version is None:
            raise EnvironmentNotFoundError("materialization metadata is incomplete")
        installed_sources = tuple(
            InstalledSource(item.source_id, item.location, item.revision)
            for item in record.installed_sources
        )
        return ResolvedEnvironment(
            environment_id=record.environment_id,
            model_card_id=record.card_id,
            root=target,
            python_executable=Path(record.python_executable),
            fingerprint=record.fingerprint,
            python_version=record.python_actual_version,
            platform=record.platform,
            installed_packages={
                item.normalized_name: item.version for item in record.installed_packages
            },
            installed_sources=installed_sources,
            valid=True,
        )


def environment_python(environment_root: Path) -> Path:
    """Return the Python executable path for POSIX or Windows venv layouts."""

    posix = environment_root / "bin" / "python"
    if posix.exists():
        return posix
    windows = environment_root / "Scripts" / "python.exe"
    if windows.exists():
        return windows
    raise EnvironmentMaterializationError(
        f"could not locate environment Python in {environment_root}"
    )


def verification_from_result(result: ManagedProcessResult) -> VerificationExecutionRecord:
    """Convert a verification process result into runtime metadata."""

    if result.returncode != 0:
        return result_to_verification("invalid", "verification script failed", result)
    return result_to_verification("valid", "verification script passed", result)


@dataclass(frozen=True)
class DistributionInfo:
    """Installed distribution location details from the target environment."""

    name: str
    version: str
    root: Path
    scripts: Path
    console_scripts: tuple[str, ...]


def installed_distribution_info(
    python_executable: Path,
    executor: CommandExecutor,
    distribution: str,
) -> DistributionInfo:
    """Inspect one distribution from the target Python environment."""

    code = """
import importlib.metadata as metadata
import json
import sys
import sysconfig
from pathlib import Path
distribution = sys.argv[1]
dist = metadata.distribution(distribution)
rows = {
    "name": dist.metadata.get("Name") or distribution,
    "version": dist.version,
    "root": str(Path(str(dist.locate_file(""))).resolve()),
    "scripts": str(Path(sysconfig.get_path("scripts")).resolve()),
    "console_scripts": sorted(
        entry.name for entry in dist.entry_points if entry.group == "console_scripts"
    ),
}
print(json.dumps(rows, sort_keys=True))
""".strip()
    result = executor.run(
        [str(python_executable), "-c", code, distribution],
        operation="distribution-inspection",
        env_remove=python_env_remove(),
        check=True,
    )
    data = json.loads(result.stdout)
    return DistributionInfo(
        name=str(data["name"]),
        version=str(data["version"]),
        root=Path(str(data["root"])),
        scripts=Path(str(data["scripts"])),
        console_scripts=tuple(str(item) for item in data["console_scripts"]),
    )


def verify_installed_wheel(
    wheel: Path,
    environment_root: Path,
    python_executable: Path,
    executor: CommandExecutor,
    *,
    expected_distribution: str | None = None,
) -> str | None:
    """Verify installed files for a wheel-managed distribution."""

    distribution, version = wheel_distribution_metadata(wheel)
    if expected_distribution is not None and canonicalize_name(distribution) != canonicalize_name(
        expected_distribution
    ):
        return f"wheel distribution mismatch: {distribution}"
    try:
        info = installed_distribution_info(python_executable, executor, distribution)
    except Exception as exc:
        return f"installed distribution is missing: {distribution}: {exc}"
    if canonicalize_name(info.name) != canonicalize_name(distribution) or info.version != version:
        return f"installed distribution metadata drifted: {distribution}"
    try:
        with zipfile.ZipFile(wheel) as archive:
            names = [name for name in archive.namelist() if not name.endswith("/")]
            record_name = next(name for name in names if name.endswith(".dist-info/RECORD"))
            record_members = {
                line.split(",", 1)[0] for line in archive.read(record_name).decode().splitlines()
            }
            for name in names:
                if name not in record_members:
                    return f"wheel RECORD is missing member: {name}"
                if name == record_name or ".data/" in name:
                    continue
                installed = contained_path(info.root, name)
                if not installed.is_file():
                    return f"installed wheel file is missing: {name}"
                if installed.read_bytes() != archive.read(name):
                    return f"installed wheel file drifted: {name}"
    except Exception as exc:
        return f"wheel integrity inspection failed: {exc}"
    if "torch-dae" in info.console_scripts and not (info.scripts / "torch-dae").is_file():
        return "console script is missing: torch-dae"
    if canonicalize_name(distribution) == "torch-dae":
        import_check = executor.run(
            [
                str(python_executable),
                "-c",
                (
                    "import sys; "
                    "import importlib.metadata as m; "
                    "import torch_dae, torch_dae.cards.models, torch_dae.environment; "
                    "assert m.version('torch-dae') == sys.argv[1]"
                ),
                version,
            ],
            operation="local-wheel-import-check",
            env={"VIRTUAL_ENV": str(environment_root)},
            env_remove=python_env_remove(),
            check=False,
        )
        if import_check.returncode != 0:
            return "local torch-dae import check failed"
    return None


def python_env_remove() -> tuple[str, str]:
    """Environment variables removed for model-environment Python commands."""

    return ("PYTHONPATH", "PYTHONHOME")
