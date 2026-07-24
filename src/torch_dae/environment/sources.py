"""Source materialization for package, pinned Git, and vendored strategies."""

from __future__ import annotations

import hashlib
import json
import shutil
import tomllib
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path

from packaging.utils import canonicalize_name

from torch_dae.contracts import canonical_json_bytes, contained_path
from torch_dae.core.errors import OfflineResourceUnavailableError, SourceMaterializationError
from torch_dae.environment.policy import ExecutionPolicy
from torch_dae.environment.runtime import (
    GitSourceWheelCacheRecord,
    InstalledPackageRecord,
    InstalledSourceRecord,
    utc_now,
    write_json_atomic,
)
from torch_dae.environment.specification import (
    EnvironmentSourcesManifest,
    OfficialPackageSource,
    PinnedGitSource,
    SourceInstallationType,
    VendoredAdaptationSource,
)
from torch_dae.environment.subprocess import CommandExecutor


@dataclass(frozen=True)
class SourceContext:
    """Context required to verify or install environment sources."""

    repository_root: Path
    runtime_root: Path
    environment_root: Path
    python_executable: Path
    lockfile_path: Path
    lockfile_sha256: str
    python_version: str
    platform: str
    local_package_wheel_path: Path
    local_package_wheel_sha256: str
    local_package_identity: str


class SourceManager:
    """Materialize upstream source strategies after locked environment sync."""

    def __init__(
        self,
        *,
        executor: CommandExecutor | None = None,
        policy: ExecutionPolicy | None = None,
    ) -> None:
        self.executor = executor or CommandExecutor()
        self.policy = policy or ExecutionPolicy()

    def materialize(
        self,
        manifest: EnvironmentSourcesManifest,
        context: SourceContext,
    ) -> tuple[InstalledSourceRecord, ...]:
        """Verify and install all sources declared in `manifest`."""

        records: list[InstalledSourceRecord] = []
        for source in manifest.sources:
            if source.installation == SourceInstallationType.PACKAGE:
                records.append(self._verify_package(source, context))
            elif source.installation == SourceInstallationType.GIT:
                records.append(self._install_git(source, context))
            elif source.installation == SourceInstallationType.VENDORED:
                records.append(self._verify_vendored(source, context))
        return tuple(records)

    def _verify_package(
        self,
        source: OfficialPackageSource,
        context: SourceContext,
    ) -> InstalledSourceRecord:
        packages = lock_packages(context.lockfile_path)
        normalized = canonicalize_name(source.package)
        if packages.get(normalized) != source.version:
            raise SourceMaterializationError(
                f"lock file does not contain {source.package}=={source.version}"
            )
        installed = installed_distributions(context.python_executable, self.executor)
        record = installed.get(normalized)
        if record is None or record.version != source.version:
            raise SourceMaterializationError(
                f"environment does not contain {source.package}=={source.version}"
            )
        return InstalledSourceRecord(
            source_id=source.source_id,
            installation=source.installation,
            location=record.location,
            version=record.version,
        )

    def _install_git(
        self,
        source: PinnedGitSource,
        context: SourceContext,
    ) -> InstalledSourceRecord:
        checkout = contained_path(
            context.runtime_root / "repositories",
            source.source_id,
            source.revision,
        )
        checkout = self._ensure_git_checkout(source, checkout, context)

        wheel_dir = contained_path(
            context.runtime_root / "source-builds",
            source.source_id,
            self._git_build_fingerprint(source, context),
        )
        metadata = wheel_dir / "source-wheel.json"
        build_fingerprint = self._git_build_fingerprint(source, context)
        wheel = self._valid_cached_git_wheel(
            source,
            context,
            wheel_dir,
            metadata,
            build_fingerprint,
        )
        if wheel is None:
            if self.policy.offline:
                raise OfflineResourceUnavailableError(
                    f"cached Git source wheel unavailable for {source.source_id}"
                )
            if wheel_dir.exists():
                replace_tree(wheel_dir)
            wheel_dir.mkdir(parents=True, exist_ok=True)
            workspace = wheel_dir / "workspace"
            if workspace.exists():
                replace_tree(workspace)
            try:
                export_revision(checkout, source.revision, workspace, self.executor, self.policy)
                self.executor.run(
                    [
                        "uv",
                        "build",
                        "--wheel",
                        "--out-dir",
                        str(wheel_dir),
                        "--no-create-gitignore",
                        "--python",
                        str(context.python_executable),
                        "--no-build-isolation",
                        *self.policy.uv_flags(),
                        str(workspace),
                    ],
                    operation="git-wheel-build",
                    cwd=context.repository_root,
                    env_remove=python_env_remove(),
                    timeout=self.policy.command_timeout_seconds,
                    check=True,
                )
                self._verify_git_checkout(source, checkout)
                wheel = valid_single_wheel(wheel_dir)
                if wheel is None:
                    raise SourceMaterializationError(f"no wheel built for {source.source_id}")
                distribution, version = wheel_distribution_metadata(wheel)
                wheel_sha = sha256_file(wheel)
                write_json_atomic(
                    metadata,
                    GitSourceWheelCacheRecord(
                        schema_version="1.0.0",
                        source_id=source.source_id,
                        source_url=source.url,
                        source_revision=source.revision,
                        build_fingerprint=build_fingerprint,
                        python_version=context.python_version,
                        platform=context.platform,
                        lockfile_sha256=context.lockfile_sha256,
                        distribution_name=distribution,
                        distribution_version=version,
                        wheel_filename=wheel.name,
                        wheel_sha256=wheel_sha,
                        created_at=utc_now(),
                    ),
                )
            finally:
                if workspace.exists():
                    replace_tree(workspace)
        wheel_sha = sha256_file(wheel)
        self.executor.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(context.python_executable),
                "--no-deps",
                *self.policy.uv_flags(),
                str(wheel),
            ],
            operation="git-wheel-install",
            cwd=context.repository_root,
            env_remove=python_env_remove(),
            timeout=self.policy.command_timeout_seconds,
            check=True,
        )
        distribution, version = wheel_distribution_metadata(wheel)
        installed = installed_distributions(context.python_executable, self.executor)
        installed_record = installed.get(canonicalize_name(distribution))
        if installed_record is None or installed_record.version != version:
            raise SourceMaterializationError(
                f"Git wheel distribution not installed exactly: {distribution}=={version}"
            )
        return InstalledSourceRecord(
            source_id=source.source_id,
            installation=source.installation,
            location=str(checkout),
            version=version,
            revision=source.revision,
            wheel_sha256=wheel_sha,
        )

    def _ensure_git_checkout(
        self,
        source: PinnedGitSource,
        checkout: Path,
        context: SourceContext,
    ) -> Path:
        if checkout.exists():
            try:
                self._verify_git_checkout(source, checkout)
                return checkout
            except SourceMaterializationError as exc:
                if self.policy.offline:
                    raise OfflineResourceUnavailableError(
                        f"cached Git checkout invalid for {source.source_id}"
                    ) from exc
                replace_tree(checkout)
        elif self.policy.offline:
            raise OfflineResourceUnavailableError(
                f"cached Git checkout unavailable for {source.source_id}"
            )

        checkout.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkout.with_name(f"{checkout.name}.clone-{uuid.uuid4().hex}")
        try:
            self.executor.run(
                ["git", "clone", source.url, str(temporary)],
                operation="git-clone",
                cwd=context.repository_root,
                env_remove=python_env_remove(),
                timeout=self.policy.command_timeout_seconds,
                check=True,
            )
            self.executor.run(
                ["git", "checkout", "--detach", source.revision],
                operation="git-checkout",
                cwd=temporary,
                env_remove=python_env_remove(),
                timeout=self.policy.command_timeout_seconds,
                check=True,
            )
            self._verify_git_checkout(source, temporary)
            if checkout.exists():
                try:
                    self._verify_git_checkout(source, checkout)
                    return checkout
                finally:
                    replace_tree(temporary)
            temporary.rename(checkout)
            return checkout
        except Exception:
            if temporary.exists():
                replace_tree(temporary)
            if checkout.exists():
                self._verify_git_checkout(source, checkout)
                return checkout
            raise

    def _valid_cached_git_wheel(
        self,
        source: PinnedGitSource,
        context: SourceContext,
        wheel_dir: Path,
        metadata: Path,
        build_fingerprint: str,
    ) -> Path | None:
        wheel = valid_single_wheel(wheel_dir)
        if wheel is None or not metadata.exists():
            return None
        try:
            record = GitSourceWheelCacheRecord.model_validate_json(metadata.read_text())
        except Exception:
            return None
        if (
            record.source_id != source.source_id
            or record.source_url != source.url
            or record.source_revision != source.revision
            or record.build_fingerprint != build_fingerprint
            or record.python_version != context.python_version
            or record.platform != context.platform
            or record.lockfile_sha256 != context.lockfile_sha256
            or record.wheel_filename != wheel.name
            or record.wheel_sha256 != sha256_file(wheel)
        ):
            return None
        try:
            distribution, version = wheel_distribution_metadata(wheel)
        except Exception:
            return None
        if record.distribution_name != distribution or record.distribution_version != version:
            return None
        return wheel

    def _verify_git_checkout(self, source: PinnedGitSource, checkout: Path) -> None:
        head = self.executor.run(
            ["git", "rev-parse", "HEAD"],
            operation="git-revision-check",
            cwd=checkout,
            env_remove=python_env_remove(),
            timeout=self.policy.command_timeout_seconds,
            check=True,
        ).stdout.strip()
        if head != source.revision:
            raise SourceMaterializationError(
                f"cached checkout HEAD mismatch for {source.source_id}"
            )
        remote = self.executor.run(
            ["git", "remote", "get-url", "origin"],
            operation="git-remote-check",
            cwd=checkout,
            env_remove=python_env_remove(),
            timeout=self.policy.command_timeout_seconds,
            check=True,
        ).stdout.strip()
        if remote != source.url:
            raise SourceMaterializationError(
                f"cached checkout remote mismatch for {source.source_id}"
            )
        status = self.executor.run(
            ["git", "status", "--porcelain"],
            operation="git-cleanliness-check",
            cwd=checkout,
            env_remove=python_env_remove(),
            timeout=self.policy.command_timeout_seconds,
            check=True,
        ).stdout.strip()
        if status:
            raise SourceMaterializationError(f"cached checkout is dirty for {source.source_id}")

    def _git_build_fingerprint(self, source: PinnedGitSource, context: SourceContext) -> str:
        payload = {
            "url": source.url,
            "revision": source.revision,
            "lockfile_sha256": context.lockfile_sha256,
            "python_version": context.python_version,
            "platform": context.platform,
            "build": source.build,
        }
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    def _verify_vendored(
        self,
        source: VendoredAdaptationSource,
        context: SourceContext,
    ) -> InstalledSourceRecord:
        if not source.adaptation_description.strip() or not source.justification.strip():
            raise SourceMaterializationError(f"vendored source notes are empty: {source.source_id}")
        file_hashes: dict[str, str] = {}
        if context.local_package_wheel_sha256 != sha256_file(context.local_package_wheel_path):
            raise SourceMaterializationError("local torch-dae wheel hash mismatch")
        wheel_members: dict[str, str] = {}
        with zipfile.ZipFile(context.local_package_wheel_path) as archive:
            wheel_names = set(archive.namelist())
            member_bytes = {name: archive.read(name) for name in wheel_names}
        for relative in source.copied_files:
            path = contained_path(context.repository_root, relative)
            if not path.is_file():
                raise SourceMaterializationError(f"vendored file is missing: {relative}")
            file_hashes[relative] = sha256_file(path)
            member = repository_path_to_wheel_member(relative)
            if member not in wheel_names:
                raise SourceMaterializationError(
                    f"vendored file absent from local wheel: {relative}"
                )
            if member_bytes[member] != path.read_bytes():
                raise SourceMaterializationError(
                    f"vendored wheel member differs from repository file: {relative}"
                )
            wheel_members[relative] = member
        return InstalledSourceRecord(
            source_id=source.source_id,
            installation=source.installation,
            location=str(context.repository_root),
            revision=source.upstream_revision,
            file_hashes=file_hashes,
            wheel_members=wheel_members,
        )


def lock_packages(lockfile_path: Path) -> dict[str, str]:
    """Return normalized package versions represented in a uv lock file."""

    data = tomllib.loads(lockfile_path.read_text(encoding="utf-8"))
    packages: dict[str, str] = {}
    for package in data.get("package", []):
        if isinstance(package, dict):
            name = package.get("name")
            version = package.get("version")
            if isinstance(name, str) and isinstance(version, str):
                packages[canonicalize_name(name)] = version
    return packages


def installed_distributions(
    python_executable: Path,
    executor: CommandExecutor,
) -> dict[str, InstalledPackageRecord]:
    """Collect installed distributions by running importlib.metadata in the target Python."""

    code = """
import importlib.metadata as metadata
import json
from pathlib import Path
rows = []
for dist in metadata.distributions():
    name = dist.metadata.get("Name") or dist.metadata.get("Summary") or "unknown"
    rows.append({
        "name": name,
        "version": dist.version,
        "location": str(Path(str(dist.locate_file(""))).resolve()),
    })
print(json.dumps(rows, sort_keys=True))
""".strip()
    result = executor.run(
        [str(python_executable), "-c", code],
        operation="installed-distributions",
        env_remove=python_env_remove(),
        check=True,
    )
    rows = json.loads(result.stdout)
    records: dict[str, InstalledPackageRecord] = {}
    for row in rows:
        name = str(row["name"])
        records[canonicalize_name(name)] = InstalledPackageRecord(
            name=name,
            normalized_name=canonicalize_name(name),
            version=str(row["version"]),
            location=str(row["location"]),
        )
    return records


def sha256_file(path: Path) -> str:
    """Hash a file by streaming bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_single_wheel(directory: Path) -> Path | None:
    """Return the only wheel in `directory`, if exactly one exists."""

    wheels = sorted(directory.glob("*.whl")) if directory.exists() else []
    if len(wheels) == 1:
        return wheels[0]
    return None


def wheel_distribution_metadata(wheel: Path) -> tuple[str, str]:
    """Read distribution name and version from wheel METADATA."""

    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = archive.read(metadata_name).decode("utf-8", errors="replace")
    name = ""
    version = ""
    for line in metadata.splitlines():
        if line.startswith("Name: "):
            name = line.removeprefix("Name: ").strip()
        elif line.startswith("Version: "):
            version = line.removeprefix("Version: ").strip()
    if not name or not version:
        raise SourceMaterializationError(f"wheel metadata missing name/version: {wheel}")
    return name, version


def replace_tree(path: Path) -> None:
    """Remove a directory tree or file for deterministic rebuilds."""

    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def repository_path_to_wheel_member(relative: str) -> str:
    """Map a repository path to the expected wheel member path."""

    prefix = "src/"
    if relative.startswith(prefix):
        return relative.removeprefix(prefix)
    return relative


def export_revision(
    checkout: Path,
    revision: str,
    workspace: Path,
    executor: CommandExecutor,
    policy: ExecutionPolicy,
) -> None:
    """Export an exact Git revision into a disposable build workspace."""

    workspace.parent.mkdir(parents=True, exist_ok=True)
    archive_path = workspace.parent / "workspace.tar"
    if archive_path.exists():
        archive_path.unlink()
    try:
        executor.run(
            ["git", "archive", "--format=tar", "--output", str(archive_path), revision],
            operation="git-archive",
            cwd=checkout,
            env_remove=python_env_remove(),
            timeout=policy.command_timeout_seconds,
            check=True,
        )
        workspace.mkdir()
        executor.run(
            ["tar", "-xf", str(archive_path)],
            operation="git-archive-extract",
            cwd=workspace,
            env_remove=python_env_remove(),
            timeout=policy.command_timeout_seconds,
            check=True,
        )
    except Exception:
        if workspace.exists():
            replace_tree(workspace)
        raise
    finally:
        if archive_path.exists():
            archive_path.unlink()


def python_env_remove() -> tuple[str, str]:
    """Environment variables removed for model-environment Python commands."""

    return ("PYTHONPATH", "PYTHONHOME")
