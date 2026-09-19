"""Offline local-wheel dependency-closure validation for accepted environment locks."""

from __future__ import annotations

import hashlib
import tomllib
import zipfile
from collections import defaultdict, deque
from collections.abc import Mapping
from email.parser import BytesParser
from pathlib import Path
from typing import Any

from packaging.markers import Marker, default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from torch_dae.core.errors import EnvironmentMaterializationError
from torch_dae.environment.results import (
    EnvironmentDependencyClosureResult,
    EnvironmentFailureClassification,
    RuntimeRequirementEvidence,
)


def validate_wheel_dependency_closure(
    *,
    environment_id: str,
    wheel_path: Path,
    project_path: Path,
    lock_path: Path,
    python_version: str,
    platform: str,
    result_path: str,
) -> EnvironmentDependencyClosureResult:
    """Validate active wheel requirements against the installable accepted lock closure.

    The operation reads only the local wheel, environment project, and accepted lock. It performs
    no package resolution, environment creation, import, model-card lookup, or network access.
    """

    distribution, distribution_version, requirements = _wheel_runtime_requirements(wheel_path)
    marker_environment = marker_environment_for(python_version, platform)
    active = tuple(
        sorted(
            (
                requirement
                for requirement in requirements
                if requirement.marker is None
                or requirement.marker.evaluate(marker_environment, context="metadata")
            ),
            key=lambda item: (canonicalize_name(item.name), str(item)),
        )
    )
    reachable = reachable_lock_versions(
        project_path=project_path,
        lock_path=lock_path,
        marker_environment=marker_environment,
    )
    satisfied: list[RuntimeRequirementEvidence] = []
    missing: list[RuntimeRequirementEvidence] = []
    incompatible: list[RuntimeRequirementEvidence] = []
    for requirement in active:
        normalized_name = canonicalize_name(requirement.name)
        versions = tuple(sorted(reachable.get(normalized_name, ()), key=Version))
        evidence = RuntimeRequirementEvidence(
            requirement=str(requirement),
            normalized_name=normalized_name,
            specifier=str(requirement.specifier) or None,
            marker=str(requirement.marker) if requirement.marker is not None else None,
            reachable_versions=versions,
        )
        if not versions:
            missing.append(evidence)
        elif requirement.specifier and not all(
            Version(version) in requirement.specifier for version in versions
        ):
            incompatible.append(evidence)
        else:
            satisfied.append(evidence)
    failure = None
    if missing or incompatible:
        failure = EnvironmentFailureClassification.DEPENDENCY_CLOSURE
    return EnvironmentDependencyClosureResult(
        schema_version="1.0.0",
        environment_id=environment_id,
        package_wheel_identity=f"{canonicalize_name(distribution)}=={distribution_version}",
        package_wheel_sha256=_sha256_file(wheel_path),
        python_version=python_version,
        platform=platform,
        installation_policy="locked-project-closure-then-local-wheel-no-deps",
        active_runtime_requirements=tuple(str(item) for item in active),
        satisfied_requirements=tuple(satisfied),
        missing_requirements=tuple(missing),
        incompatible_requirements=tuple(incompatible),
        lockfile_sha256=_sha256_file(lock_path),
        status="passed" if failure is None else "failed",
        failure_classification=failure,
        result_path=result_path,
    )


def marker_environment_for(python_version: str, platform: str) -> dict[str, str]:
    """Return deterministic PEP 508 marker values for a selected interpreter and platform."""

    release = Version(python_version).release
    if len(release) < 2:
        raise EnvironmentMaterializationError(f"invalid selected Python version: {python_version}")
    system, separator, architecture = platform.partition("-")
    if not separator or not architecture:
        raise EnvironmentMaterializationError(f"invalid canonical platform: {platform}")
    system_values = {
        "macos": ("posix", "darwin", "Darwin"),
        "linux": ("posix", "linux", "Linux"),
        "windows": ("nt", "win32", "Windows"),
    }
    try:
        os_name, sys_platform, platform_system = system_values[system]
    except KeyError as exc:
        raise EnvironmentMaterializationError(
            f"unsupported canonical platform for marker evaluation: {platform}"
        ) from exc
    platform_machine = architecture
    if architecture == "arm64" and system == "linux":
        platform_machine = "aarch64"
    environment = {key: str(value) for key, value in default_environment().items()}
    environment.update(
        {
            "implementation_name": "cpython",
            "implementation_version": python_version,
            "os_name": os_name,
            "platform_machine": platform_machine,
            "platform_python_implementation": "CPython",
            "platform_system": platform_system,
            "python_full_version": python_version,
            "python_version": ".".join(str(item) for item in release[:2]),
            "sys_platform": sys_platform,
            "extra": "",
        }
    )
    return environment


def reachable_lock_versions(
    *,
    project_path: Path,
    lock_path: Path,
    marker_environment: Mapping[str, str],
) -> dict[str, tuple[str, ...]]:
    """Return versions reachable by canonical locked-project synchronization policy."""

    try:
        project = tomllib.loads(project_path.read_text(encoding="utf-8"))
        lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise EnvironmentMaterializationError(f"invalid project or lock file: {exc}") from exc
    project_name = project.get("project", {}).get("name")
    if not isinstance(project_name, str):
        raise EnvironmentMaterializationError("environment project name is missing")
    raw_packages = lock.get("package", ())
    if not isinstance(raw_packages, list):
        raise EnvironmentMaterializationError("uv lock package collection is invalid")
    packages = [item for item in raw_packages if isinstance(item, dict)]
    root_candidates = [
        index
        for index, package in enumerate(packages)
        if canonicalize_name(str(package.get("name", ""))) == canonicalize_name(project_name)
        and isinstance(package.get("source"), dict)
        and (package["source"].get("virtual") == "." or package["source"].get("editable") == ".")
    ]
    if len(root_candidates) != 1:
        raise EnvironmentMaterializationError(
            "accepted lock must contain exactly one local environment project"
        )
    by_name: dict[str, list[int]] = defaultdict(list)
    for index, package in enumerate(packages):
        name = package.get("name")
        if isinstance(name, str):
            by_name[canonicalize_name(name)].append(index)
    requested_extras: dict[int, set[str]] = defaultdict(set)
    processed_extras: dict[int, frozenset[str]] = {}
    queue: deque[int] = deque([root_candidates[0]])
    reachable: dict[str, set[str]] = defaultdict(set)
    while queue:
        index = queue.popleft()
        package = packages[index]
        extras = frozenset(requested_extras[index])
        if processed_extras.get(index) == extras:
            continue
        processed_extras[index] = extras
        if index != root_candidates[0]:
            name = package.get("name")
            version = package.get("version")
            if isinstance(name, str) and isinstance(version, str):
                try:
                    Version(version)
                except InvalidVersion as exc:
                    raise EnvironmentMaterializationError(
                        f"invalid locked version for {name}: {version}"
                    ) from exc
                reachable[canonicalize_name(name)].add(version)
        edges = list(_active_edges(package.get("dependencies"), marker_environment))
        optional = package.get("optional-dependencies")
        if isinstance(optional, dict):
            for extra in sorted(extras):
                edges.extend(_active_edges(optional.get(extra), marker_environment))
        for edge in edges:
            name = edge.get("name")
            if not isinstance(name, str):
                raise EnvironmentMaterializationError("locked dependency edge lacks a name")
            candidates: tuple[int, ...] = tuple(by_name.get(canonicalize_name(name), ()))
            edge_version = edge.get("version")
            if isinstance(edge_version, str):
                candidates = tuple(
                    candidate
                    for candidate in candidates
                    if packages[candidate].get("version") == edge_version
                )
            if not candidates:
                raise EnvironmentMaterializationError(
                    f"accepted lock dependency edge cannot resolve package: {name}"
                )
            raw_extras = edge.get("extra", ())
            edge_extras = (
                {item for item in raw_extras if isinstance(item, str)}
                if isinstance(raw_extras, list)
                else set()
            )
            for candidate in candidates:
                before = frozenset(requested_extras[candidate])
                requested_extras[candidate].update(edge_extras)
                if candidate not in processed_extras or before != requested_extras[candidate]:
                    queue.append(candidate)
    return {
        name: tuple(sorted(versions, key=Version)) for name, versions in sorted(reachable.items())
    }


def _active_edges(
    raw_edges: object,
    marker_environment: Mapping[str, str],
) -> list[dict[str, Any]]:
    if raw_edges is None:
        return []
    if not isinstance(raw_edges, list):
        raise EnvironmentMaterializationError("locked dependency collection is invalid")
    active: list[dict[str, Any]] = []
    for edge in raw_edges:
        if not isinstance(edge, dict):
            raise EnvironmentMaterializationError("locked dependency edge is invalid")
        marker = edge.get("marker")
        if isinstance(marker, str) and not Marker(marker).evaluate(
            environment=dict(marker_environment), context="lock_file"
        ):
            continue
        active.append(edge)
    return active


def _wheel_runtime_requirements(wheel_path: Path) -> tuple[str, str, tuple[Requirement, ...]]:
    try:
        with zipfile.ZipFile(wheel_path) as archive:
            names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            if len(names) != 1:
                raise EnvironmentMaterializationError(
                    "local package wheel must contain exactly one METADATA file"
                )
            metadata = BytesParser().parsebytes(archive.read(names[0]))
    except (OSError, zipfile.BadZipFile) as exc:
        raise EnvironmentMaterializationError(f"invalid local package wheel: {wheel_path}") from exc
    distribution = metadata.get("Name")
    version = metadata.get("Version")
    if not distribution or not version:
        raise EnvironmentMaterializationError("local package wheel metadata lacks name or version")
    requirements: list[Requirement] = []
    for raw in metadata.get_all("Requires-Dist", ()):
        try:
            requirements.append(Requirement(raw))
        except InvalidRequirement as exc:
            raise EnvironmentMaterializationError(
                f"invalid Requires-Dist in local package wheel: {raw}"
            ) from exc
    return distribution, version, tuple(requirements)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
