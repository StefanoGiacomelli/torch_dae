"""Deterministic environment fingerprints."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from torch_dae.contracts import canonical_json_bytes
from torch_dae.environment.specification import EnvironmentSourcesManifest, EnvironmentSpecification


@dataclass(frozen=True)
class FingerprintInputs:
    """Inputs that define a materialized model environment."""

    specification: EnvironmentSpecification
    lockfile_bytes: bytes
    sources_manifest: EnvironmentSourcesManifest
    target_platform: str
    local_package_identity: str

    def canonical_bytes(self) -> bytes:
        """Serialize inputs deterministically for hashing."""

        payload: Mapping[str, object] = {
            "specification": self.specification.model_dump(mode="json", by_alias=True),
            "lockfile_sha256": hashlib.sha256(self.lockfile_bytes).hexdigest(),
            "sources_manifest": self.sources_manifest.model_dump(mode="json", by_alias=True),
            "resolved_python_version": self.specification.python.resolved_version,
            "target_platform": self.target_platform,
            "local_package_identity": self.local_package_identity,
        }
        return canonical_json_bytes(payload)


def calculate_environment_fingerprint(inputs: FingerprintInputs) -> str:
    """Return a deterministic SHA-256 fingerprint for an environment."""

    return hashlib.sha256(inputs.canonical_bytes()).hexdigest()


def canonical_platform_tag(system: str | None = None, machine: str | None = None) -> str:
    """Return a stable operating-system/architecture tag."""

    raw_system = (system or platform.system()).lower()
    raw_machine = (machine or platform.machine()).lower()
    os_name = {"darwin": "macos"}.get(raw_system, raw_system)
    arch = {"amd64": "x86_64", "x64": "x86_64", "aarch64": "arm64"}.get(raw_machine, raw_machine)
    return f"{os_name}-{arch}"


def local_package_identity(repository_root: Path) -> str:
    """Return a deterministic local package identity before or after first commit."""

    head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", "pyproject.toml", "src/torch_dae"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if head.returncode == 0 and status.returncode == 0 and not status.stdout.strip():
        return f"git:{head.stdout.strip()}"
    digest = hashlib.sha256()
    package_paths = [
        repository_root / "pyproject.toml",
        *sorted((repository_root / "src/torch_dae").glob("**/*.py")),
    ]
    for path in package_paths:
        digest.update(str(path.relative_to(repository_root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"content:{digest.hexdigest()}"
