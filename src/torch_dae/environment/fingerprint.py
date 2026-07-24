"""Deterministic environment fingerprints."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import tomllib
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
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    content_digest = local_package_content_digest(repository_root)
    if head.returncode == 0 and status.returncode == 0 and not status.stdout.strip():
        return f"git:{head.stdout.strip()}:content:{content_digest}"
    return f"content:{content_digest}"


def local_package_content_digest(repository_root: Path) -> str:
    """Hash every repository input that can affect the built local wheel."""

    digest = hashlib.sha256()
    for path in local_package_build_inputs(repository_root):
        relative = path.relative_to(repository_root).as_posix()
        data = path.read_bytes()
        encoded_path = relative.encode("utf-8")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def local_package_build_inputs(repository_root: Path) -> tuple[Path, ...]:
    """Return sorted regular files that feed the backend-built local wheel."""

    paths: set[Path] = set()
    pyproject = repository_root / "pyproject.toml"
    if pyproject.is_file():
        paths.add(pyproject)
        paths.update(_project_readme_paths(repository_root, pyproject))
    src_root = repository_root / "src" / "torch_dae"
    if src_root.exists():
        for root, dirs, files in os.walk(src_root):
            root_path = Path(root)
            dirs[:] = sorted(
                item
                for item in dirs
                if item not in _EXCLUDED_DIRECTORIES and not item.startswith(".")
            )
            for filename in sorted(files):
                if _excluded_file(filename):
                    continue
                path = root_path / filename
                if path.is_file():
                    paths.add(path)
    for name in (
        "setup.py",
        "setup.cfg",
        "MANIFEST.in",
        "hatch.toml",
        "requirements.txt",
    ):
        path = repository_root / name
        if path.is_file():
            paths.add(path)
    return tuple(sorted(paths, key=lambda item: item.relative_to(repository_root).as_posix()))


_EXCLUDED_DIRECTORIES = {"__pycache__", "build", "dist", ".venv", ".torch-dae"}
_EXCLUDED_SUFFIXES = {".pyc"}
_EXCLUDED_NAMES = {".DS_Store"}


def _excluded_file(filename: str) -> bool:
    return (
        filename in _EXCLUDED_NAMES
        or filename.startswith("._")
        or filename.endswith("~")
        or filename.endswith(".tmp")
        or Path(filename).suffix in _EXCLUDED_SUFFIXES
    )


def _project_readme_paths(repository_root: Path, pyproject: Path) -> tuple[Path, ...]:
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError:
        return ()
    readme = data.get("project", {}).get("readme")
    if isinstance(readme, str):
        path = repository_root / readme
        return (path,) if path.is_file() else ()
    if isinstance(readme, dict) and isinstance(readme.get("file"), str):
        path = repository_root / str(readme["file"])
        return (path,) if path.is_file() else ()
    return ()
