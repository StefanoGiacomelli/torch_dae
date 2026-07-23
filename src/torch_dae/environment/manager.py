"""Phase 00 environment manager interface."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from torch_dae.contracts import contained_path, ensure_canonical_id
from torch_dae.core.errors import NotImplementedInPhaseError
from torch_dae.environment.fingerprint import (
    FingerprintInputs,
    calculate_environment_fingerprint,
    canonical_platform_tag,
    local_package_identity,
)
from torch_dae.environment.materialization import materialization_path
from torch_dae.environment.specification import EnvironmentSourcesManifest, EnvironmentSpecification
from torch_dae.environment.subprocess import ManagedProcessResult


@dataclass(frozen=True)
class InstalledSource:
    """Installed source metadata for a resolved environment."""

    source_id: str
    location: str
    revision: str | None = None


@dataclass(frozen=True)
class ResolvedEnvironment:
    """Future materialized environment metadata."""

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
    """Read-only local-state inspection for Phase 00."""

    model_card_id: str
    specification_path: Path
    specification_exists: bool
    fingerprint: str | None
    expected_path: Path | None
    materialized: bool


@dataclass(frozen=True)
class EnvironmentVerification:
    """Environment verification result interface."""

    model_card_id: str
    passed: bool
    details: str


def discover_repository_root(start: Path | None = None) -> Path:
    """Find the repository root containing `project_spec.md`."""

    current = (start or Path.cwd()).resolve()
    for path in (current, *current.parents):
        if (path / "project_spec.md").is_file():
            return path
    raise FileNotFoundError("could not discover repository root containing project_spec.md")


class EnvironmentManager:
    """Environment manager public interface.

    Phase 00 implements deterministic read-only helpers. Mutating operations are
    deferred to Phase 01 and fail truthfully.
    """

    def __init__(self, repository_root: Path) -> None:
        self.repository_root = repository_root.resolve()
        self.runtime_root = self.repository_root / ".torch-dae"

    @classmethod
    def from_repository_root(cls, start: Path | None = None) -> EnvironmentManager:
        return cls(discover_repository_root(start))

    def specification_path(self, model_card_id: str) -> Path:
        ensure_canonical_id(model_card_id)
        return contained_path(
            self.repository_root / "environments", model_card_id, "environment.json"
        )

    def load_specification(self, model_card_id: str) -> EnvironmentSpecification:
        path = self.specification_path(model_card_id)
        specification = EnvironmentSpecification.model_validate_json(path.read_text())
        if specification.model_card_id != model_card_id:
            raise ValueError("environment specification model_card_id does not match request")
        return specification

    def load_sources_manifest(
        self, specification: EnvironmentSpecification
    ) -> EnvironmentSourcesManifest:
        path = self.repository_root / specification.sources_file
        if not path.exists():
            raise FileNotFoundError(f"missing environment source manifest: {path}")
        manifest = EnvironmentSourcesManifest.model_validate_json(path.read_text())
        if manifest.environment_id != specification.environment_id:
            raise ValueError(
                "environment source manifest environment_id does not match specification"
            )
        return manifest

    def fingerprint_for(self, specification: EnvironmentSpecification) -> str:
        lock_path = self.repository_root / specification.lockfile
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
        raise NotImplementedInPhaseError(f"env create for {model_card_id!r} belongs to Phase 01")

    def ensure(self, model_card_id: str) -> ResolvedEnvironment:
        raise NotImplementedInPhaseError(f"env ensure for {model_card_id!r} belongs to Phase 01")

    def verify(self, model_card_id: str) -> EnvironmentVerification:
        raise NotImplementedInPhaseError(f"env verify for {model_card_id!r} belongs to Phase 01")

    def remove(self, model_card_id: str) -> None:
        raise NotImplementedInPhaseError(f"env remove for {model_card_id!r} belongs to Phase 01")

    def info(self, model_card_id: str) -> EnvironmentInfo:
        specification_path = self.specification_path(model_card_id)
        if not specification_path.exists():
            return EnvironmentInfo(model_card_id, specification_path, False, None, None, False)
        specification = self.load_specification(model_card_id)
        fingerprint = self.fingerprint_for(specification)
        expected = materialization_path(self.runtime_root, model_card_id, fingerprint)
        return EnvironmentInfo(
            model_card_id=model_card_id,
            specification_path=specification_path,
            specification_exists=True,
            fingerprint=fingerprint,
            expected_path=expected,
            materialized=expected.exists(),
        )

    def run(self, model_card_id: str, command: list[str]) -> ManagedProcessResult:
        if not command:
            raise ValueError("command must not be empty")
        raise NotImplementedInPhaseError(f"env run for {model_card_id!r} belongs to Phase 01")

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
            },
            indent=2,
            sort_keys=True,
        )
