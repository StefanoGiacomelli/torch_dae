"""Environment specification contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from pydantic import Field, HttpUrl, field_validator, model_validator

from torch_dae.contracts import (
    GIT_REVISION_PATTERN,
    REPO_RELATIVE_PATTERN,
    CanonicalId,
    StrictBaseModel,
)
from torch_dae.contracts import ensure_repository_relative as _ensure_repository_relative


def ensure_repository_relative(value: str) -> str:
    return _ensure_repository_relative(value) or value


class SourceInstallationType(StrEnum):
    """Source-installation priority variants."""

    PACKAGE = "package"
    GIT = "git"
    VENDORED = "vendored"


class PythonSpecification(StrictBaseModel):
    """Declared and resolved Python version."""

    constraint: str
    resolved_version: Annotated[str, Field(pattern=r"^[0-9]+(\.[0-9]+){1,2}$")]

    @model_validator(mode="after")
    def resolved_satisfies_constraint(self) -> PythonSpecification:
        try:
            version = Version(self.resolved_version)
            specifier = SpecifierSet(self.constraint)
        except (InvalidVersion, InvalidSpecifier) as exc:
            raise ValueError("invalid Python version or constraint") from exc
        if version not in specifier:
            raise ValueError("resolved Python version does not satisfy constraint")
        return self


class PlatformSpecification(StrictBaseModel):
    """Platform evidence for a model environment."""

    resolved_on: tuple[str, ...]
    expected_compatible: tuple[str, ...]
    verified: tuple[str, ...]


class OfficialPackageSource(StrictBaseModel):
    """Official package installation source."""

    source_id: CanonicalId
    role: str
    installation: Literal[SourceInstallationType.PACKAGE]
    package: str
    version: str


class PinnedGitSource(StrictBaseModel):
    """Pinned official Git repository source."""

    source_id: CanonicalId
    role: str
    installation: Literal[SourceInstallationType.GIT]
    url: str
    revision: Annotated[str, Field(pattern=GIT_REVISION_PATTERN)]
    build: Literal["wheel"]

    @field_validator("url")
    @classmethod
    def url_is_explicit(cls, value: str) -> str:
        if not value:
            raise ValueError("git source url must be explicit")
        return value


class VendoredAdaptationSource(StrictBaseModel):
    """Minimal vendored adaptation source."""

    source_id: CanonicalId
    role: str
    installation: Literal[SourceInstallationType.VENDORED]
    upstream_url: HttpUrl
    upstream_revision: Annotated[str, Field(pattern=GIT_REVISION_PATTERN)]
    copied_files: tuple[Annotated[str, Field(pattern=REPO_RELATIVE_PATTERN)], ...]
    adaptation_description: str
    justification: str

    @field_validator("copied_files")
    @classmethod
    def copied_files_repository_relative(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            ensure_repository_relative(item)
        return value


SourceInstallation = Annotated[
    OfficialPackageSource | PinnedGitSource | VendoredAdaptationSource,
    Field(discriminator="installation"),
]


class EnvironmentVerificationSpec(StrictBaseModel):
    """Committed verification script reference."""

    script: Annotated[str, Field(pattern=REPO_RELATIVE_PATTERN)]

    @model_validator(mode="after")
    def script_repository_relative(self) -> EnvironmentVerificationSpec:
        ensure_repository_relative(self.script)
        return self


class EnvironmentSpecification(StrictBaseModel):
    """Validate the committed inputs for one isolated model environment.

    Attributes
    ----------
    environment_id
        Canonical identifier for the environment definition.
    model_card_id
        Card whose model-family, variant, and checkpoint this environment supports.
    python
        Declared version constraint and the exact resolved Python version.
    platforms
        Reported, expected-compatible, and verified platform evidence.
    lockfile
        Repository-relative path to the complete dependency lock.
    project_file
        Repository-relative path to the environment project metadata.
    sources_file
        Repository-relative path to the canonical source manifest.
    verification
        Committed verification-script reference.

    Raises
    ------
    pydantic.ValidationError
        If a path escapes the repository, an identity is malformed, or the resolved Python version
        violates its declared constraint.

    Notes
    -----
    Validation reads no runtime state and does not create the environment.
    """

    schema_version: Literal["1.0.0"]
    environment_id: CanonicalId
    model_card_id: CanonicalId
    python: PythonSpecification
    platforms: PlatformSpecification
    dependency_manager: Literal["uv"]
    lockfile: Annotated[str, Field(pattern=REPO_RELATIVE_PATTERN)]
    project_file: Annotated[str, Field(pattern=REPO_RELATIVE_PATTERN)]
    sources_file: Annotated[str, Field(pattern=REPO_RELATIVE_PATTERN)]
    verification: EnvironmentVerificationSpec

    @model_validator(mode="after")
    def paths_repository_relative(self) -> EnvironmentSpecification:
        ensure_repository_relative(self.lockfile)
        ensure_repository_relative(self.project_file)
        ensure_repository_relative(self.sources_file)
        return self


class EnvironmentSourcesManifest(StrictBaseModel):
    """Canonical committed source manifest referenced by environment.json."""

    schema_version: Literal["1.0.0"]
    environment_id: CanonicalId
    sources: tuple[SourceInstallation, ...]

    @model_validator(mode="after")
    def source_ids_are_unique(self) -> EnvironmentSourcesManifest:
        ids = [source.source_id for source in self.sources]
        if len(ids) != len(set(ids)):
            raise ValueError("source IDs must be unique")
        return self
