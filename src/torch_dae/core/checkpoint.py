"""Checkpoint contracts and Phase 00 manager interface."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, HttpUrl, field_validator, model_validator

from torch_dae.contracts import (
    GIT_REVISION_PATTERN,
    SHA256_PATTERN,
    CanonicalId,
    StrictBaseModel,
    contained_path,
    ensure_canonical_id,
)
from torch_dae.core.errors import NotImplementedInPhaseError


class CheckpointSourceType(StrEnum):
    """Supported checkpoint source variants."""

    HTTPS = "https"
    GITHUB_RELEASE = "github_release"
    HUGGINGFACE = "huggingface"
    PACKAGE_BUNDLE = "package_bundle"
    LOCAL_PATH = "local_path"


class LicenseRecord(StrictBaseModel):
    """Informational license metadata; never an automatic blocker."""

    name: str | None = None
    url: HttpUrl | None = None
    status: Literal[
        "officially_reported",
        "observed",
        "inferred",
        "unresolved",
        "not_reported",
        "not_applicable",
    ]


class CheckpointSpec(StrictBaseModel):
    """Checkpoint specification for one concrete pretrained asset."""

    schema_version: Literal["1.0.0"]
    checkpoint_id: CanonicalId
    source_type: CheckpointSourceType
    url: HttpUrl | None = None
    repository_id: str | None = None
    package: str | None = None
    package_version: str | None = None
    revision: Annotated[str | None, Field(pattern=GIT_REVISION_PATTERN)] = None
    release_tag: str | None = None
    filename: str | None = None
    local_path: str | None = None
    expected_sha256: Annotated[str | None, Field(pattern=SHA256_PATTERN)] = None
    observed_sha256: Annotated[str | None, Field(pattern=SHA256_PATTERN)] = None
    format: str
    loader: str
    license: LicenseRecord

    @field_validator("local_path")
    @classmethod
    def local_path_is_relative(cls, value: str | None) -> str | None:
        from torch_dae.contracts import ensure_repository_relative

        return ensure_repository_relative(value)

    @model_validator(mode="after")
    def validate_source_fields(self) -> CheckpointSpec:
        match self.source_type:
            case CheckpointSourceType.HTTPS:
                if self.url is None:
                    raise ValueError("https checkpoints require url")
                self._reject_non_null(
                    "https",
                    "repository_id",
                    "package",
                    "package_version",
                    "revision",
                    "release_tag",
                    "local_path",
                )
            case CheckpointSourceType.GITHUB_RELEASE:
                if self.repository_id is None or self.release_tag is None or self.filename is None:
                    raise ValueError(
                        "github_release checkpoints require repository_id, release_tag, filename"
                    )
                self._reject_non_null(
                    "github_release", "url", "package", "package_version", "local_path"
                )
            case CheckpointSourceType.HUGGINGFACE:
                if self.repository_id is None or self.filename is None:
                    raise ValueError("huggingface checkpoints require repository_id and filename")
                self._reject_non_null(
                    "huggingface", "url", "package", "package_version", "release_tag", "local_path"
                )
            case CheckpointSourceType.PACKAGE_BUNDLE:
                if self.package is None or self.package_version is None or self.filename is None:
                    raise ValueError(
                        "package_bundle checkpoints require package, package_version, filename"
                    )
                self._reject_non_null(
                    "package_bundle",
                    "url",
                    "repository_id",
                    "revision",
                    "release_tag",
                    "local_path",
                )
            case CheckpointSourceType.LOCAL_PATH:
                if self.local_path is None:
                    raise ValueError("local_path checkpoints require local_path")
                self._reject_non_null(
                    "local_path",
                    "url",
                    "repository_id",
                    "package",
                    "package_version",
                    "revision",
                    "release_tag",
                    "filename",
                )
        return self

    def _reject_non_null(self, source_type: str, *fields: str) -> None:
        present = [field for field in fields if getattr(self, field) is not None]
        if present:
            raise ValueError(f"{source_type} checkpoints do not allow fields: {present}")


class ResolvedCheckpoint(StrictBaseModel):
    """Resolved local checkpoint metadata."""

    checkpoint_id: str
    sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    path: Path
    immutable: bool


def validate_sha256(value: str) -> str:
    """Validate a lowercase SHA-256 string in Phase 00."""

    return (
        CheckpointSpec(
            schema_version="1.0.0",
            checkpoint_id="sha256-validation",
            source_type=CheckpointSourceType.LOCAL_PATH,
            local_path="fixture.bin",
            observed_sha256=value,
            format="binary",
            loader="manual",
            license=LicenseRecord(status="unresolved"),
        ).observed_sha256
        or value
    )


def checkpoint_cache_path(runtime_root: Path, checkpoint_id: str, sha256: str) -> Path:
    """Return future cache path `.torch-dae/checkpoints/<checkpoint-id>/<sha256>/`."""

    validate_sha256(sha256)
    ensure_canonical_id(checkpoint_id)
    return contained_path(runtime_root / "checkpoints", checkpoint_id, sha256)


class CheckpointManager:
    """Phase 00 checkpoint manager interface.

    Acquisition is deferred to Phase 01.
    """

    def __init__(self, repository_root: Path) -> None:
        self.repository_root = repository_root
        self.runtime_root = repository_root / ".torch-dae"

    def ensure(self, card_id: str) -> ResolvedCheckpoint:
        raise NotImplementedInPhaseError(f"checkpoint ensure for {card_id!r} belongs to Phase 01")

    def info(self, card_id: str) -> dict[str, str]:
        return {"card_id": card_id, "runtime_root": str(self.runtime_root / "checkpoints")}

    def remove(self, card_id: str) -> None:
        raise NotImplementedInPhaseError(f"checkpoint remove for {card_id!r} belongs to Phase 01")
