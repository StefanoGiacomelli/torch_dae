"""Shared strict contract helpers for committed artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

SHA256_PATTERN = r"^[0-9a-f]{64}$"
GIT_REVISION_PATTERN = r"^[0-9a-f]{40}$"
REPO_RELATIVE_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._/-]*$"
CANONICAL_ID_PATTERN = r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$"
WRAPPER_ENTRY_POINT_PATTERN = (
    r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Z][A-Za-z0-9_]*$"
)
CanonicalId = Annotated[str, Field(pattern=CANONICAL_ID_PATTERN)]


class StrictBaseModel(BaseModel):
    """Base for committed artifacts: frozen, closed, and no nested empty strings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="before")
    @classmethod
    def reject_nested_empty_strings(cls, value: object) -> object:
        reject_empty_strings(value)
        return value


def reject_empty_strings(value: object) -> None:
    """Reject empty strings recursively while permitting nulls and empty tuples/lists."""

    if value == "":
        raise ValueError("unresolved values must be explicit, not empty strings")
    if isinstance(value, dict):
        for child in value.values():
            reject_empty_strings(child)
    elif isinstance(value, list | tuple):
        for child in value:
            reject_empty_strings(child)


def ensure_repository_relative(value: str | None) -> str | None:
    """Validate strict repository-relative POSIX paths."""

    if value is None:
        return None
    if not value:
        raise ValueError("path must not be empty")
    if value.startswith("/") or "\\" in value:
        raise ValueError("path must be repository-relative POSIX")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("path must not contain empty, '.', or '..' segments")
    if re.fullmatch(REPO_RELATIVE_PATTERN, value) is None:
        raise ValueError("path contains unsupported characters")
    return value


def ensure_canonical_id(value: str) -> str:
    """Validate canonical path-safe artifact identifiers."""

    if re.fullmatch(CANONICAL_ID_PATTERN, value) is None:
        raise ValueError("identifier must be lowercase path-safe canonical ID")
    return value


def contained_path(parent: Path, *parts: str) -> Path:
    """Build a path and prove it remains under `parent`."""

    parent_resolved = parent.resolve()
    candidate = parent.joinpath(*parts).resolve()
    try:
        candidate.relative_to(parent_resolved)
    except ValueError as exc:
        raise ValueError(f"path escapes intended parent: {candidate}") from exc
    return candidate


def ensure_wrapper_entry_point(value: str) -> str:
    """Validate `python.module.path:ClassName` without importing it."""

    if re.fullmatch(WRAPPER_ENTRY_POINT_PATTERN, value) is None:
        raise ValueError("wrapper_entry_point must use python.module.path:ClassName")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic compact JSON bytes."""

    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
