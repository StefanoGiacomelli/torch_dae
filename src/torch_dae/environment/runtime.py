"""Runtime metadata models for ignored environment materializations."""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field

from torch_dae.contracts import SHA256_PATTERN, CanonicalId, StrictBaseModel

if TYPE_CHECKING:
    from torch_dae.environment.subprocess import ManagedProcessResult


def utc_now() -> str:
    """Return a stable ISO-8601 UTC timestamp."""

    return datetime.now(UTC).isoformat()


class InstalledPackageRecord(StrictBaseModel):
    """Installed Python distribution record."""

    name: str
    normalized_name: str
    version: str
    location: str


class InstalledSourceRecord(StrictBaseModel):
    """Installed upstream-source record."""

    source_id: CanonicalId
    installation: str
    location: str
    version: str | None = None
    revision: str | None = None
    wheel_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    file_hashes: dict[str, str] = Field(default_factory=dict)
    wheel_members: dict[str, str] = Field(default_factory=dict)


class VerificationExecutionRecord(StrictBaseModel):
    """Captured verification command result."""

    status: Literal["valid", "invalid", "missing", "incomplete", "stale"]
    details: str
    command: tuple[str, ...] = ()
    returncode: int | None = None
    stdout: str | None = None
    stderr: str | None = None


class EnvironmentMaterializationRecord(StrictBaseModel):
    """Strict runtime record for `torch-dae-materialization.json`."""

    schema_version: Literal["1.0.0"]
    status: Literal["building", "complete", "failed"]
    card_id: CanonicalId
    environment_id: CanonicalId
    fingerprint: str = Field(pattern=SHA256_PATTERN)
    platform: str
    python_requested_version: str
    python_actual_version: str | None = None
    python_executable: str | None = None
    python_provider: str | None = None
    created_at: str
    completed_at: str | None = None
    environment_spec_sha256: str = Field(pattern=SHA256_PATTERN)
    lockfile_sha256: str = Field(pattern=SHA256_PATTERN)
    source_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    local_package_identity: str
    local_package_wheel_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    installed_packages: tuple[InstalledPackageRecord, ...] = ()
    installed_sources: tuple[InstalledSourceRecord, ...] = ()
    verification_result: VerificationExecutionRecord | None = None
    command_log_references: tuple[str, ...] = ()


class LocalWheelCacheRecord(StrictBaseModel):
    """Strict cache metadata for a locally built `torch-dae` wheel."""

    schema_version: Literal["1.0.0"]
    package_identity: str
    distribution_name: str
    distribution_version: str
    wheel_filename: str
    wheel_sha256: str = Field(pattern=SHA256_PATTERN)
    source_date_epoch: str
    build_command: tuple[str, ...]
    created_at: str


class GitSourceWheelCacheRecord(StrictBaseModel):
    """Strict cache metadata for a Git-source wheel."""

    schema_version: Literal["1.0.0"]
    source_id: CanonicalId
    source_url: str
    source_revision: str
    build_fingerprint: str = Field(pattern=SHA256_PATTERN)
    python_version: str
    platform: str
    lockfile_sha256: str = Field(pattern=SHA256_PATTERN)
    distribution_name: str
    distribution_version: str
    wheel_filename: str
    wheel_sha256: str = Field(pattern=SHA256_PATTERN)
    created_at: str


class RuntimeReportSink:
    """Atomic writer for sanitized ignored runtime operation reports."""

    def __init__(self, runtime_root: Path, *relative_parts: str) -> None:
        self.runtime_root = runtime_root
        self.report_dir = runtime_root.joinpath(*relative_parts)
        self.references: list[str] = []

    def record_command(
        self,
        *,
        operation: str,
        arguments: tuple[str, ...],
        working_directory: str | None,
        started_at: str,
        completed_at: str,
        duration_seconds: float,
        return_code: int | None,
        stdout: str,
        stderr: str,
        status: Literal["success", "failed", "timeout", "unavailable"],
    ) -> str:
        return self.record_event(
            operation=operation,
            status=status,
            arguments=arguments,
            working_directory=working_directory,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration_seconds,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
        )

    def record_event(
        self,
        *,
        operation: str,
        status: Literal["success", "failed", "timeout", "unavailable"],
        arguments: tuple[str, ...] = (),
        working_directory: str | None = None,
        started_at: str | None = None,
        completed_at: str | None = None,
        duration_seconds: float = 0.0,
        return_code: int | None = None,
        stdout: str = "",
        stderr: str = "",
        extra: dict[str, object] | None = None,
    ) -> str:
        start = started_at or utc_now()
        complete = completed_at or utc_now()
        payload: dict[str, object] = {
            "schema_version": "1.0.0",
            "operation": sanitize_text(operation),
            "status": status,
            "arguments": [sanitize_text(item) for item in arguments],
            "working_directory": sanitize_text(working_directory) if working_directory else None,
            "started_at": start,
            "completed_at": complete,
            "duration_seconds": duration_seconds,
            "return_code": return_code,
            "stdout": sanitize_text(stdout),
            "stderr": sanitize_text(stderr),
        }
        if extra:
            payload.update({key: sanitize_object(value) for key, value in extra.items()})
        self.report_dir.mkdir(parents=True, exist_ok=True)
        path = self.report_dir / f"{operation}-{uuid.uuid4().hex}.json"
        write_json_atomic(path, payload)
        reference = str(path.relative_to(self.runtime_root))
        self.references.append(reference)
        return reference


def sanitize_object(value: object) -> object:
    """Recursively sanitize strings in report payloads."""

    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, dict):
        return {sanitize_text(str(key)): sanitize_object(child) for key, child in value.items()}
    if isinstance(value, list | tuple):
        return [sanitize_object(child) for child in value]
    return value


def sanitize_text(value: str) -> str:
    """Redact tokens, authorization values, and credential-bearing URLs."""

    redacted = _redact_urls(value)
    redacted = re.sub(r"(?i)(authorization\s*:\s*)[^\n\r]+", r"\1[redacted secret]", redacted)
    redacted = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [redacted secret]", redacted)
    redacted = re.sub(r"(?i)(token=)[^\s&]+", r"\1[redacted secret]", redacted)
    for secret in _secret_environment_values():
        if secret:
            redacted = redacted.replace(secret, "[redacted secret]")
    return redacted


def _redact_urls(value: str) -> str:
    parts = value.split()
    redacted: list[str] = []
    for part in parts:
        redacted.append(_redact_url_token(part))
    if len(parts) == 1:
        return redacted[0] if redacted else value
    rebuilt = value
    for before, after in zip(parts, redacted, strict=False):
        rebuilt = rebuilt.replace(before, after)
    return rebuilt


def _redact_url_token(value: str) -> str:
    split = urlsplit(value)
    if not split.scheme or "@" not in split.netloc:
        return value
    host = split.hostname or ""
    if split.port is not None:
        host = f"{host}:{split.port}"
    return urlunsplit(
        (split.scheme, f"[redacted secret]@{host}", split.path, split.query, split.fragment)
    )


def _secret_environment_values() -> tuple[str, ...]:
    secret_names = ("TOKEN", "SECRET", "PASSWORD", "AUTHORIZATION", "CREDENTIAL")
    return tuple(
        value
        for name, value in os.environ.items()
        if value and any(marker in name.upper() for marker in secret_names)
    )


def write_json_atomic(path: Path, data: object) -> None:
    """Write JSON data atomically to `path`."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    payload: Any
    if isinstance(data, BaseModel):
        payload = data.model_dump(mode="json")
    else:
        payload = data
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


VerificationStatus = Literal["valid", "invalid", "missing", "incomplete", "stale"]


def result_to_verification(
    status: VerificationStatus,
    details: str,
    result: ManagedProcessResult | None = None,
) -> VerificationExecutionRecord:
    """Convert an optional command result into a verification record."""

    if result is None:
        return VerificationExecutionRecord(status=status, details=details)
    return VerificationExecutionRecord(
        status=status,
        details=details,
        command=result.command,
        returncode=result.returncode,
        stdout=result.stdout or None,
        stderr=result.stderr or None,
    )
