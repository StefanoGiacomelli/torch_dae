"""Checkpoint contracts and cache-backed acquisition manager."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, BinaryIO, Literal, Protocol
from urllib.parse import quote, urlparse

from pydantic import Field, HttpUrl, field_validator, model_validator

from torch_dae.contracts import (
    GIT_REVISION_PATTERN,
    SHA256_PATTERN,
    CanonicalId,
    StrictBaseModel,
    canonical_json_bytes,
    contained_path,
    ensure_canonical_id,
    ensure_repository_relative,
)
from torch_dae.core.errors import (
    CheckpointAcquisitionError,
    CheckpointAuthorityResolutionError,
    CheckpointHashMismatchError,
    CheckpointNotFoundError,
    CheckpointPublishedChecksumMismatchError,
    CheckpointResponseTooLargeError,
    CheckpointSizeMismatchError,
    ExternalCommandError,
    OfflineResourceUnavailableError,
)
from torch_dae.environment.policy import ExecutionPolicy
from torch_dae.environment.runtime import (
    RuntimeReportSink,
    sanitize_text,
    utc_now,
    write_json_atomic,
)
from torch_dae.environment.sources import sha256_file
from torch_dae.environment.subprocess import CommandExecutor


class CheckpointSourceType(StrEnum):
    """Enumerate supported checkpoint acquisition categories.

    ``https`` streams a direct URL; ``github_release`` resolves a repository, tag, and filename;
    ``huggingface`` resolves a repository, optional revision, and filename; ``package_bundle``
    reads a file from an exact installed package; and ``local_path`` copies a repository-relative
    file. The enum selects validation and acquisition behavior.
    """

    HTTPS = "https"
    GITHUB_RELEASE = "github_release"
    HUGGINGFACE = "huggingface"
    PACKAGE_BUNDLE = "package_bundle"
    LOCAL_PATH = "local_path"


CHECKPOINT_PATH_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._=+@/-]*$")
MD5_PATTERN = r"^[0-9a-f]{32}$"
ZENODO_METADATA_MAXIMUM_BYTES = 4 * 1024 * 1024
ZENODO_HOSTS = frozenset({"zenodo.org", "www.zenodo.org"})


def ensure_checkpoint_relative_path(value: str | None) -> str | None:
    """Validate a safe provider filename or relative checkpoint resource path.

    Checkpoint-host paths intentionally have a dedicated grammar: metric-bearing ``=`` names and
    nested package resources are valid, while repository traversal, URL syntax, Windows paths,
    backslashes, NULs, empty segments, and ambiguous dot segments are not.
    """

    if value is None:
        return None
    if not value or value.startswith("/") or "\\" in value or "\x00" in value:
        raise ValueError("checkpoint path must be a nonempty relative POSIX path")
    if "?" in value or "#" in value or re.match(r"^[A-Za-z]:", value):
        raise ValueError("checkpoint path must not contain URL or drive-path syntax")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("checkpoint path must not contain empty, '.', or '..' segments")
    if CHECKPOINT_PATH_PATTERN.fullmatch(value) is None:
        raise ValueError("checkpoint path contains unsupported characters")
    return value


class ChecksumAlgorithm(StrEnum):
    """Published digest algorithms understood by the checkpoint authority contract."""

    MD5 = "md5"
    SHA256 = "sha256"


class PublishedChecksum(StrictBaseModel):
    """One digest explicitly published by an authoritative provider."""

    algorithm: ChecksumAlgorithm
    digest: str

    @model_validator(mode="after")
    def digest_matches_algorithm(self) -> PublishedChecksum:
        pattern = MD5_PATTERN if self.algorithm == ChecksumAlgorithm.MD5 else SHA256_PATTERN
        if re.fullmatch(pattern, self.digest) is None:
            raise ValueError(f"invalid {self.algorithm.value} digest")
        return self


class ObservedChecksum(StrictBaseModel):
    """One digest calculated from acquired or cached local bytes."""

    algorithm: ChecksumAlgorithm
    digest: str

    @model_validator(mode="after")
    def digest_matches_algorithm(self) -> ObservedChecksum:
        pattern = MD5_PATTERN if self.algorithm == ChecksumAlgorithm.MD5 else SHA256_PATTERN
        if re.fullmatch(pattern, self.digest) is None:
            raise ValueError(f"invalid {self.algorithm.value} digest")
        return self


class CheckpointAuthority(StrictBaseModel):
    """Immutable authoritative identity for one provider record file."""

    provider: Literal["zenodo"]
    record_id: Annotated[str, Field(pattern=r"^[1-9][0-9]*$")]
    filename: str
    expected_size_bytes: int = Field(gt=0)
    published_checksums: tuple[PublishedChecksum, ...]
    record_url: HttpUrl | None = None
    provenance_status: Literal["authoritative_provider_declared"]

    @field_validator("filename")
    @classmethod
    def filename_is_safe(cls, value: str) -> str:
        return ensure_checkpoint_relative_path(value) or value

    @field_validator("published_checksums")
    @classmethod
    def checksums_are_canonical(
        cls, value: tuple[PublishedChecksum, ...]
    ) -> tuple[PublishedChecksum, ...]:
        return tuple(sorted(value, key=lambda item: item.algorithm.value))

    @model_validator(mode="after")
    def authority_is_unambiguous(self) -> CheckpointAuthority:
        if not self.published_checksums:
            raise ValueError("checkpoint authority requires published checksums")
        algorithms = [item.algorithm for item in self.published_checksums]
        if len(algorithms) != len(set(algorithms)):
            raise ValueError("published checksum algorithms must be unique")
        if self.record_url is not None:
            parsed = urlparse(str(self.record_url))
            if parsed.scheme != "https" or parsed.hostname not in ZENODO_HOSTS:
                raise ValueError("Zenodo record URL must use an official HTTPS host")
            if parsed.path.rstrip("/") != f"/records/{self.record_id}":
                raise ValueError("Zenodo record URL must match record_id")
        return self


class CheckpointAuthorityResolution(StrictBaseModel):
    """Sanitized provenance from metadata-only authoritative resolution."""

    schema_version: Literal["1.0.0"]
    provider: Literal["zenodo"]
    record_id: str
    requested_filename: str
    resolved_filename: str
    expected_size_bytes: int = Field(gt=0)
    published_checksums: tuple[PublishedChecksum, ...]
    metadata_request_url: HttpUrl
    resolved_payload_url: HttpUrl
    http_status: int = Field(ge=200, lt=400)
    etag: str | None = None
    last_modified: str | None = None
    content_type: str | None = None
    metadata_response_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    retrieved_at: str
    metadata_source: Literal["network", "managed_cache"]
    warnings: tuple[str, ...] = ()
    failure_classification: str | None = None

    @model_validator(mode="after")
    def resolution_is_consistent(self) -> CheckpointAuthorityResolution:
        if self.requested_filename != self.resolved_filename:
            raise ValueError("authority resolution filename identity mismatch")
        algorithms = [item.algorithm for item in self.published_checksums]
        if not algorithms or len(algorithms) != len(set(algorithms)):
            raise ValueError("resolved published checksums must be nonempty and unique")
        for value, label in (
            (str(self.metadata_request_url), "metadata"),
            (str(self.resolved_payload_url), "payload"),
        ):
            parsed = urlparse(value)
            if parsed.scheme != "https" or parsed.hostname not in ZENODO_HOSTS:
                raise ValueError(f"resolved Zenodo {label} URL is not provider-controlled HTTPS")
        if self.failure_classification is not None:
            raise ValueError("successful authority resolution cannot declare a failure")
        return self


class CheckpointAcquisitionPolicy(StrictBaseModel):
    """Explicit network, authentication, authority, and resource-safety policy."""

    allow_network: bool
    allow_authentication: bool
    maximum_bytes: int | None = Field(default=None, gt=0)
    require_expected_sha256: bool = False
    require_authority: bool = False
    require_exact_size: bool = False
    require_published_checksums: bool = False
    require_observed_sha256: bool = False


class LicenseRecord(StrictBaseModel):
    """Record informational checkpoint-license evidence.

    Attributes
    ----------
    name, url
        Optional reported license name and source URL.
    status
        Provenance status: officially reported, observed, inferred, unresolved, not reported, or
        not applicable.

    Notes
    -----
    This metadata never authorizes use and never blocks acquisition automatically.
    """

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
    """Validate one concrete checkpoint source and loader contract.

    Attributes
    ----------
    schema_version, checkpoint_id, source_type
        Contract version, canonical asset identity, and acquisition category.
    url, repository_id, package, package_version, revision, release_tag, filename, local_path
        Source-specific location fields. Paths and filenames must be safe and repository-relative.
    expected_sha256, observed_sha256
        Optional lowercase hexadecimal SHA-256 values; when both exist they must agree.
    format, loader
        Serialization format and model-specific loader description.
    license
        Informational license record.

    Raises
    ------
    pydantic.ValidationError
        If required source fields are missing, contradictory fields are present, a path escapes the
        repository, or hashes disagree.

    Notes
    -----
    Validation performs no network access and does not deserialize model weights.
    """

    schema_version: Literal["1.0.0", "2.0.0"]
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
    authority: CheckpointAuthority | None = None
    format: str
    loader: str
    license: LicenseRecord

    @field_validator("local_path")
    @classmethod
    def local_path_is_relative(cls, value: str | None) -> str | None:
        return ensure_repository_relative(value)

    @field_validator("filename")
    @classmethod
    def filename_is_safe_relative(cls, value: str | None) -> str | None:
        return ensure_checkpoint_relative_path(value)

    @model_validator(mode="after")
    def validate_source_fields(self) -> CheckpointSpec:
        match self.source_type:
            case CheckpointSourceType.HTTPS:
                if self.url is None and self.authority is None:
                    raise ValueError("https checkpoints require url or structured authority")
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
        if (
            self.expected_sha256
            and self.observed_sha256
            and self.expected_sha256 != self.observed_sha256
        ):
            raise ValueError("expected_sha256 and observed_sha256 must agree when both are set")
        if self.schema_version == "1.0.0" and self.authority is not None:
            raise ValueError("structured authority requires checkpoint schema 2.0.0")
        if self.schema_version == "2.0.0":
            if self.authority is None:
                raise ValueError("checkpoint schema 2.0.0 requires structured authority")
            if self.source_type != CheckpointSourceType.HTTPS:
                raise ValueError("structured authority currently requires https source_type")
            if self.filename != self.authority.filename:
                raise ValueError("checkpoint filename must exactly match authority filename")
            if self.url is not None:
                raise ValueError(
                    "authoritative payload URL must be resolved from provider metadata"
                )
        return self

    def _reject_non_null(self, source_type: str, *fields: str) -> None:
        present = [field for field in fields if getattr(self, field) is not None]
        if present:
            raise ValueError(f"{source_type} checkpoints do not allow fields: {present}")


class ResolvedCheckpoint(StrictBaseModel):
    """Describe an integrity-checked local checkpoint.

    Attributes
    ----------
    checkpoint_id
        Canonical asset identifier.
    sha256
        Observed lowercase hexadecimal SHA-256 digest.
    path
        Local cached file path.
    immutable
        Whether the resolved cache identity is content-addressed and immutable.
    """

    checkpoint_id: str
    sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    path: Path
    immutable: bool


class CheckpointMaterializationRecord(StrictBaseModel):
    """Persist sanitized runtime metadata for a cached checkpoint.

    Attributes
    ----------
    schema_version, checkpoint_id, source_type
        Record schema and source identity.
    source_description, resolved_url_or_location, filename
        Sanitized acquisition description and stored filename.
    sha256, size_bytes, acquired_at, cache_path
        Observed integrity, byte count, timestamp, and runtime-relative cache location.
    expected_sha256, observed_sha256
        Hash evidence copied from the specification.
    environment_id
        Optional isolated environment used for package-bundle acquisition.
    specification_fingerprint
        SHA-256 identity of the canonical checkpoint specification.
    command_log_references
        Runtime-report references with secrets removed.
    """

    schema_version: Literal["1.0.0", "2.0.0"]
    checkpoint_id: CanonicalId
    source_type: CheckpointSourceType
    source_description: str
    resolved_url_or_location: str
    filename: str
    sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    size_bytes: int
    acquired_at: str
    cache_path: str
    expected_sha256: Annotated[str | None, Field(pattern=SHA256_PATTERN)] = None
    observed_sha256: Annotated[str | None, Field(pattern=SHA256_PATTERN)] = None
    environment_id: str | None = None
    specification_fingerprint: str = Field(pattern=SHA256_PATTERN)
    command_log_references: tuple[str, ...] = ()
    authority: CheckpointAuthority | None = None
    expected_size_bytes: int | None = Field(default=None, gt=0)
    observed_size_bytes: int | None = Field(default=None, ge=0)
    published_checksums: tuple[PublishedChecksum, ...] = ()
    observed_checksums: tuple[ObservedChecksum, ...] = ()
    metadata_resolution_reference: str | None = None
    metadata_resolution_sha256: Annotated[str | None, Field(pattern=SHA256_PATTERN)] = None
    payload_etag: str | None = None
    payload_last_modified: str | None = None
    materialization_status: Literal["downloaded", "local_copy", "package_bundle"] | None = None

    @model_validator(mode="after")
    def authoritative_record_is_complete(self) -> CheckpointMaterializationRecord:
        if self.schema_version == "2.0.0":
            required = {
                "authority": self.authority,
                "expected_size_bytes": self.expected_size_bytes,
                "observed_size_bytes": self.observed_size_bytes,
                "metadata_resolution_reference": self.metadata_resolution_reference,
                "metadata_resolution_sha256": self.metadata_resolution_sha256,
                "materialization_status": self.materialization_status,
            }
            missing = sorted(name for name, value in required.items() if value is None)
            if missing:
                raise ValueError(f"authoritative materialization is missing: {missing}")
            if self.expected_size_bytes != self.observed_size_bytes:
                raise ValueError("authoritative expected and observed sizes must match")
            if self.size_bytes != self.observed_size_bytes:
                raise ValueError("legacy and authoritative observed sizes must match")
            if self.observed_sha256 != self.sha256:
                raise ValueError("observed SHA-256 must equal cache identity")
            if (
                self.authority is None
                or self.published_checksums != self.authority.published_checksums
            ):
                raise ValueError("materialization published checksums must match authority")
            observed = {item.algorithm: item.digest for item in self.observed_checksums}
            if observed.get(ChecksumAlgorithm.SHA256) != self.sha256:
                raise ValueError("observed checksum set must contain cache SHA-256")
            for item in self.published_checksums:
                if observed.get(item.algorithm) != item.digest:
                    raise ValueError("published checksums must match observed local digests")
        return self


@dataclass(frozen=True)
class TransportResponse:
    """Streaming transport response."""

    status_code: int
    headers: Mapping[str, str]
    body: BinaryIO
    effective_url: str | None = None


class DownloadTransport(Protocol):
    """Injectable checkpoint download transport."""

    def open(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: float,
    ) -> TransportResponse:
        """Open a streaming response for `url`."""


class UrllibDownloadTransport:
    """Production HTTPS transport using the Python standard library."""

    def open(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: float,
    ) -> TransportResponse:
        request = urllib.request.Request(url, headers=dict(headers))
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            body = getattr(exc, "fp", None)
            if body is not None:
                try:
                    body.close()
                except OSError:
                    pass
            if exc.code == 404:
                raise CheckpointNotFoundError(
                    f"checkpoint download failed with HTTP {exc.code}"
                ) from exc
            raise CheckpointAcquisitionError(
                f"checkpoint download failed with HTTP {exc.code}"
            ) from exc
        except urllib.error.URLError as exc:
            raise CheckpointAcquisitionError(
                f"checkpoint download failed: {sanitize_text(str(exc))}"
            ) from exc
        return TransportResponse(
            status_code=int(getattr(response, "status", 200)),
            headers=dict(response.headers.items()),
            body=response,
            effective_url=str(getattr(response, "geturl", lambda: url)()),
        )


def _header(headers: Mapping[str, str], name: str) -> str | None:
    for key, value in headers.items():
        if key.lower() == name.lower():
            return value
    return None


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _validate_zenodo_url(value: str, *, label: str) -> str:
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in ZENODO_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise CheckpointAuthorityResolutionError(
            f"Zenodo {label} URL must use a provider-controlled HTTPS host"
        )
    return value


def _published_checksums_from_metadata(
    file_record: Mapping[str, Any],
) -> tuple[PublishedChecksum, ...]:
    raw_values: list[str] = []
    checksum = file_record.get("checksum")
    if isinstance(checksum, str):
        raw_values.append(checksum)
    checksums = file_record.get("checksums")
    if isinstance(checksums, list) and all(isinstance(item, str) for item in checksums):
        raw_values.extend(checksums)
    parsed: list[PublishedChecksum] = []
    for value in raw_values:
        algorithm, separator, digest = value.partition(":")
        if not separator:
            raise CheckpointAuthorityResolutionError(
                "Zenodo published checksum must be algorithm-tagged"
            )
        try:
            parsed.append(
                PublishedChecksum.model_validate(
                    {"algorithm": algorithm.lower(), "digest": digest.lower()}
                )
            )
        except Exception as exc:
            raise CheckpointAuthorityResolutionError(
                "Zenodo published checksum is unsupported or malformed"
            ) from exc
    algorithms = [item.algorithm for item in parsed]
    if not parsed or len(algorithms) != len(set(algorithms)):
        raise CheckpointAuthorityResolutionError(
            "Zenodo file requires a nonempty unique published checksum set"
        )
    return tuple(sorted(parsed, key=lambda item: item.algorithm.value))


def resolve_checkpoint_authority(
    specification: CheckpointSpec,
    *,
    transport: DownloadTransport | None = None,
    timeout: float = 300.0,
) -> CheckpointAuthorityResolution:
    """Resolve provider metadata for one checkpoint without downloading payload bytes.

    Parameters
    ----------
    specification
        Strict authority-complete checkpoint specification.
    transport
        Optional injectable metadata transport.
    timeout
        Metadata request timeout in seconds.

    Returns
    -------
    CheckpointAuthorityResolution
        Sanitized exact-file authority and metadata-response provenance.
    """

    spec = CheckpointSpec.model_validate(specification)
    authority = spec.authority
    if authority is None:
        raise CheckpointAuthorityResolutionError(
            "checkpoint specification has no structured authority"
        )
    metadata_url = f"https://zenodo.org/api/records/{authority.record_id}"
    client = transport or UrllibDownloadTransport()
    response: TransportResponse | None = None
    try:
        response = client.open(
            metadata_url,
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
        effective_url = response.effective_url or metadata_url
        _validate_zenodo_url(effective_url, label="metadata response")
        if response.status_code < 200 or response.status_code >= 400:
            raise CheckpointAuthorityResolutionError(
                f"Zenodo metadata request failed with HTTP {response.status_code}"
            )
        content_type = _header(response.headers, "Content-Type")
        warnings: list[str] = []
        if content_type is None:
            warnings.append("metadata response omitted Content-Type")
        elif "json" not in content_type.lower():
            raise CheckpointAuthorityResolutionError(
                "Zenodo metadata response Content-Type is not JSON"
            )
        raw = bytearray()
        while True:
            chunk = response.body.read(64 * 1024)
            if not chunk:
                break
            raw.extend(chunk)
            if len(raw) > ZENODO_METADATA_MAXIMUM_BYTES:
                raise CheckpointAuthorityResolutionError(
                    "Zenodo metadata response exceeded the bounded size limit"
                )
        try:
            data = json.loads(raw, object_pairs_hook=_reject_duplicate_json_keys)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            raise CheckpointAuthorityResolutionError(
                "Zenodo metadata response is invalid JSON"
            ) from exc
        if not isinstance(data, dict) or str(data.get("id")) != authority.record_id:
            raise CheckpointAuthorityResolutionError("Zenodo metadata record identity mismatch")
        files = data.get("files")
        if not isinstance(files, list):
            raise CheckpointAuthorityResolutionError("Zenodo metadata files are missing")
        matches: list[Mapping[str, Any]] = []
        for item in files:
            if not isinstance(item, dict):
                raise CheckpointAuthorityResolutionError("Zenodo metadata file entry is malformed")
            filename = item.get("key", item.get("filename"))
            if filename == authority.filename:
                matches.append(item)
        if not matches:
            raise CheckpointNotFoundError(
                f"authoritative checkpoint file not found: {authority.filename}"
            )
        if len(matches) != 1:
            raise CheckpointAuthorityResolutionError(
                "Zenodo metadata contains duplicate requested filename identity"
            )
        file_record = matches[0]
        resolved_filename = file_record.get("key", file_record.get("filename"))
        try:
            resolved_filename = ensure_checkpoint_relative_path(str(resolved_filename))
            resolved_size = int(file_record["size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointAuthorityResolutionError(
                "Zenodo metadata file identity or size is malformed"
            ) from exc
        if resolved_filename != authority.filename:
            raise CheckpointAuthorityResolutionError("metadata/file identity mismatch")
        if resolved_size != authority.expected_size_bytes:
            raise CheckpointAuthorityResolutionError(
                "authoritative metadata exact size differs from the specification"
            )
        published = _published_checksums_from_metadata(file_record)
        expected_published = tuple(
            sorted(authority.published_checksums, key=lambda item: item.algorithm.value)
        )
        if published != expected_published:
            raise CheckpointAuthorityResolutionError(
                "authoritative metadata published checksums differ from the specification"
            )
        links = file_record.get("links")
        if not isinstance(links, dict):
            raise CheckpointAuthorityResolutionError("Zenodo payload links are missing")
        payload_url = next(
            (
                links[key]
                for key in ("content", "download", "self")
                if isinstance(links.get(key), str)
            ),
            None,
        )
        if payload_url is None:
            raise CheckpointAuthorityResolutionError("Zenodo payload URL is missing")
        _validate_zenodo_url(payload_url, label="payload")
        return CheckpointAuthorityResolution(
            schema_version="1.0.0",
            provider=authority.provider,
            record_id=authority.record_id,
            requested_filename=authority.filename,
            resolved_filename=resolved_filename,
            expected_size_bytes=resolved_size,
            published_checksums=published,
            metadata_request_url=HttpUrl(metadata_url),
            resolved_payload_url=HttpUrl(payload_url),
            http_status=response.status_code,
            etag=_header(response.headers, "ETag"),
            last_modified=_header(response.headers, "Last-Modified"),
            content_type=content_type,
            metadata_response_sha256=hashlib.sha256(raw).hexdigest(),
            retrieved_at=utc_now(),
            metadata_source="network",
            warnings=tuple(warnings),
        )
    except CheckpointAcquisitionError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise CheckpointAuthorityResolutionError(
            f"Zenodo metadata resolution failed: {sanitize_text(str(exc))}"
        ) from exc
    finally:
        if response is not None:
            try:
                response.body.close()
            except OSError:
                pass


def validate_sha256(value: str) -> str:
    """Validate a lowercase SHA-256 string."""

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
    """Return cache path `.torch-dae/checkpoints/<checkpoint-id>/<sha256>/`."""

    validate_sha256(sha256)
    ensure_canonical_id(checkpoint_id)
    return contained_path(runtime_root / "checkpoints", checkpoint_id, sha256)


class CheckpointManager:
    """Acquire, inspect, and remove content-addressed checkpoint cache entries.

    Parameters
    ----------
    repository_root
        Repository containing model cards and ignored ``.torch-dae`` runtime state.
    policy
        Network, offline, timeout, and cache execution policy.
    transport
        Optional streaming HTTP transport.
    executor
        Optional managed subprocess executor.

    Notes
    -----
    Cache entries live below ``.torch-dae/checkpoints/<checkpoint-id>/<sha256>``. Authentication
    tokens are read only at acquisition boundaries and sanitized from reports. The manager does not
    deserialize weights into a model.
    """

    def __init__(
        self,
        repository_root: Path,
        *,
        policy: ExecutionPolicy | None = None,
        transport: DownloadTransport | None = None,
        executor: CommandExecutor | None = None,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.runtime_root = self.repository_root / ".torch-dae"
        self.policy = policy or ExecutionPolicy()
        self.transport = transport or UrllibDownloadTransport()
        self.executor = executor or CommandExecutor()
        self._report_sink: RuntimeReportSink | None = None

    def ensure(self, card_id: str) -> ResolvedCheckpoint:
        """Return a valid cache entry, acquiring it when necessary.

        Parameters
        ----------
        card_id
            Canonical model-card identifier whose checkpoint specification is used.

        Returns
        -------
        ResolvedCheckpoint
            Integrity-checked, content-addressed local asset.

        Raises
        ------
        CheckpointNotFoundError
            If the card identifier is invalid, missing, or its asset cannot be found.
        OfflineResourceUnavailableError
            If offline policy forbids acquisition and no valid cached entry exists.
        CheckpointHashMismatchError
            If acquired bytes disagree with expected or observed SHA-256 evidence.
        CheckpointAcquisitionError
            If transport, package extraction, or cache installation fails.

        Notes
        -----
        Remote bodies are streamed to temporary runtime state, hashed, and atomically installed.
        Valid cached content is reused offline. Sanitized runtime reports are written under
        ``.torch-dae/reports/checkpoints``.
        """

        from torch_dae.core.registry import ModelCardRegistry

        try:
            ensure_canonical_id(card_id)
        except ValueError as exc:
            raise CheckpointNotFoundError(f"invalid checkpoint card ID: {card_id}") from exc
        try:
            card = ModelCardRegistry(self.repository_root).get_card(card_id)
            spec = card.checkpoint
        except KeyError as exc:
            raise CheckpointNotFoundError(f"model card not found: {card_id}") from exc
        return self.ensure_checkpoint(
            spec,
            environment_id=card.usage.recommended_environment.environment_id,
        )

    def ensure_checkpoint(
        self,
        specification: CheckpointSpec,
        *,
        environment_id: str | None = None,
        acquisition_policy: CheckpointAcquisitionPolicy | None = None,
    ) -> ResolvedCheckpoint:
        """Acquire one explicit checkpoint specification without requiring a model card.

        Parameters
        ----------
        specification
            Strict checkpoint source and integrity contract.
        environment_id
            Isolated environment required only by package-bundled sources.
        acquisition_policy
            Optional explicit authority, network, authentication, and byte-limit policy.

        Returns
        -------
        ResolvedCheckpoint
            Integrity-checked content-addressed local checkpoint.
        """

        spec = CheckpointSpec.model_validate(specification)
        original_sink = self._report_sink
        original_executor = self.executor
        self._report_sink = RuntimeReportSink(
            self.runtime_root,
            "reports",
            "checkpoints",
            spec.checkpoint_id,
        )
        self.executor = original_executor.with_report_sink(self._report_sink)
        try:
            policy = acquisition_policy or CheckpointAcquisitionPolicy(
                allow_network=not self.policy.offline,
                allow_authentication=True,
            )
            return self._ensure_spec(environment_id, spec, policy)
        finally:
            self._report_sink = original_sink
            self.executor = original_executor

    def _ensure_spec(
        self,
        environment_id: str | None,
        spec: CheckpointSpec,
        acquisition_policy: CheckpointAcquisitionPolicy,
    ) -> ResolvedCheckpoint:
        """Resolve a loaded checkpoint spec with acquisition reporting enabled."""

        self._validate_acquisition_policy(spec, acquisition_policy)
        cached = self._cached(spec)
        if cached is not None:
            return cached
        if (self.policy.offline or not acquisition_policy.allow_network) and spec.source_type in {
            CheckpointSourceType.HTTPS,
            CheckpointSourceType.GITHUB_RELEASE,
            CheckpointSourceType.HUGGINGFACE,
        }:
            self._record_checkpoint_event(
                "offline-cache-lookup",
                spec,
                "failed",
                location=remote_source_location(spec),
                failure_classification="offline_cache_miss",
                failure_detail=f"offline checkpoint cache miss for {spec.checkpoint_id}",
            )
            raise OfflineResourceUnavailableError(f"checkpoint cache miss for {spec.checkpoint_id}")
        match spec.source_type:
            case CheckpointSourceType.LOCAL_PATH:
                return self._from_local_path(spec)
            case CheckpointSourceType.HTTPS:
                resolution = (
                    self.resolve_checkpoint_authority(spec) if spec.authority is not None else None
                )
                url = str(resolution.resolved_payload_url) if resolution else str(spec.url)
                return self._from_remote(
                    spec,
                    url,
                    {},
                    "https",
                    acquisition_policy=acquisition_policy,
                    authority_resolution=resolution,
                )
            case CheckpointSourceType.GITHUB_RELEASE:
                return self._from_remote(
                    spec,
                    github_release_url(spec),
                    optional_auth_header("GITHUB_TOKEN")
                    if acquisition_policy.allow_authentication
                    else {},
                    "github_release",
                    acquisition_policy=acquisition_policy,
                )
            case CheckpointSourceType.HUGGINGFACE:
                return self._from_remote(
                    spec,
                    huggingface_url(spec),
                    (optional_auth_header("HF_TOKEN") or optional_auth_header("HUGGINGFACE_TOKEN"))
                    if acquisition_policy.allow_authentication
                    else {},
                    "huggingface",
                    acquisition_policy=acquisition_policy,
                )
            case CheckpointSourceType.PACKAGE_BUNDLE:
                if environment_id is None:
                    raise CheckpointAcquisitionError(
                        "package-bundle checkpoint acquisition requires environment_id"
                    )
                return self._from_package_bundle(environment_id, spec)

    @staticmethod
    def _validate_acquisition_policy(
        spec: CheckpointSpec,
        policy: CheckpointAcquisitionPolicy,
    ) -> None:
        if policy.require_authority and spec.authority is None:
            raise CheckpointAcquisitionError("checkpoint acquisition policy requires authority")
        if policy.require_expected_sha256 and spec.expected_sha256 is None:
            raise CheckpointAcquisitionError(
                "checkpoint acquisition policy requires expected SHA-256"
            )
        if policy.require_exact_size and (
            spec.authority is None or spec.authority.expected_size_bytes <= 0
        ):
            raise CheckpointAcquisitionError("checkpoint acquisition policy requires exact size")
        if policy.require_published_checksums and (
            spec.authority is None or not spec.authority.published_checksums
        ):
            raise CheckpointAcquisitionError(
                "checkpoint acquisition policy requires published checksums"
            )

    def resolve_checkpoint_authority(
        self,
        specification: CheckpointSpec,
    ) -> CheckpointAuthorityResolution:
        """Resolve or reuse bounded authoritative metadata without acquiring payload bytes.

        Parameters
        ----------
        specification
            Strict authority-complete checkpoint specification.

        Returns
        -------
        CheckpointAuthorityResolution
            Network or managed-cache metadata provenance.
        """

        spec = CheckpointSpec.model_validate(specification)
        if spec.authority is None:
            raise CheckpointAuthorityResolutionError(
                "checkpoint specification has no structured authority"
            )
        path = self._authority_cache_path(spec)
        cached = self._load_cached_authority_resolution(spec, path)
        if cached is not None:
            return cached.model_copy(update={"metadata_source": "managed_cache"})
        if self.policy.offline:
            raise OfflineResourceUnavailableError(
                f"checkpoint authority metadata cache miss for {spec.checkpoint_id}"
            )
        try:
            resolved = resolve_checkpoint_authority(
                spec,
                transport=self.transport,
                timeout=self.policy.download_timeout_seconds,
            )
            write_json_atomic(path, resolved)
        except OSError as exc:
            raise CheckpointAuthorityResolutionError(
                f"checkpoint authority cache write failed: {sanitize_text(str(exc))}"
            ) from exc
        return resolved

    def _authority_cache_path(self, spec: CheckpointSpec) -> Path:
        return contained_path(
            self.runtime_root / "checkpoints",
            spec.checkpoint_id,
            ".authority",
            checkpoint_specification_fingerprint(spec),
            "checkpoint-authority-resolution.json",
        )

    def _load_cached_authority_resolution(
        self,
        spec: CheckpointSpec,
        path: Path,
    ) -> CheckpointAuthorityResolution | None:
        if not path.is_file() or spec.authority is None:
            return None
        try:
            result = CheckpointAuthorityResolution.model_validate_json(path.read_text())
        except (OSError, ValueError):
            return None
        authority = spec.authority
        if (
            result.provider != authority.provider
            or result.record_id != authority.record_id
            or result.requested_filename != authority.filename
            or result.expected_size_bytes != authority.expected_size_bytes
            or result.published_checksums
            != tuple(sorted(authority.published_checksums, key=lambda item: item.algorithm.value))
        ):
            return None
        return result

    def info(self, card_id: str) -> dict[str, object]:
        """Inspect a checkpoint specification and local cache without acquisition.

        Parameters
        ----------
        card_id
            Canonical model-card identifier.

        Returns
        -------
        dict
            Card and checkpoint identifiers, source type, declared hashes, and cache entries with
            path, digest, and validity.

        Raises
        ------
        CheckpointNotFoundError
            If ``card_id`` is invalid or absent.

        Notes
        -----
        The method reads and hashes local cache files but performs no download or model import.
        """

        from torch_dae.core.registry import ModelCardRegistry

        try:
            ensure_canonical_id(card_id)
        except ValueError as exc:
            raise CheckpointNotFoundError(f"invalid checkpoint card ID: {card_id}") from exc
        try:
            spec = ModelCardRegistry(self.repository_root).get_card(card_id).checkpoint
        except KeyError as exc:
            raise CheckpointNotFoundError(f"model card not found: {card_id}") from exc
        data = self.info_checkpoint(spec)
        data["card_id"] = card_id
        return data

    def info_checkpoint(self, specification: CheckpointSpec) -> dict[str, object]:
        """Inspect an explicit specification and its managed caches without network access.

        Parameters
        ----------
        specification
            Strict checkpoint specification to inspect.

        Returns
        -------
        dict
            Specification fingerprint, authority state, and validated cache inventory.
        """

        spec = CheckpointSpec.model_validate(specification)
        root = contained_path(self.runtime_root / "checkpoints", spec.checkpoint_id)
        entries: list[dict[str, object]] = []
        if root.exists():
            for child in sorted(root.iterdir()):
                if child.is_dir():
                    validity = self._validate_cache_dir(spec, child)
                    entries.append(
                        {
                            "sha256": child.name,
                            "cache_identity": f"{spec.checkpoint_id}:{child.name}",
                            "runtime_path": str(child),
                            "valid": validity is not None,
                        }
                    )
        authority_cache = None
        if spec.authority is not None:
            authority_path = self._authority_cache_path(spec)
            cached_authority = self._load_cached_authority_resolution(spec, authority_path)
            authority_cache = {
                "cached": cached_authority is not None,
                "reference": (
                    str(authority_path.relative_to(self.runtime_root))
                    if authority_path.is_file()
                    else None
                ),
                "metadata_response_sha256": (
                    cached_authority.metadata_response_sha256
                    if cached_authority is not None
                    else None
                ),
            }
        return {
            "checkpoint_id": spec.checkpoint_id,
            "source_type": spec.source_type.value,
            "expected_sha256": spec.expected_sha256,
            "observed_sha256": spec.observed_sha256,
            "specification_fingerprint": checkpoint_specification_fingerprint(spec),
            "authority": spec.authority.model_dump(mode="json") if spec.authority else None,
            "authority_cache": authority_cache,
            "cached": entries,
        }

    def remove(self, card_id: str) -> None:
        """Remove runtime cache state for one card's checkpoint.

        Parameters
        ----------
        card_id
            Canonical model-card identifier.

        Returns
        -------
        None
            The matching runtime cache tree is absent on return.

        Raises
        ------
        CheckpointNotFoundError
            If ``card_id`` is invalid or absent.

        Notes
        -----
        Removal is idempotent. It never edits the model card, source asset, environment, or
        committed metadata.
        """

        from torch_dae.core.registry import ModelCardRegistry

        try:
            ensure_canonical_id(card_id)
        except ValueError as exc:
            raise CheckpointNotFoundError(f"invalid checkpoint card ID: {card_id}") from exc
        try:
            spec = ModelCardRegistry(self.repository_root).get_card(card_id).checkpoint
        except KeyError as exc:
            raise CheckpointNotFoundError(f"model card not found: {card_id}") from exc
        root = contained_path(self.runtime_root / "checkpoints", spec.checkpoint_id)
        if root.exists():
            shutil.rmtree(root)

    def _cached(self, spec: CheckpointSpec) -> ResolvedCheckpoint | None:
        root = contained_path(self.runtime_root / "checkpoints", spec.checkpoint_id)
        if not root.exists():
            return None
        for child in sorted(root.iterdir()):
            if child.is_dir():
                cached = self._validate_cache_dir(spec, child)
                if cached is not None:
                    return cached
        return None

    def _validate_cache_dir(
        self, spec: CheckpointSpec, directory: Path
    ) -> ResolvedCheckpoint | None:
        try:
            validate_sha256(directory.name)
        except Exception:
            return None
        metadata_path = directory / "checkpoint-materialization.json"
        if not metadata_path.exists():
            return None
        try:
            metadata = CheckpointMaterializationRecord.model_validate_json(
                metadata_path.read_text()
            )
        except Exception:
            return None
        if metadata.checkpoint_id != spec.checkpoint_id or metadata.sha256 != directory.name:
            return None
        if metadata.specification_fingerprint != checkpoint_specification_fingerprint(spec):
            return None
        file_path = directory / metadata.filename
        if not file_path.is_file():
            return None
        try:
            actual = sha256_file(file_path)
        except OSError:
            return None
        hashes = {directory.name, metadata.sha256, actual}
        for value in (
            spec.expected_sha256,
            spec.observed_sha256,
            metadata.expected_sha256,
            metadata.observed_sha256,
        ):
            if value:
                hashes.add(value)
        if len(hashes) != 1:
            return None
        if spec.authority is not None:
            if metadata.schema_version != "2.0.0" or metadata.authority != spec.authority:
                return None
            try:
                actual_size = file_path.stat().st_size
            except OSError:
                return None
            if (
                actual_size != spec.authority.expected_size_bytes
                or metadata.expected_size_bytes != spec.authority.expected_size_bytes
                or metadata.observed_size_bytes != actual_size
                or metadata.size_bytes != actual_size
            ):
                return None
            algorithms = {item.algorithm for item in spec.authority.published_checksums}
            algorithms.add(ChecksumAlgorithm.SHA256)
            digesters = {algorithm: hashlib.new(algorithm.value) for algorithm in algorithms}
            try:
                with file_path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        for digest in digesters.values():
                            digest.update(chunk)
            except OSError:
                return None
            observed = {algorithm: digest.hexdigest() for algorithm, digest in digesters.items()}
            if observed[ChecksumAlgorithm.SHA256] != actual:
                return None
            for published in spec.authority.published_checksums:
                if observed.get(published.algorithm) != published.digest:
                    return None
            if (
                metadata.metadata_resolution_reference is None
                or metadata.metadata_resolution_sha256 is None
            ):
                return None
            try:
                ensure_repository_relative(metadata.metadata_resolution_reference)
                resolution_path = contained_path(
                    self.runtime_root, metadata.metadata_resolution_reference
                )
                if sha256_file(resolution_path) != metadata.metadata_resolution_sha256:
                    return None
                resolution = CheckpointAuthorityResolution.model_validate_json(
                    resolution_path.read_text()
                )
            except (OSError, ValueError):
                return None
            if (
                resolution.record_id != spec.authority.record_id
                or resolution.resolved_filename != spec.authority.filename
                or resolution.expected_size_bytes != spec.authority.expected_size_bytes
                or resolution.published_checksums
                != tuple(
                    sorted(
                        spec.authority.published_checksums,
                        key=lambda item: item.algorithm.value,
                    )
                )
            ):
                return None
        return ResolvedCheckpoint(
            checkpoint_id=spec.checkpoint_id,
            sha256=actual,
            path=file_path,
            immutable=True,
        )

    def _from_local_path(self, spec: CheckpointSpec) -> ResolvedCheckpoint:
        if spec.local_path is None:
            raise CheckpointAcquisitionError("local_path checkpoint missing local_path")
        source = contained_path(self.repository_root, spec.local_path)
        if not source.is_file():
            self._record_checkpoint_event(
                "local-path-copy",
                spec,
                "failed",
                location=str(source),
                failure_classification="missing_source",
            )
            raise CheckpointNotFoundError(f"local checkpoint file not found: {source}")
        try:
            source.stat()
        except OSError as exc:
            self._record_checkpoint_event(
                "local-path-copy",
                spec,
                "failed",
                location=str(source),
                failure_classification=type(exc).__name__,
                failure_detail=str(exc),
            )
            raise CheckpointAcquisitionError(
                f"local checkpoint is unreadable: {sanitize_text(str(source))}"
            ) from exc
        return self._install_file(
            spec,
            source,
            source.name,
            str(source),
            "local_path",
            acquisition_operation="local-path-copy",
        )

    def _from_remote(
        self,
        spec: CheckpointSpec,
        url: str,
        headers: Mapping[str, str],
        source_description: str,
        *,
        acquisition_policy: CheckpointAcquisitionPolicy,
        authority_resolution: CheckpointAuthorityResolution | None = None,
    ) -> ResolvedCheckpoint:
        if not url.startswith("https://"):
            raise CheckpointAcquisitionError("remote checkpoint URLs must use HTTPS")
        if authority_resolution is not None:
            _validate_zenodo_url(url, label="payload")
        downloads = contained_path(self.runtime_root / "checkpoints", ".downloads")
        tmp = downloads / f"{spec.checkpoint_id}.{uuid.uuid4().hex}.download"
        response: TransportResponse | None = None
        acquisition_error: CheckpointAcquisitionError | None = None
        try:
            try:
                response = self.transport.open(
                    url,
                    headers=headers,
                    timeout=self.policy.download_timeout_seconds,
                )
            except CheckpointAcquisitionError as exc:
                self._record_checkpoint_event(
                    "remote-open",
                    spec,
                    "failed",
                    location=url,
                    source_description=source_description,
                    failure_classification=checkpoint_failure_classification(exc),
                    failure_detail=str(exc),
                )
                raise
            except OSError as exc:
                self._record_checkpoint_event(
                    "remote-open",
                    spec,
                    "failed",
                    location=url,
                    source_description=source_description,
                    failure_classification=type(exc).__name__,
                    failure_detail=str(exc),
                )
                raise CheckpointAcquisitionError(
                    f"checkpoint transport failed: {sanitize_text(str(exc))}"
                ) from exc
            self._record_checkpoint_event(
                "remote-open",
                spec,
                "success",
                location=url,
                source_description=source_description,
            )
            if response.status_code < 200 or response.status_code >= 400:
                self._record_checkpoint_event(
                    "remote-open",
                    spec,
                    "failed",
                    location=url,
                    source_description=source_description,
                    failure_classification=f"http_{response.status_code}",
                    failure_detail=f"download returned HTTP {response.status_code}",
                )
                message = f"download returned HTTP {response.status_code}"
                if response.status_code == 404:
                    raise CheckpointNotFoundError(message)
                raise CheckpointAcquisitionError(message)
            effective_url = response.effective_url or url
            if not effective_url.startswith("https://"):
                raise CheckpointAcquisitionError(
                    "checkpoint redirect resolved to an unsupported scheme"
                )
            if authority_resolution is not None:
                _validate_zenodo_url(effective_url, label="payload response")
            filename = spec.filename or Path(url).name or spec.checkpoint_id
            algorithms = {ChecksumAlgorithm.SHA256}
            if authority_resolution is not None:
                algorithms.update(
                    item.algorithm for item in authority_resolution.published_checksums
                )
            digests = {
                algorithm: hashlib.new(algorithm.value)
                for algorithm in sorted(algorithms, key=lambda item: item.value)
            }
            size = 0
            try:
                downloads.mkdir(parents=True, exist_ok=True)
                with tmp.open("wb") as handle:
                    for chunk in iter(lambda: response.body.read(1024 * 1024), b""):
                        size += len(chunk)
                        if (
                            acquisition_policy.maximum_bytes is not None
                            and size > acquisition_policy.maximum_bytes
                        ):
                            self._record_checkpoint_event(
                                "remote-stream",
                                spec,
                                "failed",
                                location=url,
                                source_description=source_description,
                                byte_count=size,
                                failure_classification="oversized_response",
                                failure_detail="checkpoint response exceeded maximum_bytes",
                            )
                            raise CheckpointResponseTooLargeError(
                                "checkpoint response exceeded maximum_bytes"
                            )
                        for digest in digests.values():
                            digest.update(chunk)
                        handle.write(chunk)
            except OSError as exc:
                self._record_checkpoint_event(
                    "remote-stream",
                    spec,
                    "failed",
                    location=url,
                    source_description=source_description,
                    byte_count=size,
                    failure_classification=type(exc).__name__,
                    failure_detail=str(exc),
                )
                raise CheckpointAcquisitionError(
                    f"checkpoint download stream failed: {sanitize_text(str(exc))}"
                ) from exc
            observed = tuple(
                ObservedChecksum(algorithm=algorithm, digest=digest.hexdigest())
                for algorithm, digest in digests.items()
            )
            observed_by_algorithm = {item.algorithm: item.digest for item in observed}
            sha = observed_by_algorithm[ChecksumAlgorithm.SHA256]
            self._record_checkpoint_event(
                "remote-stream",
                spec,
                "success",
                location=url,
                source_description=source_description,
                byte_count=size,
                sha256=sha,
            )
            if authority_resolution is not None:
                expected_size = authority_resolution.expected_size_bytes
                if size != expected_size:
                    declared_length = _header(response.headers, "Content-Length")
                    truncated = False
                    if declared_length is not None:
                        try:
                            truncated = size < expected_size <= int(declared_length)
                        except ValueError:
                            truncated = False
                    classification = "truncated_transfer" if truncated else "exact_size_mismatch"
                    self._record_checkpoint_event(
                        "size-validation",
                        spec,
                        "failed",
                        location=url,
                        source_description=source_description,
                        byte_count=size,
                        sha256=sha,
                        failure_classification=classification,
                        failure_detail=(
                            f"checkpoint exact size mismatch: expected {expected_size}, got {size}"
                        ),
                    )
                    raise CheckpointSizeMismatchError(
                        f"checkpoint exact size mismatch: expected {expected_size}, got {size}"
                    )
                self._record_checkpoint_event(
                    "size-validation",
                    spec,
                    "success",
                    location=url,
                    source_description=source_description,
                    byte_count=size,
                    sha256=sha,
                    report_extra={"expected_size_bytes": expected_size},
                )
                for published in authority_resolution.published_checksums:
                    actual = observed_by_algorithm[published.algorithm]
                    if actual != published.digest:
                        self._record_checkpoint_event(
                            "published-checksum-validation",
                            spec,
                            "failed",
                            location=url,
                            source_description=source_description,
                            byte_count=size,
                            sha256=sha,
                            failure_classification="published_checksum_mismatch",
                            failure_detail=(
                                f"published {published.algorithm.value} checksum mismatch"
                            ),
                            report_extra={
                                "algorithm": published.algorithm.value,
                                "published_digest": published.digest,
                                "observed_digest": actual,
                            },
                        )
                        raise CheckpointPublishedChecksumMismatchError(
                            f"published {published.algorithm.value} checkpoint checksum mismatch"
                        )
                self._record_checkpoint_event(
                    "published-checksum-validation",
                    spec,
                    "success",
                    location=url,
                    source_description=source_description,
                    byte_count=size,
                    sha256=sha,
                )
            self._validate_checkpoint_hashes(spec, sha, url, source_description)
            return self._finalize_tmp(
                spec,
                tmp,
                filename,
                effective_url,
                source_description,
                sha,
                size,
                authority_resolution=authority_resolution,
                observed_checksums=observed,
                payload_etag=_header(response.headers, "ETag"),
                payload_last_modified=_header(response.headers, "Last-Modified"),
            )
        except CheckpointAcquisitionError as exc:
            acquisition_error = exc
            self._cleanup_path(tmp, spec, source_description)
            raise
        finally:
            if response is not None:
                try:
                    response.body.close()
                except OSError as exc:
                    self._record_checkpoint_event(
                        "response-close",
                        spec,
                        "failed",
                        location=url,
                        source_description=source_description,
                        failure_classification=type(exc).__name__,
                        failure_detail=str(exc),
                    )
                    if acquisition_error is None:
                        raise CheckpointAcquisitionError(
                            f"checkpoint response close failed: {sanitize_text(str(exc))}"
                        ) from exc

    def _from_package_bundle(
        self,
        environment_id: str,
        spec: CheckpointSpec,
    ) -> ResolvedCheckpoint:
        from torch_dae.environment.manager import EnvironmentManager

        manager = EnvironmentManager(
            self.repository_root,
            policy=self.policy,
            executor=self.executor,
        )
        manager.materialize_environment(environment_id)
        manager.verify_environment(
            environment_id,
            expected_fingerprint=manager.resolve_environment(
                environment_id
            ).environment_fingerprint,
        )
        resolved = manager.resolved_environment(environment_id)
        if spec.package is None or spec.package_version is None or spec.filename is None:
            raise CheckpointAcquisitionError("package bundle checkpoint is incomplete")
        code = """
import importlib.metadata as metadata
import json
import sys
from pathlib import Path
package, version, filename = sys.argv[1:4]
dist = metadata.distribution(package)
if dist.version != version:
    raise SystemExit(f"version mismatch: {dist.version}")
root = Path(str(dist.locate_file(""))).resolve()
target = Path(str(dist.locate_file(filename))).resolve()
files = dist.files or ()
members = {str(item).replace("\\\\", "/") for item in files}
if filename.replace("\\\\", "/") not in members:
    raise SystemExit("resource is not owned by distribution")
target.relative_to(root)
print(json.dumps({"path": str(target), "root": str(root)}))
""".strip()
        try:
            result = self.executor.run(
                [
                    str(resolved.python_executable),
                    "-c",
                    code,
                    spec.package,
                    spec.package_version,
                    spec.filename,
                ],
                operation="package-bundle-lookup",
                cwd=self.repository_root,
                timeout=self.policy.command_timeout_seconds,
                env_remove=python_env_remove(),
                check=True,
            )
        except ExternalCommandError as exc:
            self._record_checkpoint_event(
                "package-bundle-lookup",
                spec,
                "failed",
                location=f"{spec.package}:{spec.filename}",
                failure_classification="lookup_failed",
            )
            raise CheckpointNotFoundError(
                f"package bundle checkpoint unavailable: {spec.package}:{spec.filename}"
            ) from exc
        try:
            data = json.loads(result.stdout)
            source = Path(data["path"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            self._record_checkpoint_event(
                "package-bundle-lookup",
                spec,
                "failed",
                location=f"{spec.package}:{spec.filename}",
                failure_classification=type(exc).__name__,
                failure_detail=str(exc),
            )
            raise CheckpointAcquisitionError(
                f"package bundle lookup returned invalid metadata: {spec.package}:{spec.filename}"
            ) from exc
        if not source.is_file():
            self._record_checkpoint_event(
                "package-bundle-lookup",
                spec,
                "failed",
                location=str(source),
                failure_classification="missing_source",
            )
            raise CheckpointNotFoundError(f"package checkpoint file not found: {source}")
        try:
            size = source.stat().st_size
            sha = sha256_file(source)
        except OSError as exc:
            self._record_checkpoint_event(
                "package-bundle-lookup",
                spec,
                "failed",
                location=str(source),
                failure_classification=type(exc).__name__,
                failure_detail=str(exc),
            )
            raise CheckpointAcquisitionError(
                f"package checkpoint is unreadable: {sanitize_text(str(source))}"
            ) from exc
        self._record_checkpoint_event(
            "package-bundle-lookup",
            spec,
            "success",
            location=str(source),
            byte_count=size,
            sha256=sha,
        )
        return self._install_file(
            spec,
            source,
            Path(spec.filename).name,
            str(source),
            "package_bundle",
            environment_id=resolved.environment_id,
            acquisition_operation="package-bundle-copy",
        )

    def _install_file(
        self,
        spec: CheckpointSpec,
        source: Path,
        filename: str,
        location: str,
        source_description: str,
        *,
        environment_id: str | None = None,
        acquisition_operation: str,
    ) -> ResolvedCheckpoint:
        try:
            sha = sha256_file(source)
        except OSError as exc:
            self._record_checkpoint_event(
                acquisition_operation,
                spec,
                "failed",
                location=location,
                source_description=source_description,
                failure_classification=type(exc).__name__,
                failure_detail=str(exc),
            )
            raise CheckpointAcquisitionError(
                f"checkpoint hash calculation failed: {sanitize_text(str(source))}"
            ) from exc
        self._validate_checkpoint_hashes(spec, sha, location, source_description)
        return self._copy_to_cache(
            spec,
            source,
            filename,
            location,
            source_description,
            sha,
            environment_id=environment_id,
            acquisition_operation=acquisition_operation,
        )

    def _finalize_tmp(
        self,
        spec: CheckpointSpec,
        tmp: Path,
        filename: str,
        location: str,
        source_description: str,
        sha: str,
        size: int,
        *,
        authority_resolution: CheckpointAuthorityResolution | None = None,
        observed_checksums: tuple[ObservedChecksum, ...] = (),
        payload_etag: str | None = None,
        payload_last_modified: str | None = None,
    ) -> ResolvedCheckpoint:
        cache_dir = checkpoint_cache_path(self.runtime_root, spec.checkpoint_id, sha)
        final = contained_path(cache_dir, Path(filename).name)
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._replace_file(tmp, final)
        except OSError as exc:
            self._record_checkpoint_event(
                "remote-finalize",
                spec,
                "failed",
                location=location,
                source_description=source_description,
                byte_count=size,
                sha256=sha,
                failure_classification=type(exc).__name__,
                failure_detail=str(exc),
            )
            self._cleanup_path(tmp, spec, source_description)
            raise CheckpointAcquisitionError(
                f"checkpoint cache finalization failed: {sanitize_text(str(exc))}"
            ) from exc
        self._record_checkpoint_event(
            "remote-finalize",
            spec,
            "success",
            location=location,
            source_description=source_description,
            byte_count=size,
            sha256=sha,
        )
        self._write_metadata_or_cleanup(
            spec,
            final,
            location,
            source_description,
            sha,
            size,
            authority_resolution=authority_resolution,
            observed_checksums=observed_checksums,
            payload_etag=payload_etag,
            payload_last_modified=payload_last_modified,
            materialization_status="downloaded",
        )
        return ResolvedCheckpoint(
            checkpoint_id=spec.checkpoint_id,
            sha256=sha,
            path=final,
            immutable=True,
        )

    def _copy_to_cache(
        self,
        spec: CheckpointSpec,
        source: Path,
        filename: str,
        location: str,
        source_description: str,
        sha: str,
        *,
        environment_id: str | None = None,
        acquisition_operation: str,
    ) -> ResolvedCheckpoint:
        cache_dir = checkpoint_cache_path(self.runtime_root, spec.checkpoint_id, sha)
        final = contained_path(cache_dir, Path(filename).name)
        tmp = final.with_name(f".{final.name}.{uuid.uuid4().hex}.tmp")
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, tmp)
        except (OSError, shutil.Error) as exc:
            self._record_checkpoint_event(
                acquisition_operation,
                spec,
                "failed",
                location=location,
                source_description=source_description,
                failure_classification=type(exc).__name__,
                failure_detail=str(exc),
            )
            self._cleanup_path(tmp, spec, source_description)
            raise CheckpointAcquisitionError(
                f"checkpoint copy failed: {sanitize_text(str(exc))}"
            ) from exc
        try:
            copied_size = tmp.stat().st_size
        except OSError as exc:
            self._record_checkpoint_event(
                acquisition_operation,
                spec,
                "failed",
                location=str(tmp),
                source_description=source_description,
                failure_classification=type(exc).__name__,
                failure_detail=str(exc),
            )
            self._cleanup_path(tmp, spec, source_description)
            raise CheckpointAcquisitionError(
                f"checkpoint temporary copy is unreadable: {sanitize_text(str(exc))}"
            ) from exc
        self._record_checkpoint_event(
            acquisition_operation,
            spec,
            "success",
            location=location,
            source_description=source_description,
            byte_count=copied_size,
            sha256=sha,
        )
        try:
            self._replace_file(tmp, final)
            final_size = final.stat().st_size
        except OSError as exc:
            self._record_checkpoint_event(
                "cache-finalize",
                spec,
                "failed",
                location=location,
                source_description=source_description,
                byte_count=copied_size,
                sha256=sha,
                failure_classification=type(exc).__name__,
                failure_detail=str(exc),
            )
            self._cleanup_path(tmp, spec, source_description)
            raise CheckpointAcquisitionError(
                f"checkpoint cache finalization failed: {sanitize_text(str(exc))}"
            ) from exc
        self._record_checkpoint_event(
            "cache-finalize",
            spec,
            "success",
            location=location,
            source_description=source_description,
            byte_count=final_size,
            sha256=sha,
        )
        self._write_metadata_or_cleanup(
            spec,
            final,
            location,
            source_description,
            sha,
            final_size,
            environment_id=environment_id,
        )
        return ResolvedCheckpoint(
            checkpoint_id=spec.checkpoint_id,
            sha256=sha,
            path=final,
            immutable=True,
        )

    def _validate_checkpoint_hashes(
        self,
        spec: CheckpointSpec,
        actual_sha256: str,
        location: str,
        source_description: str,
    ) -> None:
        for label, value in (
            ("expected", spec.expected_sha256),
            ("observed", spec.observed_sha256),
        ):
            if value is not None and value != actual_sha256:
                classification = (
                    "expected_hash_mismatch" if label == "expected" else "observed_hash_mismatch"
                )
                self._record_checkpoint_event(
                    "hash-validation",
                    spec,
                    "failed",
                    location=location,
                    source_description=source_description,
                    sha256=actual_sha256,
                    failure_classification=classification,
                    failure_detail=(
                        f"{label} checkpoint hash mismatch: expected {value}, got {actual_sha256}"
                    ),
                    report_extra={
                        f"{label}_sha256": value,
                        "actual_sha256": actual_sha256,
                    },
                )
                raise CheckpointHashMismatchError(
                    f"{label} checkpoint hash mismatch: expected {value}, got {actual_sha256}"
                )
        self._record_checkpoint_event(
            "hash-validation",
            spec,
            "success",
            location=location,
            source_description=source_description,
            sha256=actual_sha256,
            report_extra={
                "expected_sha256": spec.expected_sha256,
                "observed_sha256": spec.observed_sha256,
                "actual_sha256": actual_sha256,
            },
        )

    def _write_metadata_or_cleanup(
        self,
        spec: CheckpointSpec,
        final: Path,
        location: str,
        source_description: str,
        sha: str,
        size: int,
        *,
        environment_id: str | None = None,
        authority_resolution: CheckpointAuthorityResolution | None = None,
        observed_checksums: tuple[ObservedChecksum, ...] = (),
        payload_etag: str | None = None,
        payload_last_modified: str | None = None,
        materialization_status: Literal["downloaded", "local_copy", "package_bundle"] | None = None,
    ) -> None:
        try:
            self._write_metadata(
                spec,
                final,
                location,
                source_description,
                sha,
                size,
                environment_id=environment_id,
                authority_resolution=authority_resolution,
                observed_checksums=observed_checksums,
                payload_etag=payload_etag,
                payload_last_modified=payload_last_modified,
                materialization_status=materialization_status,
            )
        except OSError as exc:
            self._record_checkpoint_event(
                "metadata-write",
                spec,
                "failed",
                location=location,
                source_description=source_description,
                byte_count=size,
                sha256=sha,
                failure_classification=type(exc).__name__,
                failure_detail=str(exc),
            )
            self._cleanup_path(final, spec, source_description)
            self._cleanup_path(final.parent, spec, source_description)
            raise CheckpointAcquisitionError(
                f"checkpoint metadata write failed: {sanitize_text(str(exc))}"
            ) from exc

    def _replace_file(self, source: Path, target: Path) -> None:
        os.replace(source, target)

    def _cleanup_path(
        self,
        path: Path,
        spec: CheckpointSpec,
        source_description: str,
    ) -> None:
        if not path.exists():
            return
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except OSError as exc:
            self._record_checkpoint_event(
                "failure-cleanup",
                spec,
                "failed",
                location=str(path),
                source_description=source_description,
                failure_classification=type(exc).__name__,
                failure_detail=str(exc),
            )
            return
        self._record_checkpoint_event(
            "failure-cleanup",
            spec,
            "success",
            location=str(path),
            source_description=source_description,
        )

    def _write_metadata(
        self,
        spec: CheckpointSpec,
        final: Path,
        location: str,
        source_description: str,
        sha: str,
        size: int,
        *,
        environment_id: str | None = None,
        authority_resolution: CheckpointAuthorityResolution | None = None,
        observed_checksums: tuple[ObservedChecksum, ...] = (),
        payload_etag: str | None = None,
        payload_last_modified: str | None = None,
        materialization_status: Literal["downloaded", "local_copy", "package_bundle"] | None = None,
    ) -> None:
        authority_path = self._authority_cache_path(spec) if spec.authority is not None else None
        metadata_reference: str | None = None
        metadata_sha256: str | None = None
        if authority_resolution is not None and authority_path is not None:
            if not authority_path.is_file():
                raise OSError("checkpoint authority resolution cache is missing")
            metadata_reference = str(authority_path.relative_to(self.runtime_root))
            metadata_sha256 = sha256_file(authority_path)
        metadata = CheckpointMaterializationRecord(
            schema_version="2.0.0" if spec.authority is not None else "1.0.0",
            checkpoint_id=spec.checkpoint_id,
            source_type=spec.source_type,
            source_description=source_description,
            resolved_url_or_location=location,
            filename=final.name,
            sha256=sha,
            size_bytes=size,
            acquired_at=utc_now(),
            cache_path=str(final.parent.relative_to(self.runtime_root)),
            expected_sha256=spec.expected_sha256,
            observed_sha256=spec.observed_sha256 or sha,
            environment_id=environment_id,
            specification_fingerprint=checkpoint_specification_fingerprint(spec),
            command_log_references=tuple(
                self._report_sink.references if self._report_sink is not None else ()
            ),
            authority=spec.authority,
            expected_size_bytes=(
                authority_resolution.expected_size_bytes
                if authority_resolution is not None
                else None
            ),
            observed_size_bytes=size if authority_resolution is not None else None,
            published_checksums=(
                authority_resolution.published_checksums if authority_resolution is not None else ()
            ),
            observed_checksums=observed_checksums,
            metadata_resolution_reference=metadata_reference,
            metadata_resolution_sha256=metadata_sha256,
            payload_etag=payload_etag,
            payload_last_modified=payload_last_modified,
            materialization_status=materialization_status,
        )
        write_json_atomic(final.parent / "checkpoint-materialization.json", metadata)

    def _record_checkpoint_event(
        self,
        operation: str,
        spec: CheckpointSpec,
        status: Literal["success", "failed", "timeout", "unavailable"],
        *,
        location: str,
        source_description: str | None = None,
        byte_count: int | None = None,
        sha256: str | None = None,
        failure_classification: str | None = None,
        failure_detail: str | None = None,
        report_extra: Mapping[str, object] | None = None,
    ) -> None:
        if self._report_sink is None:
            return
        payload_extra: dict[str, object] = {
            "checkpoint_id": spec.checkpoint_id,
            "source_type": spec.source_type.value,
            "source_description": source_description or spec.source_type.value,
            "sanitized_source": location,
            "result_status": status,
            "byte_count": byte_count,
            "sha256": sha256,
            "failure_classification": failure_classification,
            "failure_detail": failure_detail,
        }
        if report_extra:
            payload_extra.update(report_extra)
        self._report_sink.record_event(
            operation=operation,
            status=status,
            arguments=(location,),
            working_directory=str(self.repository_root),
            return_code=0 if status == "success" else 1,
            stderr=failure_detail or "",
            extra=payload_extra,
        )


def validate_checkpoint_hashes(spec: CheckpointSpec, actual_sha256: str) -> None:
    """Enforce expected and observed checkpoint hashes against `actual_sha256`."""

    for label, value in (("expected", spec.expected_sha256), ("observed", spec.observed_sha256)):
        if value is not None and value != actual_sha256:
            raise CheckpointHashMismatchError(
                f"{label} checkpoint hash mismatch: expected {value}, got {actual_sha256}"
            )


def checkpoint_failure_classification(exc: BaseException) -> str:
    """Return the underlying operational class when a typed wrapper has one."""

    if exc.__cause__ is not None:
        return type(exc.__cause__).__name__
    return type(exc).__name__


def remote_source_location(spec: CheckpointSpec) -> str:
    """Return the source location string without performing network construction."""

    if spec.url is not None:
        return str(spec.url)
    if spec.source_type == CheckpointSourceType.GITHUB_RELEASE:
        return (
            f"github_release:{spec.repository_id}:{spec.release_tag}:"
            f"{spec.filename or spec.checkpoint_id}"
        )
    if spec.source_type == CheckpointSourceType.HUGGINGFACE:
        return f"huggingface:{spec.repository_id}:{spec.revision or 'main'}:{spec.filename}"
    return spec.source_type.value


def github_release_url(spec: CheckpointSpec) -> str:
    """Construct a canonical GitHub release asset URL."""

    if spec.repository_id is None or spec.release_tag is None or spec.filename is None:
        raise CheckpointAcquisitionError("github release checkpoint is incomplete")
    return (
        f"https://github.com/{spec.repository_id}/releases/download/"
        f"{quote(spec.release_tag)}/{quote(spec.filename)}"
    )


def huggingface_url(spec: CheckpointSpec) -> str:
    """Construct a Hugging Face resolve URL."""

    if spec.repository_id is None or spec.filename is None:
        raise CheckpointAcquisitionError("huggingface checkpoint is incomplete")
    revision = spec.revision or "main"
    filename = "/".join(quote(part) for part in spec.filename.split("/"))
    return f"https://huggingface.co/{spec.repository_id}/resolve/{revision}/{filename}"


def optional_auth_header(env_name: str) -> dict[str, str]:
    """Return an Authorization header from an optional environment token."""

    token = os.environ.get(env_name)
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def checkpoint_specification_fingerprint(spec: CheckpointSpec) -> str:
    """Return a deterministic fingerprint for acquisition-relevant checkpoint fields."""

    payload = {
        "checkpoint_id": spec.checkpoint_id,
        "source_type": spec.source_type.value,
        "url": str(spec.url) if spec.url is not None else None,
        "repository_id": spec.repository_id,
        "release_tag": spec.release_tag,
        "revision": spec.revision,
        "package": spec.package,
        "package_version": spec.package_version,
        "filename": spec.filename,
        "local_path": spec.local_path,
        "expected_sha256": spec.expected_sha256,
        "observed_sha256": spec.observed_sha256,
        "authority": spec.authority.model_dump(mode="json") if spec.authority is not None else None,
        "format": spec.format,
        "loader": spec.loader,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def python_env_remove() -> tuple[str, str]:
    """Environment variables removed for model-environment Python commands."""

    return ("PYTHONPATH", "PYTHONHOME")
