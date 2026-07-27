"""Checkpoint contracts and cache-backed acquisition manager."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, BinaryIO, Literal, Protocol
from urllib.parse import quote

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
    CheckpointHashMismatchError,
    CheckpointNotFoundError,
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
        return ensure_repository_relative(value)

    @field_validator("filename")
    @classmethod
    def filename_is_safe_relative(cls, value: str | None) -> str | None:
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
        if (
            self.expected_sha256
            and self.observed_sha256
            and self.expected_sha256 != self.observed_sha256
        ):
            raise ValueError("expected_sha256 and observed_sha256 must agree when both are set")
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

    schema_version: Literal["1.0.0"]
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


@dataclass(frozen=True)
class TransportResponse:
    """Streaming transport response."""

    status_code: int
    headers: Mapping[str, str]
    body: BinaryIO


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
        )


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
            spec = ModelCardRegistry(self.repository_root).get_card(card_id).checkpoint
        except KeyError as exc:
            raise CheckpointNotFoundError(f"model card not found: {card_id}") from exc
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
            return self._ensure_spec(card_id, spec)
        finally:
            self._report_sink = original_sink
            self.executor = original_executor

    def _ensure_spec(self, card_id: str, spec: CheckpointSpec) -> ResolvedCheckpoint:
        """Resolve a loaded checkpoint spec with acquisition reporting enabled."""

        cached = self._cached(spec)
        if cached is not None:
            return cached
        if self.policy.offline and spec.source_type in {
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
                return self._from_remote(spec, str(spec.url), {}, "https")
            case CheckpointSourceType.GITHUB_RELEASE:
                return self._from_remote(
                    spec,
                    github_release_url(spec),
                    optional_auth_header("GITHUB_TOKEN"),
                    "github_release",
                )
            case CheckpointSourceType.HUGGINGFACE:
                return self._from_remote(
                    spec,
                    huggingface_url(spec),
                    optional_auth_header("HF_TOKEN") or optional_auth_header("HUGGINGFACE_TOKEN"),
                    "huggingface",
                )
            case CheckpointSourceType.PACKAGE_BUNDLE:
                return self._from_package_bundle(card_id, spec)

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
        root = contained_path(self.runtime_root / "checkpoints", spec.checkpoint_id)
        entries: list[dict[str, object]] = []
        if root.exists():
            for child in sorted(root.iterdir()):
                if child.is_dir():
                    validity = self._validate_cache_dir(spec, child)
                    entries.append(
                        {
                            "sha256": child.name,
                            "path": str(child),
                            "valid": validity is not None,
                        }
                    )
        return {
            "card_id": card_id,
            "checkpoint_id": spec.checkpoint_id,
            "source_type": spec.source_type.value,
            "expected_sha256": spec.expected_sha256,
            "observed_sha256": spec.observed_sha256,
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
    ) -> ResolvedCheckpoint:
        if not url.startswith("https://"):
            raise CheckpointAcquisitionError("remote checkpoint URLs must use HTTPS")
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
            filename = spec.filename or Path(url).name or spec.checkpoint_id
            digest = hashlib.sha256()
            size = 0
            try:
                downloads.mkdir(parents=True, exist_ok=True)
                with tmp.open("wb") as handle:
                    for chunk in iter(lambda: response.body.read(1024 * 1024), b""):
                        size += len(chunk)
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
            sha = digest.hexdigest()
            self._record_checkpoint_event(
                "remote-stream",
                spec,
                "success",
                location=url,
                source_description=source_description,
                byte_count=size,
                sha256=sha,
            )
            self._validate_checkpoint_hashes(spec, sha, url, source_description)
            return self._finalize_tmp(spec, tmp, filename, url, source_description, sha, size)
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

    def _from_package_bundle(self, card_id: str, spec: CheckpointSpec) -> ResolvedCheckpoint:
        from torch_dae.environment.manager import EnvironmentManager

        resolved = EnvironmentManager(
            self.repository_root,
            policy=self.policy,
            executor=self.executor,
        ).ensure(card_id)
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
        self._write_metadata_or_cleanup(spec, final, location, source_description, sha, size)
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
    ) -> None:
        metadata = CheckpointMaterializationRecord(
            schema_version="1.0.0",
            checkpoint_id=spec.checkpoint_id,
            source_type=spec.source_type,
            source_description=source_description,
            resolved_url_or_location=location,
            filename=final.name,
            sha256=sha,
            size_bytes=size,
            acquired_at=utc_now(),
            cache_path=str(final.parent),
            expected_sha256=spec.expected_sha256,
            observed_sha256=spec.observed_sha256 or sha,
            environment_id=environment_id,
            specification_fingerprint=checkpoint_specification_fingerprint(spec),
            command_log_references=tuple(
                self._report_sink.references if self._report_sink is not None else ()
            ),
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
        "format": spec.format,
        "loader": spec.loader,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def python_env_remove() -> tuple[str, str]:
    """Environment variables removed for model-environment Python commands."""

    return ("PYTHONPATH", "PYTHONHOME")
