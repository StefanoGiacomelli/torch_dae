from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from collections.abc import Mapping
from pathlib import Path

import pytest

from tests.environment.test_phase01_materialization import write_phase01_repo
from torch_dae.core import checkpoint as checkpoint_module
from torch_dae.core.checkpoint import (
    CheckpointManager,
    CheckpointSpec,
    TransportResponse,
    UrllibDownloadTransport,
    github_release_url,
    huggingface_url,
    optional_auth_header,
)
from torch_dae.core.errors import (
    CheckpointAcquisitionError,
    CheckpointHashMismatchError,
    CheckpointNotFoundError,
    OfflineResourceUnavailableError,
)
from torch_dae.environment.manager import ResolvedEnvironment
from torch_dae.environment.policy import ExecutionPolicy
from torch_dae.environment.subprocess import CommandExecutor


class FakeTransport:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.calls: list[tuple[str, Mapping[str, str]]] = []

    def open(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: float,
    ) -> TransportResponse:
        self.calls.append((url, dict(headers)))
        return TransportResponse(
            200, {"content-length": str(len(self.payload))}, io.BytesIO(self.payload)
        )


class ObservableBody(io.BytesIO):
    def __init__(
        self,
        payload: bytes,
        *,
        fail_after_reads: int | None = None,
        close_error: str | None = None,
    ) -> None:
        super().__init__(payload)
        self.closed_observed = False
        self.reads = 0
        self.fail_after_reads = fail_after_reads
        self.close_error = close_error

    def read(self, size: int = -1) -> bytes:
        self.reads += 1
        if self.fail_after_reads is not None and self.reads > self.fail_after_reads:
            raise OSError("Authorization: Bearer secret-token")
        return super().read(size)

    def close(self) -> None:
        self.closed_observed = True
        if self.close_error is not None:
            raise OSError(self.close_error)
        super().close()


class ObservableTransport:
    def __init__(self, status_code: int, body: ObservableBody) -> None:
        self.status_code = status_code
        self.body = body
        self.calls: list[tuple[str, Mapping[str, str]]] = []

    def open(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: float,
    ) -> TransportResponse:
        self.calls.append((url, dict(headers)))
        return TransportResponse(self.status_code, {}, self.body)


class FailingOpenTransport:
    def __init__(self, error: OSError) -> None:
        self.error = error

    def open(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: float,
    ) -> TransportResponse:
        raise self.error


class CommandCompletedResult:
    def __init__(self, stdout: str) -> None:
        self.returncode = 0
        self.stdout = stdout
        self.stderr = ""


class LookupRunner:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout

    def run(
        self,
        command: tuple[str, ...],
        *,
        cwd: Path | None,
        env: Mapping[str, str],
        timeout: float | None,
    ) -> CommandCompletedResult:
        return CommandCompletedResult(self.stdout)


def update_checkpoint(root: Path, checkpoint: dict[str, object]) -> None:
    path = root / "model_cards/synthetic/card.json"
    card = json.loads(path.read_text())
    card["checkpoint"] = checkpoint
    path.write_text(json.dumps(card, indent=2))


def test_phase01_local_checkpoint_cache(
    tmp_path: Path, repo_root: Path, valid_fixture_dir: Path
) -> None:
    card_id = write_phase01_repo(tmp_path, repo_root, valid_fixture_dir)
    resolved = CheckpointManager(tmp_path).ensure(card_id)
    assert resolved.path.is_file()
    assert resolved.sha256 == hashlib.sha256(b"synthetic checkpoint bytes\n").hexdigest()
    metadata = json.loads((resolved.path.parent / "checkpoint-materialization.json").read_text())
    refs = metadata["command_log_references"]
    assert refs
    assert {"local-path-copy", "hash-validation", "cache-finalize"}.issubset(
        operations_for_refs(tmp_path, refs)
    )
    info = CheckpointManager(tmp_path).info(card_id)
    assert info["cached"]
    reused = CheckpointManager(tmp_path, policy=ExecutionPolicy(offline=True)).ensure(card_id)
    assert reused.path == resolved.path
    CheckpointManager(tmp_path).remove(card_id)
    assert not resolved.path.exists()


def test_phase01_remote_checkpoint_cache_hit_and_offline(
    tmp_path: Path,
    repo_root: Path,
    valid_fixture_dir: Path,
) -> None:
    card_id = write_phase01_repo(tmp_path, repo_root, valid_fixture_dir)
    payload = b"remote synthetic bytes"
    sha = hashlib.sha256(payload).hexdigest()
    update_checkpoint(
        tmp_path,
        {
            "schema_version": "1.0.0",
            "checkpoint_id": "phase01-remote-checkpoint",
            "source_type": "https",
            "url": "https://example.invalid/checkpoint.bin",
            "filename": "checkpoint.bin",
            "expected_sha256": sha,
            "format": "binary",
            "loader": "manual",
            "license": {"status": "not_applicable"},
        },
    )
    transport = FakeTransport(payload)
    resolved = CheckpointManager(tmp_path, transport=transport).ensure(card_id)
    assert resolved.sha256 == sha
    assert len(transport.calls) == 1
    metadata = json.loads((resolved.path.parent / "checkpoint-materialization.json").read_text())
    assert {"remote-open", "remote-stream", "hash-validation", "remote-finalize"}.issubset(
        operations_for_refs(tmp_path, metadata["command_log_references"])
    )

    offline = CheckpointManager(
        tmp_path,
        policy=ExecutionPolicy(offline=True),
        transport=FakeTransport(b"must not be used"),
    )
    assert offline.ensure(card_id).path == resolved.path
    assert offline.transport.calls == []


def test_phase01_checkpoint_spec_fingerprint_reacquires_changed_local_source(
    tmp_path: Path,
    repo_root: Path,
    valid_fixture_dir: Path,
) -> None:
    card_id = write_phase01_repo(tmp_path, repo_root, valid_fixture_dir)
    first = CheckpointManager(tmp_path).ensure(card_id)
    replacement = tmp_path / "tests/fixtures/phase01/checkpoint-replacement.bin"
    replacement.write_bytes(b"replacement checkpoint bytes\n")
    update_checkpoint(
        tmp_path,
        {
            "schema_version": "1.0.0",
            "checkpoint_id": "phase01-synthetic-checkpoint",
            "source_type": "local_path",
            "local_path": "tests/fixtures/phase01/checkpoint-replacement.bin",
            "format": "binary",
            "loader": "manual",
            "license": {"status": "not_applicable"},
        },
    )
    second = CheckpointManager(tmp_path).ensure(card_id)
    assert second.sha256 == hashlib.sha256(b"replacement checkpoint bytes\n").hexdigest()
    assert second.sha256 != first.sha256


def test_phase01_checkpoint_spec_fingerprint_reacquires_changed_remote_source(
    tmp_path: Path,
    repo_root: Path,
    valid_fixture_dir: Path,
) -> None:
    card_id = write_phase01_repo(tmp_path, repo_root, valid_fixture_dir)
    payload = b"same remote bytes"
    sha = hashlib.sha256(payload).hexdigest()
    checkpoint = {
        "schema_version": "1.0.0",
        "checkpoint_id": "phase01-remote-checkpoint",
        "source_type": "https",
        "url": "https://example.invalid/one.bin",
        "filename": "checkpoint.bin",
        "expected_sha256": sha,
        "format": "binary",
        "loader": "manual",
        "license": {"status": "not_applicable"},
    }
    update_checkpoint(tmp_path, checkpoint)
    first_transport = FakeTransport(payload)
    CheckpointManager(tmp_path, transport=first_transport).ensure(card_id)
    assert len(first_transport.calls) == 1
    checkpoint["url"] = "https://example.invalid/two.bin"
    update_checkpoint(tmp_path, checkpoint)
    second_transport = FakeTransport(payload)
    CheckpointManager(tmp_path, transport=second_transport).ensure(card_id)
    assert len(second_transport.calls) == 1


@pytest.mark.parametrize(
    ("source_type", "token_name"),
    [("github_release", "GITHUB_TOKEN"), ("huggingface", "HF_TOKEN")],
)
def test_phase01_hosted_checkpoint_sources_use_fake_transport_and_optional_auth(
    tmp_path: Path,
    repo_root: Path,
    valid_fixture_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_type: str,
    token_name: str,
) -> None:
    card_id = write_phase01_repo(tmp_path, repo_root, valid_fixture_dir)
    payload = f"{source_type} bytes".encode()
    sha = hashlib.sha256(payload).hexdigest()
    common: dict[str, object] = {
        "schema_version": "1.0.0",
        "checkpoint_id": f"phase01-{source_type}-checkpoint",
        "source_type": source_type,
        "repository_id": "owner/repo",
        "filename": "checkpoint.bin",
        "expected_sha256": sha,
        "format": "binary",
        "loader": "manual",
        "license": {"status": "not_applicable"},
    }
    if source_type == "github_release":
        common["release_tag"] = "v1.0.0"
    else:
        common["revision"] = "a" * 40
    update_checkpoint(tmp_path, common)
    monkeypatch.setenv(token_name, "secret-token")
    transport = FakeTransport(payload)
    resolved = CheckpointManager(tmp_path, transport=transport).ensure(card_id)
    assert resolved.sha256 == sha
    assert transport.calls[0][0].startswith("https://")
    assert transport.calls[0][1]["Authorization"] == "Bearer secret-token"


def test_phase01_remote_checkpoint_hash_mismatch(
    tmp_path: Path,
    repo_root: Path,
    valid_fixture_dir: Path,
) -> None:
    card_id = write_phase01_repo(tmp_path, repo_root, valid_fixture_dir)
    update_checkpoint(
        tmp_path,
        {
            "schema_version": "1.0.0",
            "checkpoint_id": "phase01-mismatch-checkpoint",
            "source_type": "https",
            "url": "https://example.invalid/checkpoint.bin",
            "filename": "checkpoint.bin",
            "expected_sha256": "0" * 64,
            "format": "binary",
            "loader": "manual",
            "license": {"status": "not_applicable"},
        },
    )
    with pytest.raises(CheckpointHashMismatchError):
        CheckpointManager(tmp_path, transport=FakeTransport(b"different")).ensure(card_id)
    assert not (
        tmp_path / ".torch-dae/checkpoints/.downloads/phase01-mismatch-checkpoint.download"
    ).exists()
    reports = report_payloads_for_checkpoint(tmp_path, "phase01-mismatch-checkpoint")
    hash_report = single_report(reports, "hash-validation")
    assert hash_report["status"] == "failed"
    assert hash_report["failure_classification"] == "expected_hash_mismatch"
    assert hash_report["actual_sha256"] == hashlib.sha256(b"different").hexdigest()
    assert "failure-cleanup" in {report["operation"] for report in reports}


@pytest.mark.parametrize(
    ("status_code", "expected_exception"),
    [(200, None), (404, CheckpointNotFoundError)],
)
def test_phase01_remote_response_is_closed_for_success_and_http_failure(
    tmp_path: Path,
    repo_root: Path,
    valid_fixture_dir: Path,
    status_code: int,
    expected_exception: type[Exception] | None,
) -> None:
    card_id = write_phase01_repo(tmp_path, repo_root, valid_fixture_dir)
    payload = b"remote close bytes"
    sha = hashlib.sha256(payload).hexdigest()
    update_checkpoint(
        tmp_path,
        {
            "schema_version": "1.0.0",
            "checkpoint_id": f"phase01-close-{status_code}",
            "source_type": "https",
            "url": "https://example.invalid/checkpoint.bin",
            "filename": "checkpoint.bin",
            "expected_sha256": sha,
            "format": "binary",
            "loader": "manual",
            "license": {"status": "not_applicable"},
        },
    )
    body = ObservableBody(payload)
    transport = ObservableTransport(status_code, body)
    manager = CheckpointManager(tmp_path, transport=transport)
    if expected_exception is None:
        manager.ensure(card_id)
    else:
        with pytest.raises(expected_exception):
            manager.ensure(card_id)
    assert body.closed_observed


@pytest.mark.parametrize("hash_field", ["expected_sha256", "observed_sha256"])
def test_phase01_remote_response_is_closed_for_hash_mismatch(
    tmp_path: Path,
    repo_root: Path,
    valid_fixture_dir: Path,
    hash_field: str,
) -> None:
    card_id = write_phase01_repo(tmp_path, repo_root, valid_fixture_dir)
    payload = b"remote mismatch bytes"
    checkpoint = {
        "schema_version": "1.0.0",
        "checkpoint_id": f"phase01-{hash_field}-mismatch",
        "source_type": "https",
        "url": "https://example.invalid/checkpoint.bin",
        "filename": "checkpoint.bin",
        hash_field: "0" * 64,
        "format": "binary",
        "loader": "manual",
        "license": {"status": "not_applicable"},
    }
    update_checkpoint(tmp_path, checkpoint)
    body = ObservableBody(payload)
    with pytest.raises(CheckpointHashMismatchError):
        CheckpointManager(tmp_path, transport=ObservableTransport(200, body)).ensure(card_id)
    assert body.closed_observed
    reports = report_payloads_for_checkpoint(tmp_path, f"phase01-{hash_field}-mismatch")
    hash_report = single_report(reports, "hash-validation")
    assert hash_report["status"] == "failed"
    assert hash_report["failure_classification"] == hash_field.replace("sha256", "hash_mismatch")
    assert hash_report["actual_sha256"] == hashlib.sha256(payload).hexdigest()
    assert not list(
        (tmp_path / f".torch-dae/checkpoints/phase01-{hash_field}-mismatch").glob(
            "*/checkpoint-materialization.json"
        )
    )


def test_phase01_interrupted_remote_read_cleans_partial_state_and_reports_failure(
    tmp_path: Path,
    repo_root: Path,
    valid_fixture_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card_id = write_phase01_repo(tmp_path, repo_root, valid_fixture_dir)
    monkeypatch.setenv("GITHUB_TOKEN", "secret-token")
    update_checkpoint(
        tmp_path,
        {
            "schema_version": "1.0.0",
            "checkpoint_id": "phase01-interrupted-checkpoint",
            "source_type": "github_release",
            "repository_id": "owner/repo",
            "release_tag": "v1.0.0",
            "filename": "checkpoint.bin",
            "format": "binary",
            "loader": "manual",
            "license": {"status": "not_applicable"},
        },
    )
    body = ObservableBody(b"partial bytes", fail_after_reads=1)
    with pytest.raises(CheckpointAcquisitionError) as exc_info:
        CheckpointManager(tmp_path, transport=ObservableTransport(200, body)).ensure(card_id)
    assert isinstance(exc_info.value.__cause__, OSError)

    assert body.closed_observed
    assert not list((tmp_path / ".torch-dae/checkpoints/.downloads").glob("phase01-interrupted*"))
    assert not list(
        (tmp_path / ".torch-dae/checkpoints/phase01-interrupted-checkpoint").glob(
            "*/checkpoint-materialization.json"
        )
    )
    reports = sorted(
        (tmp_path / ".torch-dae/reports/checkpoints/phase01-interrupted-checkpoint").glob("*.json")
    )
    assert reports
    payload = "\n".join(path.read_text() for path in reports)
    assert "secret-token" not in payload
    assert "redacted secret" in payload
    operations = operations_from_report_paths(reports)
    assert "remote-stream" in operations
    assert "failure-cleanup" in operations
    stream_report = single_report(
        [json.loads(path.read_text()) for path in reports],
        "remote-stream",
    )
    assert stream_report["status"] == "failed"
    assert stream_report["failure_classification"] == "OSError"
    assert stream_report["failure_detail"] == "Authorization: [redacted secret]"
    assert stream_report["byte_count"] == len(b"partial bytes")


def test_phase01_transport_open_oserror_is_typed_reported_and_redacted(
    tmp_path: Path,
    repo_root: Path,
    valid_fixture_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card_id = write_phase01_repo(tmp_path, repo_root, valid_fixture_dir)
    monkeypatch.setenv("GITHUB_TOKEN", "secret-token")
    update_checkpoint(
        tmp_path,
        {
            "schema_version": "1.0.0",
            "checkpoint_id": "phase01-open-error-checkpoint",
            "source_type": "github_release",
            "repository_id": "owner/repo",
            "release_tag": "v1.0.0",
            "filename": "checkpoint.bin",
            "format": "binary",
            "loader": "manual",
            "license": {"status": "not_applicable"},
        },
    )
    with pytest.raises(CheckpointAcquisitionError) as exc_info:
        CheckpointManager(
            tmp_path,
            transport=FailingOpenTransport(OSError("Authorization: Bearer secret-token")),
        ).ensure(card_id)
    assert isinstance(exc_info.value.__cause__, OSError)
    reports = report_payloads_for_checkpoint(tmp_path, "phase01-open-error-checkpoint")
    serialized = json.dumps(reports)
    assert "secret-token" not in serialized
    assert "redacted secret" in serialized
    open_report = single_report(reports, "remote-open")
    assert open_report["status"] == "failed"
    assert open_report["failure_classification"] == "OSError"


def test_phase01_urllib_transport_urlerror_is_typed_and_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "secret-token")

    def raise_url_error(request: urllib.request.Request, timeout: float) -> object:
        raise urllib.error.URLError("Authorization: Bearer secret-token")

    monkeypatch.setattr(urllib.request, "urlopen", raise_url_error)
    with pytest.raises(CheckpointAcquisitionError) as exc_info:
        UrllibDownloadTransport().open(
            "https://example.invalid/checkpoint.bin",
            headers={"Authorization": "Bearer secret-token"},
            timeout=1,
        )
    assert isinstance(exc_info.value.__cause__, urllib.error.URLError)
    assert "secret-token" not in str(exc_info.value)
    assert "redacted secret" in str(exc_info.value)


def test_phase01_urllib_transport_httperror_closes_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = ObservableBody(b"")

    def raise_http_error(request: urllib.request.Request, timeout: float) -> object:
        raise urllib.error.HTTPError(
            "https://example.invalid/checkpoint.bin?token=secret-token",
            404,
            "not found",
            {},
            body,
        )

    monkeypatch.setattr(urllib.request, "urlopen", raise_http_error)
    with pytest.raises(CheckpointNotFoundError) as exc_info:
        UrllibDownloadTransport().open(
            "https://example.invalid/checkpoint.bin?token=secret-token",
            headers={},
            timeout=1,
        )
    assert isinstance(exc_info.value.__cause__, urllib.error.HTTPError)
    assert body.closed_observed


def test_phase01_successful_response_close_failure_is_reported_and_typed(
    tmp_path: Path,
    repo_root: Path,
    valid_fixture_dir: Path,
) -> None:
    card_id = write_phase01_repo(tmp_path, repo_root, valid_fixture_dir)
    payload = b"close failure bytes"
    sha = hashlib.sha256(payload).hexdigest()
    update_checkpoint(
        tmp_path,
        {
            "schema_version": "1.0.0",
            "checkpoint_id": "phase01-close-failure-checkpoint",
            "source_type": "https",
            "url": "https://example.invalid/checkpoint.bin",
            "filename": "checkpoint.bin",
            "expected_sha256": sha,
            "format": "binary",
            "loader": "manual",
            "license": {"status": "not_applicable"},
        },
    )
    body = ObservableBody(payload, close_error="close denied")
    with pytest.raises(CheckpointAcquisitionError) as exc_info:
        CheckpointManager(tmp_path, transport=ObservableTransport(200, body)).ensure(card_id)
    assert isinstance(exc_info.value.__cause__, OSError)
    assert body.closed_observed
    close_report = single_report(
        report_payloads_for_checkpoint(tmp_path, "phase01-close-failure-checkpoint"),
        "response-close",
    )
    assert close_report["status"] == "failed"
    assert close_report["failure_classification"] == "OSError"


def test_phase01_failed_response_close_does_not_mask_stream_error(
    tmp_path: Path,
    repo_root: Path,
    valid_fixture_dir: Path,
) -> None:
    card_id = write_phase01_repo(tmp_path, repo_root, valid_fixture_dir)
    update_checkpoint(
        tmp_path,
        {
            "schema_version": "1.0.0",
            "checkpoint_id": "phase01-stream-and-close-failure",
            "source_type": "https",
            "url": "https://example.invalid/checkpoint.bin",
            "filename": "checkpoint.bin",
            "format": "binary",
            "loader": "manual",
            "license": {"status": "not_applicable"},
        },
    )
    body = ObservableBody(b"partial", fail_after_reads=1, close_error="close denied")
    with pytest.raises(CheckpointAcquisitionError) as exc_info:
        CheckpointManager(tmp_path, transport=ObservableTransport(200, body)).ensure(card_id)
    assert isinstance(exc_info.value.__cause__, OSError)
    assert "download stream failed" in str(exc_info.value)
    operations = {
        report["operation"]
        for report in report_payloads_for_checkpoint(tmp_path, "phase01-stream-and-close-failure")
    }
    assert {"remote-stream", "response-close", "failure-cleanup"}.issubset(operations)


def test_phase01_offline_checkpoint_miss(
    tmp_path: Path,
    repo_root: Path,
    valid_fixture_dir: Path,
) -> None:
    card_id = write_phase01_repo(tmp_path, repo_root, valid_fixture_dir)
    update_checkpoint(
        tmp_path,
        {
            "schema_version": "1.0.0",
            "checkpoint_id": "phase01-offline-remote-checkpoint",
            "source_type": "https",
            "url": "https://example.invalid/checkpoint.bin",
            "filename": "checkpoint.bin",
            "expected_sha256": "0" * 64,
            "format": "binary",
            "loader": "manual",
            "license": {"status": "not_applicable"},
        },
    )
    with pytest.raises(OfflineResourceUnavailableError):
        CheckpointManager(tmp_path, policy=ExecutionPolicy(offline=True)).ensure(card_id)
    reports = report_payloads_for_checkpoint(tmp_path, "phase01-offline-remote-checkpoint")
    lookup_report = single_report(reports, "offline-cache-lookup")
    assert lookup_report["status"] == "failed"
    assert lookup_report["failure_classification"] == "offline_cache_miss"


def test_phase01_local_copy_failure_is_typed_reported_and_cleans_tmp(
    tmp_path: Path,
    repo_root: Path,
    valid_fixture_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card_id = write_phase01_repo(tmp_path, repo_root, valid_fixture_dir)

    def fail_copy(source: Path, destination: Path) -> None:
        raise PermissionError("copy denied")

    monkeypatch.setattr(checkpoint_module.shutil, "copyfile", fail_copy)
    with pytest.raises(CheckpointAcquisitionError) as exc_info:
        CheckpointManager(tmp_path).ensure(card_id)
    assert isinstance(exc_info.value.__cause__, PermissionError)
    reports = report_payloads_for_checkpoint(tmp_path, "phase01-synthetic-checkpoint")
    copy_report = single_report(reports, "local-path-copy", status="failed")
    assert copy_report["failure_classification"] == "PermissionError"
    assert not list(
        (tmp_path / ".torch-dae/checkpoints/phase01-synthetic-checkpoint").glob("*/.*.tmp")
    )
    assert not list(
        (tmp_path / ".torch-dae/checkpoints/phase01-synthetic-checkpoint").glob(
            "*/checkpoint-materialization.json"
        )
    )


def test_phase01_cache_finalize_failure_is_typed_reported_and_cleans_tmp(
    tmp_path: Path,
    repo_root: Path,
    valid_fixture_dir: Path,
) -> None:
    card_id = write_phase01_repo(tmp_path, repo_root, valid_fixture_dir)

    class ReplaceFailureManager(CheckpointManager):
        def _replace_file(self, source: Path, target: Path) -> None:
            raise PermissionError("replace denied")

    with pytest.raises(CheckpointAcquisitionError) as exc_info:
        ReplaceFailureManager(tmp_path).ensure(card_id)
    assert isinstance(exc_info.value.__cause__, PermissionError)
    reports = report_payloads_for_checkpoint(tmp_path, "phase01-synthetic-checkpoint")
    finalize_report = single_report(reports, "cache-finalize", status="failed")
    assert finalize_report["failure_classification"] == "PermissionError"
    assert "failure-cleanup" in {report["operation"] for report in reports}
    assert not list(
        (tmp_path / ".torch-dae/checkpoints/phase01-synthetic-checkpoint").glob("*/.*.tmp")
    )


def test_phase01_metadata_write_failure_removes_incomplete_cache_entry_and_reports(
    tmp_path: Path,
    repo_root: Path,
    valid_fixture_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card_id = write_phase01_repo(tmp_path, repo_root, valid_fixture_dir)

    def fail_metadata(path: Path, data: object) -> None:
        raise PermissionError("metadata denied")

    monkeypatch.setattr(checkpoint_module, "write_json_atomic", fail_metadata)
    with pytest.raises(CheckpointAcquisitionError) as exc_info:
        CheckpointManager(tmp_path).ensure(card_id)
    assert isinstance(exc_info.value.__cause__, PermissionError)
    reports = report_payloads_for_checkpoint(tmp_path, "phase01-synthetic-checkpoint")
    metadata_report = single_report(reports, "metadata-write")
    assert metadata_report["status"] == "failed"
    assert metadata_report["failure_classification"] == "PermissionError"
    assert "failure-cleanup" in {report["operation"] for report in reports}
    assert CheckpointManager(tmp_path).info(card_id)["cached"] == []


def test_phase01_package_bundle_malformed_lookup_is_typed_and_reported(
    tmp_path: Path,
    repo_root: Path,
    valid_fixture_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card_id = write_phase01_repo(tmp_path, repo_root, valid_fixture_dir)
    update_checkpoint(
        tmp_path,
        {
            "schema_version": "1.0.0",
            "checkpoint_id": "phase01-package-json-checkpoint",
            "source_type": "package_bundle",
            "package": "torch-dae",
            "package_version": "0.1.0",
            "filename": "torch_dae/__init__.py",
            "format": "python",
            "loader": "manual",
            "license": {"status": "not_applicable"},
        },
    )

    def fake_ensure(self: object, requested: str) -> ResolvedEnvironment:
        return ResolvedEnvironment(
            environment_id="synthetic-package-environment",
            model_card_id=requested,
            root=tmp_path / "env",
            python_executable=Path(sys.executable),
            fingerprint="a" * 64,
            python_version=".".join(str(part) for part in sys.version_info[:3]),
            platform="synthetic",
            installed_packages={},
            installed_sources=(),
            valid=True,
        )

    from torch_dae.environment import manager as environment_manager

    monkeypatch.setattr(environment_manager.EnvironmentManager, "ensure", fake_ensure)
    with pytest.raises(CheckpointAcquisitionError) as exc_info:
        CheckpointManager(
            tmp_path,
            executor=CommandExecutor(LookupRunner("{bad json")),
        ).ensure(card_id)
    assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)
    report = single_report(
        report_payloads_for_checkpoint(tmp_path, "phase01-package-json-checkpoint"),
        "package-bundle-lookup",
        status="failed",
    )
    assert report["failure_classification"] == "JSONDecodeError"


def test_phase01_offline_local_checkpoint_first_acquisition(
    tmp_path: Path,
    repo_root: Path,
    valid_fixture_dir: Path,
) -> None:
    card_id = write_phase01_repo(tmp_path, repo_root, valid_fixture_dir)
    resolved = CheckpointManager(tmp_path, policy=ExecutionPolicy(offline=True)).ensure(card_id)
    assert resolved.path.is_file()


@pytest.mark.integration
def test_phase01_package_bundle_checkpoint_uses_owned_distribution_file_offline(
    tmp_path: Path,
    repo_root: Path,
    valid_fixture_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path.parent / f"{tmp_path.name}-uv-cache"))
    card_id = write_phase01_repo(tmp_path, repo_root, valid_fixture_dir)
    package_file = tmp_path / "src/torch_dae/__init__.py"
    sha = hashlib.sha256(package_file.read_bytes()).hexdigest()
    update_checkpoint(
        tmp_path,
        {
            "schema_version": "1.0.0",
            "checkpoint_id": "phase01-package-bundle-checkpoint",
            "source_type": "package_bundle",
            "package": "torch-dae",
            "package_version": "0.1.0",
            "filename": "torch_dae/__init__.py",
            "expected_sha256": sha,
            "format": "python",
            "loader": "manual",
            "license": {"status": "not_applicable"},
        },
    )
    from torch_dae.environment.manager import EnvironmentManager

    EnvironmentManager(tmp_path, policy=ExecutionPolicy(command_timeout_seconds=120)).ensure(
        card_id
    )
    resolved = CheckpointManager(tmp_path, policy=ExecutionPolicy(offline=True)).ensure(card_id)
    assert resolved.sha256 == sha


@pytest.mark.integration
@pytest.mark.parametrize(
    ("filename", "version"),
    [
        ("torch_dae/missing-resource.bin", "0.1.0"),
        ("torch_dae/__init__.py", "9.9.9"),
    ],
)
def test_phase01_package_bundle_rejects_missing_resource_or_version(
    tmp_path: Path,
    repo_root: Path,
    valid_fixture_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    version: str,
) -> None:
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path.parent / f"{tmp_path.name}-uv-cache"))
    card_id = write_phase01_repo(tmp_path, repo_root, valid_fixture_dir)
    update_checkpoint(
        tmp_path,
        {
            "schema_version": "1.0.0",
            "checkpoint_id": "phase01-package-bundle-checkpoint",
            "source_type": "package_bundle",
            "package": "torch-dae",
            "package_version": version,
            "filename": filename,
            "format": "python",
            "loader": "manual",
            "license": {"status": "not_applicable"},
        },
    )
    with pytest.raises(CheckpointNotFoundError):
        CheckpointManager(
            tmp_path,
            policy=ExecutionPolicy(command_timeout_seconds=120),
        ).ensure(card_id)


@pytest.mark.integration
def test_phase01_package_bundle_rejects_file_owned_by_other_distribution(
    tmp_path: Path,
    repo_root: Path,
    valid_fixture_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card_id = write_phase01_repo(tmp_path, repo_root, valid_fixture_dir)
    env_root = tmp_path / "two-dist-env"
    subprocess.run(
        ["uv", "venv", str(env_root), "--python", sys.executable],
        check=True,
        capture_output=True,
        text=True,
    )
    wheel_a = tmp_path / "distribution_a-0.1.0-py3-none-any.whl"
    wheel_b = tmp_path / "distribution_b-0.1.0-py3-none-any.whl"
    make_owned_wheel(wheel_a, "distribution-a", "distribution_a/resource_a.bin", b"a\n")
    make_owned_wheel(wheel_b, "distribution-b", "distribution_b/resource_b.bin", b"b\n")
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(env_root / "bin/python"),
            str(wheel_a),
            str(wheel_b),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    update_checkpoint(
        tmp_path,
        {
            "schema_version": "1.0.0",
            "checkpoint_id": "phase01-cross-distribution-checkpoint",
            "source_type": "package_bundle",
            "package": "distribution-a",
            "package_version": "0.1.0",
            "filename": "distribution_b/resource_b.bin",
            "format": "binary",
            "loader": "manual",
            "license": {"status": "not_applicable"},
        },
    )

    from torch_dae.environment import manager as environment_manager
    from torch_dae.environment.manager import ResolvedEnvironment

    def fake_ensure(self: object, requested: str) -> ResolvedEnvironment:
        return ResolvedEnvironment(
            environment_id="synthetic-two-dist-environment",
            model_card_id=requested,
            root=env_root,
            python_executable=env_root / "bin/python",
            fingerprint="a" * 64,
            python_version=".".join(str(part) for part in sys.version_info[:3]),
            platform="synthetic",
            installed_packages={},
            installed_sources=(),
            valid=True,
        )

    monkeypatch.setattr(environment_manager.EnvironmentManager, "ensure", fake_ensure)
    with pytest.raises(CheckpointNotFoundError):
        CheckpointManager(tmp_path, policy=ExecutionPolicy(offline=True)).ensure(card_id)


def test_phase01_remote_url_construction(valid_fixture_dir: Path) -> None:
    github = CheckpointSpec.model_validate_json(
        (valid_fixture_dir / "checkpoint.github_release.json").read_text()
    )
    huggingface = CheckpointSpec.model_validate_json(
        (valid_fixture_dir / "checkpoint.huggingface.json").read_text()
    )
    assert github_release_url(github).startswith("https://github.com/")
    assert "/releases/download/" in github_release_url(github)
    assert huggingface_url(huggingface).startswith("https://huggingface.co/")
    assert "/resolve/" in huggingface_url(huggingface)


def test_phase01_optional_auth_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TORCH_DAE_MISSING_TOKEN", raising=False)
    assert optional_auth_header("TORCH_DAE_MISSING_TOKEN") == {}


def test_phase01_checkpoint_expected_observed_disagreement_fails(
    valid_fixture_dir: Path,
) -> None:
    data = json.loads((valid_fixture_dir / "checkpoint.local_path.json").read_text())
    data["expected_sha256"] = "1" * 64
    data["observed_sha256"] = "2" * 64
    with pytest.raises(ValueError, match="must agree"):
        CheckpointSpec.model_validate(data)


def test_phase01_checkpoint_info_ignores_invalid_cache_entries(
    tmp_path: Path,
    repo_root: Path,
    valid_fixture_dir: Path,
) -> None:
    card_id = write_phase01_repo(tmp_path, repo_root, valid_fixture_dir)
    root = tmp_path / ".torch-dae/checkpoints/phase01-synthetic-checkpoint"
    (root / "not-a-sha").mkdir(parents=True)
    corrupt = "1" * 64
    (root / corrupt).mkdir()
    (root / corrupt / "checkpoint-materialization.json").write_text("{bad json")
    mismatched = "2" * 64
    (root / mismatched).mkdir()
    (root / mismatched / "checkpoint-materialization.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "checkpoint_id": "other",
                "source_type": "local_path",
                "source_description": "local_path",
                "resolved_url_or_location": "x",
                "filename": "missing.bin",
                "sha256": mismatched,
                "size_bytes": 0,
                "acquired_at": "2026-01-01T00:00:00+00:00",
                "cache_path": str(root / mismatched),
            }
        )
    )
    info = CheckpointManager(tmp_path).info(card_id)
    assert all(not item["valid"] for item in info["cached"])


def operations_for_refs(root: Path, references: list[str]) -> set[str]:
    return {json.loads((root / ".torch-dae" / ref).read_text())["operation"] for ref in references}


def operations_from_report_paths(paths: list[Path]) -> set[str]:
    return {json.loads(path.read_text())["operation"] for path in paths}


def report_payloads_for_checkpoint(root: Path, checkpoint_id: str) -> list[dict[str, object]]:
    report_dir = root / ".torch-dae/reports/checkpoints" / checkpoint_id
    return [json.loads(path.read_text()) for path in sorted(report_dir.glob("*.json"))]


def single_report(
    reports: list[dict[str, object]],
    operation: str,
    *,
    status: str | None = None,
) -> dict[str, object]:
    matches = [
        report
        for report in reports
        if report["operation"] == operation and (status is None or report["status"] == status)
    ]
    assert len(matches) == 1
    return matches[0]


def make_owned_wheel(path: Path, distribution: str, member: str, payload: bytes) -> None:
    normalized = distribution.replace("-", "_")
    dist_info = f"{normalized}-0.1.0.dist-info"
    metadata = f"Metadata-Version: 2.1\nName: {distribution}\nVersion: 0.1.0\n".encode()
    wheel = b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    record = f"{member},,\n{dist_info}/METADATA,,\n{dist_info}/WHEEL,,\n{dist_info}/RECORD,,\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, payload)
        archive.writestr(f"{dist_info}/METADATA", metadata)
        archive.writestr(f"{dist_info}/WHEEL", wheel)
        archive.writestr(f"{dist_info}/RECORD", record.encode())
