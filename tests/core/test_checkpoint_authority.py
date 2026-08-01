from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping
from pathlib import Path

import pytest
from typer.testing import CliRunner

from torch_dae.cli import checkpoints as checkpoint_cli
from torch_dae.cli.main import app
from torch_dae.core.checkpoint import (
    CheckpointAcquisitionPolicy,
    CheckpointManager,
    CheckpointSpec,
    TransportResponse,
    checkpoint_specification_fingerprint,
    resolve_checkpoint_authority,
)
from torch_dae.core.errors import (
    CheckpointAuthorityResolutionError,
    CheckpointNotFoundError,
    CheckpointPublishedChecksumMismatchError,
    CheckpointResponseTooLargeError,
    CheckpointSizeMismatchError,
    OfflineResourceUnavailableError,
)
from torch_dae.environment.policy import ExecutionPolicy


class MappingTransport:
    def __init__(
        self,
        responses: Mapping[str, tuple[int, Mapping[str, str], bytes, str | None]],
    ) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def open(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: float,
    ) -> TransportResponse:
        self.calls.append(url)
        status, response_headers, body, effective_url = self.responses[url]
        return TransportResponse(status, response_headers, io.BytesIO(body), effective_url)


class RejectingTransport:
    def open(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: float,
    ) -> TransportResponse:
        raise AssertionError(f"network transport must not be used offline: {url}")


def authority_spec(
    payload: bytes,
    *,
    checkpoint_id: str = "synthetic-authoritative-checkpoint",
    size: int | None = None,
    md5: str | None = None,
    include_sha256: bool = False,
) -> CheckpointSpec:
    checksums: list[dict[str, str]] = [
        {
            "algorithm": "md5",
            "digest": md5 or hashlib.md5(payload).hexdigest(),
        }
    ]
    if include_sha256:
        checksums.append({"algorithm": "sha256", "digest": hashlib.sha256(payload).hexdigest()})
    return CheckpointSpec.model_validate(
        {
            "schema_version": "2.0.0",
            "checkpoint_id": checkpoint_id,
            "source_type": "https",
            "filename": "models/Cnn14_16k_mAP=0.438.pth",
            "authority": {
                "provider": "zenodo",
                "record_id": "12345",
                "filename": "models/Cnn14_16k_mAP=0.438.pth",
                "expected_size_bytes": size if size is not None else len(payload),
                "published_checksums": checksums,
                "record_url": "https://zenodo.org/records/12345",
                "provenance_status": "authoritative_provider_declared",
            },
            "format": "pytorch_state_dict",
            "loader": "synthetic loader",
            "license": {"status": "officially_reported", "name": "synthetic"},
        }
    )


def metadata_bytes(
    spec: CheckpointSpec,
    *,
    duplicate: bool = False,
    payload_url: str = "https://zenodo.org/api/records/12345/files/checkpoint/content",
) -> bytes:
    assert spec.authority is not None
    checksum_values = [
        f"{item.algorithm.value}:{item.digest}" for item in spec.authority.published_checksums
    ]
    entry = {
        "key": spec.authority.filename,
        "size": spec.authority.expected_size_bytes,
        "checksum": checksum_values[0],
        "links": {"content": payload_url},
    }
    if len(checksum_values) > 1:
        entry["checksums"] = checksum_values[1:]
    files = [entry, dict(entry)] if duplicate else [entry]
    return json.dumps({"id": int(spec.authority.record_id), "files": files}).encode()


def transport_for(
    spec: CheckpointSpec,
    payload: bytes,
    *,
    metadata: bytes | None = None,
    payload_effective_url: str | None = None,
) -> MappingTransport:
    metadata_url = "https://zenodo.org/api/records/12345"
    payload_url = "https://zenodo.org/api/records/12345/files/checkpoint/content"
    return MappingTransport(
        {
            metadata_url: (
                200,
                {
                    "Content-Type": "application/json",
                    "ETag": '"metadata-etag"',
                    "Last-Modified": "Fri, 01 Aug 2026 00:00:00 GMT",
                },
                metadata if metadata is not None else metadata_bytes(spec),
                metadata_url,
            ),
            payload_url: (
                200,
                {
                    "Content-Length": str(
                        spec.authority.expected_size_bytes if spec.authority else len(payload)
                    ),
                    "ETag": '"payload-etag"',
                    "Last-Modified": "Fri, 01 Aug 2026 00:00:01 GMT",
                },
                payload,
                payload_effective_url or payload_url,
            ),
        }
    )


def complete_policy(
    *, maximum_bytes: int | None = None, allow_network: bool = True
) -> CheckpointAcquisitionPolicy:
    return CheckpointAcquisitionPolicy(
        allow_network=allow_network,
        allow_authentication=False,
        maximum_bytes=maximum_bytes,
        require_authority=True,
        require_exact_size=True,
        require_published_checksums=True,
        require_observed_sha256=True,
    )


def test_zenodo_metadata_resolves_without_payload_acquisition() -> None:
    payload = b"authoritative bytes"
    spec = authority_spec(payload)
    transport = transport_for(spec, payload)

    result = resolve_checkpoint_authority(spec, transport=transport)

    assert transport.calls == ["https://zenodo.org/api/records/12345"]
    assert result.requested_filename == result.resolved_filename == spec.filename
    assert result.expected_size_bytes == len(payload)
    assert result.etag == '"metadata-etag"'
    assert result.last_modified is not None
    assert result.metadata_response_sha256 == hashlib.sha256(metadata_bytes(spec)).hexdigest()


def test_zenodo_missing_and_duplicate_file_identity_fail() -> None:
    payload = b"authoritative bytes"
    spec = authority_spec(payload)
    missing = json.dumps({"id": 12345, "files": []}).encode()
    with pytest.raises(CheckpointNotFoundError):
        resolve_checkpoint_authority(spec, transport=transport_for(spec, payload, metadata=missing))
    with pytest.raises(CheckpointAuthorityResolutionError, match="duplicate"):
        resolve_checkpoint_authority(
            spec,
            transport=transport_for(spec, payload, metadata=metadata_bytes(spec, duplicate=True)),
        )


@pytest.mark.parametrize(
    "metadata",
    [b"{bad json", b'{"id":12345,"id":12345,"files":[]}'],
)
def test_malformed_or_duplicate_key_metadata_fails(metadata: bytes) -> None:
    payload = b"authoritative bytes"
    spec = authority_spec(payload)
    with pytest.raises(CheckpointAuthorityResolutionError, match="invalid JSON"):
        resolve_checkpoint_authority(
            spec, transport=transport_for(spec, payload, metadata=metadata)
        )


def test_untrusted_authority_urls_fail() -> None:
    payload = b"authoritative bytes"
    data = authority_spec(payload).model_dump(mode="json")
    data["authority"]["record_url"] = "https://example.invalid/records/12345"
    with pytest.raises(ValueError, match="official HTTPS"):
        CheckpointSpec.model_validate(data)

    spec = authority_spec(payload)
    malicious = metadata_bytes(spec, payload_url="file:///tmp/checkpoint")
    with pytest.raises(CheckpointAuthorityResolutionError, match="provider-controlled HTTPS"):
        resolve_checkpoint_authority(
            spec, transport=transport_for(spec, payload, metadata=malicious)
        )


def test_authoritative_md5_exact_size_and_observed_sha256_are_independent(
    tmp_path: Path,
) -> None:
    payload = b"authoritative md5-only payload"
    spec = authority_spec(payload)
    manager = CheckpointManager(tmp_path, transport=transport_for(spec, payload))

    resolved = manager.ensure_checkpoint(spec, acquisition_policy=complete_policy())

    assert resolved.sha256 == hashlib.sha256(payload).hexdigest()
    record = json.loads((resolved.path.parent / "checkpoint-materialization.json").read_text())
    assert record["expected_size_bytes"] == record["observed_size_bytes"] == len(payload)
    assert record["published_checksums"] == [
        {"algorithm": "md5", "digest": hashlib.md5(payload).hexdigest()}
    ]
    assert {item["algorithm"] for item in record["observed_checksums"]} == {"md5", "sha256"}
    assert record["observed_sha256"] == resolved.sha256
    assert record["sha256"] == resolved.sha256
    assert record["payload_etag"] == '"payload-etag"'
    assert not Path(record["cache_path"]).is_absolute()


def test_published_sha256_is_verified_when_present(tmp_path: Path) -> None:
    payload = b"payload with two published digests"
    spec = authority_spec(payload, include_sha256=True)
    resolved = CheckpointManager(
        tmp_path, transport=transport_for(spec, payload)
    ).ensure_checkpoint(spec, acquisition_policy=complete_policy())
    assert resolved.sha256 == hashlib.sha256(payload).hexdigest()


def test_exact_size_mismatch_prevents_cache_installation(tmp_path: Path) -> None:
    declared_payload = b"declared payload"
    acquired_payload = b"short"
    spec = authority_spec(declared_payload)
    with pytest.raises(CheckpointSizeMismatchError):
        CheckpointManager(
            tmp_path, transport=transport_for(spec, acquired_payload)
        ).ensure_checkpoint(spec, acquisition_policy=complete_policy())
    assert not list(
        (tmp_path / ".torch-dae/checkpoints/synthetic-authoritative-checkpoint").glob(
            "*/checkpoint-materialization.json"
        )
    )


def test_published_checksum_mismatch_prevents_cache_installation(tmp_path: Path) -> None:
    declared_payload = b"declared payload"
    acquired_payload = b"tampered payload"
    assert len(declared_payload) == len(acquired_payload)
    spec = authority_spec(declared_payload)
    with pytest.raises(CheckpointPublishedChecksumMismatchError):
        CheckpointManager(
            tmp_path, transport=transport_for(spec, acquired_payload)
        ).ensure_checkpoint(spec, acquisition_policy=complete_policy())
    assert not list(
        (tmp_path / ".torch-dae/checkpoints/synthetic-authoritative-checkpoint").glob(
            "*/checkpoint-materialization.json"
        )
    )


def test_maximum_bytes_is_independent_from_exact_size(tmp_path: Path) -> None:
    payload = b"resource ceiling payload"
    spec = authority_spec(payload)
    with pytest.raises(CheckpointResponseTooLargeError):
        CheckpointManager(tmp_path, transport=transport_for(spec, payload)).ensure_checkpoint(
            spec, acquisition_policy=complete_policy(maximum_bytes=len(payload) - 1)
        )


def test_valid_authoritative_cache_reuses_offline_and_revalidates_bytes(tmp_path: Path) -> None:
    payload = b"offline authoritative payload"
    spec = authority_spec(payload)
    first = CheckpointManager(tmp_path, transport=transport_for(spec, payload)).ensure_checkpoint(
        spec, acquisition_policy=complete_policy()
    )
    offline = CheckpointManager(
        tmp_path,
        policy=ExecutionPolicy(offline=True),
        transport=RejectingTransport(),
    )
    reused = offline.ensure_checkpoint(
        spec, acquisition_policy=complete_policy(allow_network=False)
    )
    assert reused.path == first.path


def test_invalid_authoritative_cache_fails_offline(tmp_path: Path) -> None:
    payload = b"offline authoritative payload"
    spec = authority_spec(payload)
    first = CheckpointManager(tmp_path, transport=transport_for(spec, payload)).ensure_checkpoint(
        spec, acquisition_policy=complete_policy()
    )
    first.path.write_bytes(b"tampered")
    offline = CheckpointManager(tmp_path, policy=ExecutionPolicy(offline=True))
    with pytest.raises(OfflineResourceUnavailableError):
        offline.ensure_checkpoint(spec, acquisition_policy=complete_policy(allow_network=False))


def test_authority_fingerprint_is_canonical_and_sensitive() -> None:
    payload = b"fingerprint payload"
    original = authority_spec(payload)
    reordered_data = original.model_dump(mode="json")
    reordered_data["authority"]["published_checksums"] = list(
        reversed(reordered_data["authority"]["published_checksums"])
    )
    reordered = CheckpointSpec.model_validate(reordered_data)
    assert checkpoint_specification_fingerprint(original) == checkpoint_specification_fingerprint(
        reordered
    )

    for field, value in (
        ("record_id", "54321"),
        ("filename", "models/other=checkpoint.pth"),
        ("expected_size_bytes", len(payload) + 1),
    ):
        data = original.model_dump(mode="json")
        data["authority"][field] = value
        if field == "filename":
            data["filename"] = value
        if field == "record_id":
            data["authority"]["record_url"] = None
        changed = CheckpointSpec.model_validate(data)
        assert checkpoint_specification_fingerprint(
            changed
        ) != checkpoint_specification_fingerprint(original)


def test_metadata_provenance_is_deterministic_except_runtime_fields() -> None:
    payload = b"metadata deterministic payload"
    spec = authority_spec(payload)
    first = resolve_checkpoint_authority(spec, transport=transport_for(spec, payload))
    second = resolve_checkpoint_authority(spec, transport=transport_for(spec, payload))
    first_data = first.model_dump(mode="json")
    second_data = second.model_dump(mode="json")
    for data in (first_data, second_data):
        data.pop("retrieved_at")
        data.pop("metadata_source")
    assert first_data == second_data


def test_card_independent_checkpoint_cli_resolve_ensure_and_info(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"CLI authority payload"
    spec = authority_spec(payload)
    spec_path = tmp_path / "checkpoint.json"
    spec_path.write_text(spec.model_dump_json(indent=2))
    manager = CheckpointManager(tmp_path, transport=transport_for(spec, payload))
    monkeypatch.setattr(checkpoint_cli, "_manager", lambda offline=False: manager)
    runner = CliRunner()

    resolved = runner.invoke(app, ["checkpoint", "resolve", "--spec", str(spec_path), "--json"])
    assert resolved.exit_code == 0, resolved.output
    assert '"metadata_source": "network"' in resolved.output
    ensured = runner.invoke(app, ["checkpoint", "ensure-spec", "--spec", str(spec_path), "--json"])
    assert ensured.exit_code == 0, ensured.output
    info = runner.invoke(app, ["checkpoint", "info-spec", "--spec", str(spec_path), "--json"])
    assert info.exit_code == 0, info.output
    assert '"authority_cache"' in info.output
    assert '"valid": true' in info.output
