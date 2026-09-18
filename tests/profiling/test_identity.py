from __future__ import annotations

from torch_dae.profiling.identity import (
    TechnicalCardIdentityInputs,
    build_technical_card_identity,
    compute_identity_digest,
    generate_nonce,
    validate_identity_recomputation,
)


def _inputs(**overrides: object) -> TechnicalCardIdentityInputs:
    base = dict(
        technical_card_schema_version="1.0.0",
        model_card_id="panns-cnn14-16k-map-0438",
        model_card_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        profiling_protocol_id="audio-inference-v1",
        profiling_protocol_version="1.0.0",
        torch_dae_content_identity="content-sha256:" + "c" * 64,
        source_revision="d" * 40,
        repository_dirty=False,
        hardware_fingerprint="e" * 64,
        execution_context_fingerprint="f" * 64,
        device_backend="cpu",
        device_index=None,
        nonce="1234567890abcdef",
    )
    base.update(overrides)
    return TechnicalCardIdentityInputs(**base)


def test_deterministic_for_same_inputs() -> None:
    inputs = _inputs()
    assert compute_identity_digest(inputs) == compute_identity_digest(inputs)


def test_nonce_changes_identity() -> None:
    a = _inputs(nonce="1111111111111111")
    b = _inputs(nonce="2222222222222222")
    assert compute_identity_digest(a) != compute_identity_digest(b)


def test_device_changes_identity() -> None:
    a = _inputs(device_backend="cpu")
    b = _inputs(device_backend="cuda", device_index=0)
    assert compute_identity_digest(a) != compute_identity_digest(b)


def test_measurement_values_are_not_part_of_the_signature() -> None:
    # TechnicalCardIdentityInputs has no timing/measurement fields at all by construction.
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(TechnicalCardIdentityInputs)}
    assert not field_names & {"mean_ns", "raw_ns", "timing", "throughput"}


def test_human_id_form() -> None:
    identity = build_technical_card_identity(_inputs())
    assert identity.technical_card_id.startswith("tc-")
    assert identity.identity_digest_sha256.startswith(identity.technical_card_id[3:])


def test_recomputation_validates() -> None:
    inputs = _inputs()
    identity = build_technical_card_identity(inputs)
    validate_identity_recomputation(identity, inputs)  # must not raise


def test_recomputation_detects_tamper() -> None:
    import pytest

    inputs = _inputs()
    identity = build_technical_card_identity(inputs)
    tampered_inputs = _inputs(nonce="ffffffffffffffff")
    with pytest.raises(ValueError):
        validate_identity_recomputation(identity, tampered_inputs)


def test_nonce_is_random() -> None:
    assert generate_nonce() != generate_nonce()
