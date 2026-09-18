"""Hash-derived Technical Card identity (Section 5 of the profiling implementation prompt).

Identity binds schema/protocol/model/checkpoint/runtime/hardware/execution-context/device
identity plus a random per-run nonce. Measurement values are never part of identity.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass

from torch_dae.contracts import canonical_json_bytes
from torch_dae.profiling.contracts import TechnicalCardIdentity

_HUMAN_ID_DIGEST_LENGTH = 16


@dataclass(frozen=True)
class TechnicalCardIdentityInputs:
    """Every field that participates in Technical Card identity."""

    technical_card_schema_version: str
    model_card_id: str
    model_card_sha256: str
    checkpoint_sha256: str
    profiling_protocol_id: str
    profiling_protocol_version: str
    torch_dae_content_identity: str
    source_revision: str | None
    repository_dirty: bool | None
    hardware_fingerprint: str
    execution_context_fingerprint: str
    device_backend: str
    device_index: int | None
    nonce: str


def _canonical_payload(inputs: TechnicalCardIdentityInputs) -> bytes:
    return canonical_json_bytes(
        {
            "technical_card_schema_version": inputs.technical_card_schema_version,
            "model_card_id": inputs.model_card_id,
            "model_card_sha256": inputs.model_card_sha256,
            "checkpoint_sha256": inputs.checkpoint_sha256,
            "profiling_protocol_id": inputs.profiling_protocol_id,
            "profiling_protocol_version": inputs.profiling_protocol_version,
            "torch_dae_content_identity": inputs.torch_dae_content_identity,
            "source_revision": inputs.source_revision,
            "repository_dirty": inputs.repository_dirty,
            "hardware_fingerprint": inputs.hardware_fingerprint,
            "execution_context_fingerprint": inputs.execution_context_fingerprint,
            "device_backend": inputs.device_backend,
            "device_index": inputs.device_index,
            "nonce": inputs.nonce,
        }
    )


def generate_nonce() -> str:
    """Return a fresh random per-run nonce for Technical Card identity."""

    return secrets.token_hex(16)


def compute_identity_digest(inputs: TechnicalCardIdentityInputs) -> str:
    """Return the lowercase SHA-256 identity digest for the given identity inputs."""

    return hashlib.sha256(_canonical_payload(inputs)).hexdigest()


def human_technical_card_id(identity_digest_sha256: str) -> str:
    """Return the stable human-facing 'tc-<truncated digest>' form."""

    return f"tc-{identity_digest_sha256[:_HUMAN_ID_DIGEST_LENGTH]}"


def build_technical_card_identity(
    inputs: TechnicalCardIdentityInputs,
) -> TechnicalCardIdentity:
    """Build the full identity object (digest, nonce, human-facing ID) for one profiling run."""

    digest = compute_identity_digest(inputs)
    return TechnicalCardIdentity(
        technical_card_id=human_technical_card_id(digest),
        identity_digest_sha256=digest,
        nonce=inputs.nonce,
    )


def recompute_identity_digest(inputs: TechnicalCardIdentityInputs) -> str:
    """Recompute the identity digest deterministically for validation purposes."""

    return compute_identity_digest(inputs)


def validate_identity_recomputation(
    identity: TechnicalCardIdentity, inputs: TechnicalCardIdentityInputs
) -> None:
    """Raise ``ValueError`` if the stored identity does not match a recomputation of inputs."""

    recomputed = recompute_identity_digest(inputs)
    if recomputed != identity.identity_digest_sha256:
        raise ValueError(
            "Technical Card identity digest does not match recomputation from stored inputs"
        )
    if identity.technical_card_id != human_technical_card_id(recomputed):
        raise ValueError("Technical Card human-facing ID does not match its identity digest")
