"""Environment verification contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, model_validator

from torch_dae.contracts import SHA256_PATTERN, CanonicalId, StrictBaseModel


class VerificationCheck(StrictBaseModel):
    """Single runtime verification check."""

    name: str
    status: Literal["passed", "failed", "unsupported"]
    details: str | None = None


class TensorDimension(StrictBaseModel):
    """Structured tensor dimension observation."""

    name: str
    size: int | None = Field(default=None, ge=0)
    dynamic: bool = False
    description: str | None = None


class TensorObservation(StrictBaseModel):
    """Structured tensor observation from runtime verification."""

    name: str
    role: str
    component_path: str
    shape: tuple[TensorDimension, ...]
    rank: int = Field(ge=0)
    dtype: str
    device: str
    lengths: str | None = None
    temporal_metadata: str | None = None

    @model_validator(mode="after")
    def rank_matches_shape(self) -> TensorObservation:
        if len(self.shape) != self.rank:
            raise ValueError("tensor observation rank must match shape length")
        return self


class VerificationReport(StrictBaseModel):
    """Runtime verification report contract."""

    schema_version: Literal["1.0.0"]
    report_id: CanonicalId
    model_card_id: CanonicalId
    environment_id: CanonicalId
    environment_fingerprint: Annotated[str, Field(pattern=SHA256_PATTERN)]
    created_at: datetime
    platform: str
    device: str
    checkpoint_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    input_contracts: tuple[str, ...]
    tensor_observations: tuple[TensorObservation, ...]
    embedding_results: tuple[str, ...]
    passed_capabilities: tuple[str, ...]
    unsupported_capabilities: tuple[str, ...]
    known_limitations: tuple[str, ...]
    checks: tuple[VerificationCheck, ...]
