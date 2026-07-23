"""Embedding contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from torch_dae.contracts import CanonicalId, StrictBaseModel


class EmbeddingSpec(StrictBaseModel):
    """Model-independent embedding declaration."""

    schema_version: Literal["1.0.0"]
    embedding_id: CanonicalId
    name: str
    description: str
    officially_defined: bool
    default: bool
    network_location: str
    layout: str
    dimension: int | None = Field(default=None, gt=0)
    granularity: str
    temporal_hop_seconds: float | None = Field(default=None, gt=0)
    pooling: str
    projection: str
    normalization: str
    task_head_relation: str
    dtype: str
    status: Literal["declared", "verified", "unsupported"]
    selection_rationale: str
    evidence_ids: tuple[str, ...] = ()
