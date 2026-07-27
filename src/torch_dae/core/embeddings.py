"""Embedding contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from torch_dae.contracts import CanonicalId, StrictBaseModel


class EmbeddingSpec(StrictBaseModel):
    """Declare one selectable model-independent embedding.

    Attributes
    ----------
    schema_version
        Contract version; currently ``1.0.0``.
    embedding_id, name, description
        Stable identity and human-readable semantics.
    officially_defined
        Whether upstream material explicitly defines this representation.
    default
        Whether this is the card's selected default embedding.
    network_location, layout
        Extraction location and ordered tensor-axis notation.
    dimension
        Optional positive feature dimension.
    granularity, temporal_hop_seconds
        Sequence/frame scope and optional frame spacing in seconds.
    pooling, projection, normalization, task_head_relation, dtype
        Transform and representation properties.
    status
        One of ``declared``, ``verified``, or ``unsupported``.
    selection_rationale, evidence_ids
        Justification and references to supporting evidence.

    Raises
    ------
    pydantic.ValidationError
        If identifiers, positive dimensions, positive hop duration, or allowed values are invalid.
    """

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
