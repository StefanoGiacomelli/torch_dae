"""Output dataclasses that avoid a root PyTorch import."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


class TensorLike(Protocol):
    """Minimal structural tensor type used by the model-agnostic control plane.

    Attributes
    ----------
    shape
        Runtime-specific shape object. Public contracts describe its axes separately.

    Notes
    -----
    No numerical, dtype, device, gradient, or mutation behavior is implied.
    """

    @property
    def shape(self) -> object:
        """Tensor shape."""


@dataclass(frozen=True)
class AudioModelOutput:
    """Carry native forward outputs without constraining a tensor runtime.

    Attributes
    ----------
    primary
        Integration-defined primary result.
    tensors
        Named differentiable tensor outputs.
    lengths
        Optional valid output lengths, normally one value per batch item.
    metadata
        Non-tensor descriptive values.
    native_output
        Optional unmodified upstream return object.

    Notes
    -----
    The model card, rather than this container, defines ranks, layouts, units, and semantics.
    """

    primary: object
    tensors: Mapping[str, TensorLike]
    lengths: TensorLike | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    native_output: object | None = None


@dataclass(frozen=True)
class EmbeddingOutput:
    """Carry one selected embedding and its temporal metadata.

    Attributes
    ----------
    embedding_id
        Identifier matching an :class:`~torch_dae.core.embeddings.EmbeddingSpec`.
    tensor
        Runtime tensor whose axes are described by ``layout``.
    layout
        Ordered dimension labels such as ``B,D`` or ``B,T,D``.
    lengths
        Optional valid output-frame lengths shaped ``[B]``.
    timestamps
        Optional time coordinates in implementation-declared units.
    metadata
        Additional non-tensor values.

    Notes
    -----
    No fixed embedding shape is assumed. Batch, temporal, and feature axes are declared per
    embedding and verified at runtime.
    """

    embedding_id: str
    tensor: TensorLike
    layout: str
    lengths: TensorLike | None = None
    timestamps: TensorLike | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PreprocessingOutput:
    """Carry the model-native result of canonical waveform preprocessing.

    Attributes
    ----------
    model_input
        Object consumed by the wrapper's forward implementation.
    sample_rate
        Effective sampling frequency in hertz after any allowed resampling.
    valid_lengths
        Optional valid lengths in the model-input domain.
    tensors
        Named intermediate tensor-like values.
    metadata
        Preprocessing decisions and other non-tensor values.

    Notes
    -----
    This immutable container performs no preprocessing itself.
    """

    model_input: object
    sample_rate: int
    valid_lengths: TensorLike | None = None
    tensors: Mapping[str, TensorLike] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
