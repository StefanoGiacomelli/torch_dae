"""Output dataclasses that avoid a root PyTorch import."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


class TensorLike(Protocol):
    """Structural tensor placeholder used without importing PyTorch."""

    @property
    def shape(self) -> object:
        """Tensor shape."""


@dataclass(frozen=True)
class AudioModelOutput:
    """Native differentiable model output contract; implemented in Phase 00."""

    primary: object
    tensors: Mapping[str, TensorLike]
    lengths: TensorLike | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    native_output: object | None = None


@dataclass(frozen=True)
class EmbeddingOutput:
    """Embedding output contract; tensors remain model-runtime objects."""

    embedding_id: str
    tensor: TensorLike
    layout: str
    lengths: TensorLike | None = None
    timestamps: TensorLike | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PreprocessingOutput:
    """Model-native preprocessing output for waveform, feature, token, or mapping inputs."""

    model_input: object
    sample_rate: int
    valid_lengths: TensorLike | None = None
    tensors: Mapping[str, TensorLike] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
