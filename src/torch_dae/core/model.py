"""Public audio model API definition without importing PyTorch."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Self

from torch_dae.core.checkpoint import CheckpointSpec
from torch_dae.core.embeddings import EmbeddingSpec
from torch_dae.core.outputs import (
    AudioModelOutput,
    EmbeddingOutput,
    PreprocessingOutput,
    TensorLike,
)


class AudioModelProtocol(Protocol):
    """Frozen Phase 00 API expected from future PyTorch wrappers."""

    @classmethod
    def from_random(cls, *, variant: str | None = None, **architecture_kwargs: object) -> Self:
        """Construct from random initialization when supported by the integration."""

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: CheckpointSpec | str | Path | None = None,
        *,
        variant: str | None = None,
        **kwargs: object,
    ) -> Self:
        """Construct and load a pretrained checkpoint in a model environment."""

    def load_checkpoint(
        self,
        checkpoint: CheckpointSpec | str | Path,
        *,
        strict: bool = True,
        map_location: str | object = "cpu",
    ) -> None:
        """Load checkpoint weights; runtime implementation is model-specific."""

    def preprocess(
        self,
        waveform: TensorLike,
        sample_rate: int,
        *,
        valid_lengths: TensorLike | None = None,
        allow_resample: bool = True,
    ) -> PreprocessingOutput:
        """Preprocess public `[B,C,T]` waveform input."""

    def forward(
        self,
        waveform: TensorLike,
        sample_rate: int,
        *,
        valid_lengths: TensorLike | None = None,
    ) -> AudioModelOutput:
        """Return native differentiable outputs."""

    def predict_probability(
        self,
        waveform: TensorLike,
        sample_rate: int,
        *,
        valid_lengths: TensorLike | None = None,
    ) -> TensorLike:
        """Return only a probability tensor or raise UnsupportedCapabilityError."""

    def available_embeddings(self) -> tuple[EmbeddingSpec, ...]:
        """Return all declared embeddings."""

    def compute_embedding(
        self,
        waveform: TensorLike,
        sample_rate: int,
        *,
        embedding_id: str | None = None,
        valid_lengths: TensorLike | None = None,
    ) -> EmbeddingOutput:
        """Return the requested embedding output."""
