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
    """Structural execution contract implemented by audio-model wrappers.

    Public waveform inputs have shape ``[B, C, T]``: ``B`` is batch size, ``C`` is channel count,
    and ``T`` is the number of samples. ``sample_rate`` is an integer frequency in hertz;
    ``valid_lengths``, when present, has shape ``[B]`` and contains valid sample counts before
    padding.

    Notes
    -----
    This protocol is structural and imports no tensor runtime. Wrapper construction, numerical
    determinism, preprocessing, checkpoint deserialization, and output values remain
    implementation-dependent. Implementations should import model-specific dependencies lazily.

    See Also
    --------
    torch_dae.core.preprocessing.WaveformInputContract
    torch_dae.core.outputs.AudioModelOutput
    torch_dae.core.outputs.EmbeddingOutput
    """

    @classmethod
    def from_random(cls, *, variant: str | None = None, **architecture_kwargs: object) -> Self:
        """Construct a randomly initialized wrapper when supported.

        Parameters
        ----------
        variant
            Optional integration-defined architecture variant.
        **architecture_kwargs
            Model-specific constructor values.

        Returns
        -------
        AudioModelProtocol
            A new wrapper instance.

        Raises
        ------
        UnsupportedCapabilityError
            If random initialization is not supported.

        Notes
        -----
        Randomness and determinism are controlled by the implementation and its tensor runtime.
        """

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: CheckpointSpec | str | Path | None = None,
        *,
        variant: str | None = None,
        **kwargs: object,
    ) -> Self:
        """Construct a wrapper and load pretrained weights.

        Parameters
        ----------
        checkpoint
            Checkpoint contract, identifier, filesystem path, or ``None`` for the integration's
            declared default.
        variant
            Optional integration-defined architecture variant.
        **kwargs
            Model-specific construction or loading options.

        Returns
        -------
        AudioModelProtocol
            A wrapper with checkpoint state loaded.

        Raises
        ------
        UnsupportedCapabilityError
            If checkpoint loading is unsupported.
        FileNotFoundError
            If a supplied local checkpoint does not exist.

        Notes
        -----
        Acquisition belongs to checkpoint management; deserialization belongs to the wrapper.
        """

    def load_checkpoint(
        self,
        checkpoint: CheckpointSpec | str | Path,
        *,
        strict: bool = True,
        map_location: str | object = "cpu",
    ) -> None:
        """Load checkpoint weights into an existing wrapper.

        Parameters
        ----------
        checkpoint
            Checkpoint specification, identifier, or filesystem path.
        strict
            Whether model-specific state-key matching must be exact.
        map_location
            Runtime-specific device or mapping; ``"cpu"`` is the portable default.

        Returns
        -------
        None
            The wrapper is updated in place.

        Raises
        ------
        UnsupportedCapabilityError
            If the wrapper does not support checkpoint loading.
        ValueError
            If checkpoint contents are incompatible with the wrapper.

        Notes
        -----
        The implementation controls deserialization and device semantics.
        """

    def preprocess(
        self,
        waveform: TensorLike,
        sample_rate: int,
        *,
        valid_lengths: TensorLike | None = None,
        allow_resample: bool = True,
    ) -> PreprocessingOutput:
        """Convert canonical waveform input to model-native inputs.

        Parameters
        ----------
        waveform
            Tensor-like audio shaped ``[B, C, T]``.
        sample_rate
            Sampling frequency in hertz.
        valid_lengths
            Optional tensor-like valid sample counts shaped ``[B]``.
        allow_resample
            Whether preprocessing may resample when the rate differs from the model requirement.

        Returns
        -------
        PreprocessingOutput
            Model-native input, effective sample rate, lengths, and named tensors.

        Raises
        ------
        ValueError
            If ranks, lengths, channels, or sampling rate violate the integration contract.

        Notes
        -----
        Resampling, padding, normalization, and numerical determinism are implementation-dependent.
        """

    def forward(
        self,
        waveform: TensorLike,
        sample_rate: int,
        *,
        valid_lengths: TensorLike | None = None,
    ) -> AudioModelOutput:
        """Run the differentiable forward path on canonical waveform input.

        Parameters
        ----------
        waveform
            Tensor-like audio shaped ``[B, C, T]``.
        sample_rate
            Sampling frequency in hertz.
        valid_lengths
            Optional valid sample counts shaped ``[B]``.

        Returns
        -------
        AudioModelOutput
            Primary output plus named tensors and optional length or native-output metadata.

        Raises
        ------
        ValueError
            If the input contract is invalid.

        Notes
        -----
        Tensor values, gradient behavior, and exact output shapes are wrapper-defined.
        """

    def predict_probability(
        self,
        waveform: TensorLike,
        sample_rate: int,
        *,
        valid_lengths: TensorLike | None = None,
    ) -> TensorLike:
        """Return the declared probability tensor for canonical waveform input.

        Parameters
        ----------
        waveform
            Tensor-like audio shaped ``[B, C, T]``.
        sample_rate
            Sampling frequency in hertz.
        valid_lengths
            Optional valid sample counts shaped ``[B]``.

        Returns
        -------
        TensorLike
            Integration-declared probabilities; the model card defines layout and dimensions.

        Raises
        ------
        UnsupportedCapabilityError
            If probability output is not a supported capability.
        ValueError
            If the waveform contract is invalid.
        """

    def available_embeddings(self) -> tuple[EmbeddingSpec, ...]:
        """Return the wrapper's ordered embedding declarations.

        Returns
        -------
        tuple of EmbeddingSpec
            All selectable embeddings, including the single declared default where supported.

        Notes
        -----
        Enumeration is metadata-only and should not execute a forward pass.
        """

    def compute_embedding(
        self,
        waveform: TensorLike,
        sample_rate: int,
        *,
        embedding_id: str | None = None,
        valid_lengths: TensorLike | None = None,
    ) -> EmbeddingOutput:
        """Compute a selected embedding from canonical waveform input.

        Parameters
        ----------
        waveform
            Tensor-like audio shaped ``[B, C, T]``.
        sample_rate
            Sampling frequency in hertz.
        embedding_id
            Declared embedding identifier, or ``None`` to select the default.
        valid_lengths
            Optional valid sample counts shaped ``[B]``.

        Returns
        -------
        EmbeddingOutput
            Selected tensor, layout, and optional temporal metadata. Shape is defined by its
            :class:`EmbeddingSpec`, not fixed by this protocol.

        Raises
        ------
        UnsupportedCapabilityError
            If embeddings are unsupported.
        KeyError
            If ``embedding_id`` is not declared.
        ValueError
            If waveform inputs violate the public contract.
        """
