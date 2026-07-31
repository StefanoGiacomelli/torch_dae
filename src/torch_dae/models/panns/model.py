"""Runtime PANNs adapters for three fixed model/checkpoint identities."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar, Self, cast

from torch_dae.core.checkpoint import CheckpointSourceType, CheckpointSpec
from torch_dae.core.embeddings import EmbeddingSpec
from torch_dae.core.errors import MissingModelRuntimeDependencyError, UnsupportedCapabilityError
from torch_dae.core.outputs import AudioModelOutput, EmbeddingOutput, PreprocessingOutput
from torch_dae.models.panns.identities import (
    PANNs_CNN14_16K_MAP_0438,
    PANNs_RESNET38_MAP_0434,
    PANNs_WAVEGRAM_LOGMEL_CNN14_MAP_0439,
    PannsModelIdentity,
)
from torch_dae.models.panns.labels import class_labels

try:
    import torch  # type: ignore[import-not-found]
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by root subprocess tests
    raise MissingModelRuntimeDependencyError(
        "PANNs execution requires its isolated model environment with "
        "torch==2.13.0 and torchlibrosa==0.1.0; the root control-plane environment "
        "intentionally does not install them."
    ) from exc

_CLASS_COUNT = 527
_DEFAULT_EMBEDDING_ID = "panns-official-post-fc1-embedding"
_MINIMUM_SAMPLE_COUNTS = {
    "panns-cnn14-16k-map-0438": 4_960,
    "panns-resnet38-map-0434": 9_920,
    "panns-wavegram-logmel-cnn14-map-0439": 10_236,
}
_WAVEGRAM_ALIGNMENT_MODULUS = 640
_WAVEGRAM_INCOMPATIBLE_REMAINDER = range(320, 636)


class _PannsAudioTagger(nn.Module):  # type: ignore[misc]
    """Shared adapter that preserves native PANNs frontend and forward semantics."""

    identity: ClassVar[PannsModelIdentity]

    def __init__(self) -> None:
        super().__init__()
        try:
            from torch_dae.models.panns._vendor import models as upstream
        except ModuleNotFoundError as exc:
            raise MissingModelRuntimeDependencyError(
                "PANNs execution requires its isolated model environment with "
                "numpy==2.4.6, torch==2.13.0, and torchlibrosa==0.1.0."
            ) from exc

        upstream_class = getattr(upstream, self.identity.upstream_class)
        self.upstream_model = upstream_class(
            sample_rate=self.identity.sample_rate,
            window_size=self.identity.window_size,
            hop_size=self.identity.hop_size,
            mel_bins=self.identity.mel_bins,
            fmin=self.identity.fmin,
            fmax=self.identity.fmax,
            classes_num=_CLASS_COUNT,
        )

    @classmethod
    def _validate_variant(cls, variant: str | None) -> None:
        if variant is not None and variant != cls.identity.variant_id:
            raise ValueError(
                f"{cls.__name__} has fixed variant {cls.identity.variant_id!r}; "
                f"received {variant!r}"
            )

    @classmethod
    def from_random(cls, *, variant: str | None = None, **architecture_kwargs: object) -> Self:
        """Construct the fixed architecture with random upstream initialization."""

        cls._validate_variant(variant)
        if architecture_kwargs:
            raise TypeError(
                f"{cls.__name__} has fixed architecture parameters; unexpected keys: "
                f"{sorted(architecture_kwargs)}"
            )
        return cls()

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: CheckpointSpec | str | Path | None = None,
        *,
        variant: str | None = None,
        **kwargs: object,
    ) -> Self:
        """Construct the fixed architecture and load an explicitly materialized checkpoint."""

        cls._validate_variant(variant)
        if kwargs:
            raise TypeError(f"unexpected construction keys: {sorted(kwargs)}")
        if checkpoint is None:
            raise ValueError(
                "checkpoint must be explicitly materialized and supplied; this wrapper never "
                "downloads a default checkpoint"
            )
        model = cls()
        model.load_checkpoint(checkpoint)
        return model

    @staticmethod
    def _checkpoint_path(checkpoint: CheckpointSpec | str | Path) -> Path:
        if isinstance(checkpoint, CheckpointSpec):
            if checkpoint.source_type is not CheckpointSourceType.LOCAL_PATH:
                raise ValueError(
                    "non-local CheckpointSpec values must be materialized by CheckpointManager "
                    "before wrapper loading"
                )
            if checkpoint.local_path is None:  # enforced by CheckpointSpec, retained defensively
                raise ValueError("local checkpoint specification has no local_path")
            return Path(checkpoint.local_path)
        return Path(checkpoint)

    def load_checkpoint(
        self,
        checkpoint: CheckpointSpec | str | Path,
        *,
        strict: bool = True,
        map_location: str | torch.device = "cpu",
    ) -> None:
        """Load the upstream ``checkpoint['model']`` state-dictionary convention."""

        path = self._checkpoint_path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location=map_location, weights_only=True)
        if not isinstance(payload, Mapping) or "model" not in payload:
            raise ValueError("PANNs checkpoint must be a mapping containing the 'model' key")
        state_dict = payload["model"]
        if not isinstance(state_dict, Mapping):
            raise ValueError("PANNs checkpoint 'model' value must be a state-dictionary mapping")
        self.upstream_model.load_state_dict(state_dict, strict=strict)

    def preprocess(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        *,
        valid_lengths: torch.Tensor | None = None,
        allow_resample: bool = True,
    ) -> PreprocessingOutput:
        """Validate ``[B,1,T]`` input and convert it to upstream ``[B,T]`` layout."""

        del allow_resample
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("waveform must be a torch.Tensor")
        if waveform.ndim != 3:
            raise ValueError("PANNs waveform must have shape [B,C,T] with exactly three axes")
        if waveform.shape[1] != 1:
            raise ValueError(
                "PANNs requires mono waveform input with C=1; no downmixing is applied"
            )
        if not torch.is_floating_point(waveform):
            raise TypeError("PANNs waveform must use a floating-point dtype")
        if type(sample_rate) is not int or sample_rate != self.identity.sample_rate:
            raise ValueError(
                f"{self.identity.model_id} requires sample_rate={self.identity.sample_rate}; "
                "automatic resampling is not supported"
            )

        batch_size, _, sample_count = waveform.shape
        minimum_samples = _MINIMUM_SAMPLE_COUNTS[self.identity.model_id]
        if sample_count < minimum_samples:
            raise ValueError(
                f"{self.identity.model_id} requires at least {minimum_samples} waveform samples; "
                f"received T={sample_count}. No automatic padding is applied."
            )
        if self.identity.model_id == "panns-wavegram-logmel-cnn14-map-0439":
            remainder = sample_count % _WAVEGRAM_ALIGNMENT_MODULUS
            if remainder in _WAVEGRAM_INCOMPATIBLE_REMAINDER:
                raise ValueError(
                    "panns-wavegram-logmel-cnn14-map-0439 requires matching Wavegram and "
                    "log-mel branch lengths: T % 640 must be 0..319 or 636..639; "
                    f"received T={sample_count} with remainder {remainder}. No automatic "
                    "padding, cropping, or truncation is applied."
                )
        if valid_lengths is not None:
            if not isinstance(valid_lengths, torch.Tensor):
                raise TypeError("valid_lengths must be a torch.Tensor when supplied")
            if valid_lengths.ndim != 1 or valid_lengths.shape[0] != batch_size:
                raise ValueError("valid_lengths must have shape [B]")
            expected = torch.full_like(valid_lengths, sample_count)
            if not torch.equal(valid_lengths, expected):
                raise UnsupportedCapabilityError(
                    "PANNs integration does not yet define padded-batch semantics; every "
                    "valid_lengths value must equal the waveform time dimension T"
                )

        model_input = waveform[:, 0, :]
        return PreprocessingOutput(
            model_input=model_input,
            sample_rate=sample_rate,
            valid_lengths=valid_lengths,
            tensors={"waveform": waveform},
            metadata={
                "channel_policy": "strict_mono",
                "resampled": False,
                "normalized": False,
                "padded_or_truncated": False,
            },
        )

    def forward(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        *,
        valid_lengths: torch.Tensor | None = None,
    ) -> AudioModelOutput:
        """Return raw logits while retaining native sigmoid probabilities and embedding."""

        preprocessed = self.preprocess(
            waveform,
            sample_rate,
            valid_lengths=valid_lengths,
            allow_resample=False,
        )
        native_output = self.upstream_model(cast(torch.Tensor, preprocessed.model_input))
        logits = native_output["logits"]
        probabilities = native_output["clipwise_output"]
        embedding = native_output["embedding"]
        return AudioModelOutput(
            primary=logits,
            tensors={
                "logits": logits,
                "probabilities": probabilities,
                "embedding": embedding,
            },
            lengths=None,
            metadata={
                "model_id": self.identity.model_id,
                "probability_activation": "sigmoid",
                "class_count": _CLASS_COUNT,
            },
            native_output=native_output,
        )

    def predict_probability(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        *,
        valid_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return native multi-label sigmoid probabilities without thresholding."""

        output = self.forward(waveform, sample_rate, valid_lengths=valid_lengths)
        return cast(torch.Tensor, output.tensors["probabilities"])

    def available_embeddings(self) -> tuple[EmbeddingSpec, ...]:
        """Return the selected upstream-named post-fc1 clip embedding."""

        return (
            EmbeddingSpec(
                schema_version="1.0.0",
                embedding_id=_DEFAULT_EMBEDDING_ID,
                name="PANNs post-fc1 embedding",
                description=(
                    "Upstream embedding output after ReLU(fc1) and its dropout call, immediately "
                    "before the AudioSet classifier; dropout is identity in eval mode."
                ),
                officially_defined=True,
                default=True,
                network_location="upstream forward output 'embedding'; post-fc1, pre-classifier",
                layout="B,D",
                dimension=2_048,
                granularity="clipwise",
                temporal_hop_seconds=None,
                pooling="global temporal max plus mean before fc1",
                projection="2048-to-2048 fc1 followed by ReLU",
                normalization="none",
                task_head_relation="immediately before fc_audioset",
                dtype="floating_point_runtime_dtype",
                status="declared",
                selection_rationale=(
                    "Selected by the accepted workflow because upstream names and returns this "
                    "representation for transfer learning."
                ),
                evidence_ids=("ev-models-cnn14-16k", "ev-models-resnet38", "ev-models-wavegram"),
            ),
        )

    def compute_embedding(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        *,
        embedding_id: str | None = None,
        valid_lengths: torch.Tensor | None = None,
    ) -> EmbeddingOutput:
        """Return the selected upstream embedding from the same native forward path."""

        selected = embedding_id or _DEFAULT_EMBEDDING_ID
        if selected != _DEFAULT_EMBEDDING_ID:
            raise KeyError(selected)
        output = self.forward(waveform, sample_rate, valid_lengths=valid_lengths)
        return EmbeddingOutput(
            embedding_id=selected,
            tensor=output.tensors["embedding"],
            layout="B,D",
            lengths=None,
            timestamps=None,
            metadata={
                "dropout_position": "after_relu_fc1",
                "eval_behavior": "dropout_is_identity",
            },
        )

    @property
    def class_labels(self) -> tuple[str, ...]:
        """Return the 527 AudioSet display names in classifier-index order."""

        return class_labels()

    @property
    def default_embedding_id(self) -> str:
        """Return the accepted default embedding identifier."""

        return _DEFAULT_EMBEDDING_ID


class PannsCnn14_16kMap0438(_PannsAudioTagger):
    """Fixed Cnn14_16k architecture for ``Cnn14_16k_mAP=0.438.pth``."""

    identity = PANNs_CNN14_16K_MAP_0438


class PannsResNet38Map0434(_PannsAudioTagger):
    """Fixed ResNet38 architecture for ``ResNet38_mAP=0.434.pth``."""

    identity = PANNs_RESNET38_MAP_0434


class PannsWavegramLogmelCnn14Map0439(_PannsAudioTagger):
    """Fixed Wavegram_Logmel_Cnn14 architecture for its mAP=0.439 checkpoint."""

    identity = PANNs_WAVEGRAM_LOGMEL_CNN14_MAP_0439


__all__ = [
    "PannsCnn14_16kMap0438",
    "PannsResNet38Map0434",
    "PannsWavegramLogmelCnn14Map0439",
]
