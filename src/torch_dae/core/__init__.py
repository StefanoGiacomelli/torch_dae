"""Stable generic execution, output, capability, and checkpoint contracts."""

from torch_dae.core.capabilities import Capability, ModelCapabilities
from torch_dae.core.checkpoint import (
    CheckpointManager,
    CheckpointMaterializationRecord,
    CheckpointSourceType,
    CheckpointSpec,
    LicenseRecord,
    ResolvedCheckpoint,
)
from torch_dae.core.embeddings import EmbeddingSpec
from torch_dae.core.errors import FeatureNotAvailableError, UnsupportedCapabilityError
from torch_dae.core.model import AudioModelProtocol
from torch_dae.core.outputs import (
    AudioModelOutput,
    EmbeddingOutput,
    PreprocessingOutput,
    TensorLike,
)
from torch_dae.core.preprocessing import WaveformInputContract

__all__ = [
    "AudioModelOutput",
    "AudioModelProtocol",
    "Capability",
    "CheckpointManager",
    "CheckpointMaterializationRecord",
    "CheckpointSourceType",
    "CheckpointSpec",
    "EmbeddingOutput",
    "EmbeddingSpec",
    "FeatureNotAvailableError",
    "LicenseRecord",
    "ModelCapabilities",
    "PreprocessingOutput",
    "ResolvedCheckpoint",
    "TensorLike",
    "UnsupportedCapabilityError",
    "WaveformInputContract",
]
