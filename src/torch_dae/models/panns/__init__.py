"""Three fixed PANNs production identities with lazy tensor-runtime imports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from torch_dae.models.panns.identities import (
    PUBLIC_MODEL_IDENTITIES,
    PANNs_CNN14_16K_MAP_0438,
    PANNs_RESNET38_MAP_0434,
    PANNs_WAVEGRAM_LOGMEL_CNN14_MAP_0439,
    PannsModelIdentity,
)
from torch_dae.models.panns.labels import AudioSetLabel, class_labels, load_audioset_labels

if TYPE_CHECKING:
    from torch_dae.models.panns.model import (
        PannsCnn14_16kMap0438,
        PannsResNet38Map0434,
        PannsWavegramLogmelCnn14Map0439,
    )

_RUNTIME_EXPORTS = {
    "PannsCnn14_16kMap0438",
    "PannsResNet38Map0434",
    "PannsWavegramLogmelCnn14Map0439",
}


def __getattr__(name: str) -> Any:
    if name not in _RUNTIME_EXPORTS:
        raise AttributeError(name)
    module = import_module("torch_dae.models.panns.model")
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted((*globals(), *_RUNTIME_EXPORTS))


__all__ = [
    "PUBLIC_MODEL_IDENTITIES",
    "AudioSetLabel",
    "PANNs_CNN14_16K_MAP_0438",
    "PANNs_RESNET38_MAP_0434",
    "PANNs_WAVEGRAM_LOGMEL_CNN14_MAP_0439",
    "PannsCnn14_16kMap0438",
    "PannsModelIdentity",
    "PannsResNet38Map0434",
    "PannsWavegramLogmelCnn14Map0439",
    "class_labels",
    "load_audioset_labels",
]
