"""Dependency-free identities for the three accepted PANNs tuples."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PannsModelIdentity:
    """Describe one fixed PANNs variant/checkpoint integration identity."""

    model_id: str
    variant_id: str
    upstream_class: str
    checkpoint_filename: str
    wrapper_class: str
    sample_rate: int
    window_size: int
    hop_size: int
    mel_bins: int
    fmin: int
    fmax: int


PANNs_CNN14_16K_MAP_0438 = PannsModelIdentity(
    model_id="panns-cnn14-16k-map-0438",
    variant_id="panns-cnn14-16k",
    upstream_class="Cnn14_16k",
    checkpoint_filename="Cnn14_16k_mAP=0.438.pth",
    wrapper_class="PannsCnn14_16kMap0438",
    sample_rate=16_000,
    window_size=512,
    hop_size=160,
    mel_bins=64,
    fmin=50,
    fmax=8_000,
)

PANNs_RESNET38_MAP_0434 = PannsModelIdentity(
    model_id="panns-resnet38-map-0434",
    variant_id="panns-resnet38",
    upstream_class="ResNet38",
    checkpoint_filename="ResNet38_mAP=0.434.pth",
    wrapper_class="PannsResNet38Map0434",
    sample_rate=32_000,
    window_size=1_024,
    hop_size=320,
    mel_bins=64,
    fmin=50,
    fmax=14_000,
)

PANNs_WAVEGRAM_LOGMEL_CNN14_MAP_0439 = PannsModelIdentity(
    model_id="panns-wavegram-logmel-cnn14-map-0439",
    variant_id="panns-wavegram-logmel-cnn14",
    upstream_class="Wavegram_Logmel_Cnn14",
    checkpoint_filename="Wavegram_Logmel_Cnn14_mAP=0.439.pth",
    wrapper_class="PannsWavegramLogmelCnn14Map0439",
    sample_rate=32_000,
    window_size=1_024,
    hop_size=320,
    mel_bins=64,
    fmin=50,
    fmax=14_000,
)

PUBLIC_MODEL_IDENTITIES = (
    PANNs_CNN14_16K_MAP_0438,
    PANNs_RESNET38_MAP_0434,
    PANNs_WAVEGRAM_LOGMEL_CNN14_MAP_0439,
)
