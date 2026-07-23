"""Canonical public input contract helpers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class WaveformInputContract:
    """Represents waveform `[B,C,T]`, sample rate, and optional `[B]` valid lengths."""

    waveform_shape: str = "B,C,T"
    sample_rate_name: str = "sample_rate"
    valid_lengths_shape: str = "B"
    valid_lengths_optional: bool = True
