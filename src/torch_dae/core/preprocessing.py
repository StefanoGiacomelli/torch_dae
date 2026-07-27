"""Canonical public input contract helpers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class WaveformInputContract:
    """Describe the canonical model-independent waveform interface.

    Attributes
    ----------
    waveform_shape
        Axis labels ``B,C,T`` for batch, channel, and sample.
    sample_rate_name
        Public parameter whose integer value is measured in hertz.
    valid_lengths_shape
        Axis label ``B`` for optional per-example valid sample counts.
    valid_lengths_optional
        Whether callers may omit valid lengths.

    Notes
    -----
    This frozen value object documents shape names; it does not validate a tensor or resample audio.
    """

    waveform_shape: str = "B,C,T"
    sample_rate_name: str = "sample_rate"
    valid_lengths_shape: str = "B"
    valid_lengths_optional: bool = True
