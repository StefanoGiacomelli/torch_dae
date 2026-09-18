from __future__ import annotations

import numpy as np

from torch_dae.profiling.synthetic_input import (
    build_provenance,
    generate_white_noise,
    waveform_sha256,
)


def test_deterministic_for_same_seed() -> None:
    a = generate_white_noise(seed=42, batch_size=2, channel_count=1, sample_count=1000)
    b = generate_white_noise(seed=42, batch_size=2, channel_count=1, sample_count=1000)
    assert np.array_equal(a, b)


def test_different_seed_differs() -> None:
    a = generate_white_noise(seed=1, batch_size=1, channel_count=1, sample_count=1000)
    b = generate_white_noise(seed=2, batch_size=1, channel_count=1, sample_count=1000)
    assert not np.array_equal(a, b)


def test_dtype_is_float32() -> None:
    waveform = generate_white_noise(seed=1, batch_size=1, channel_count=1, sample_count=10)
    assert waveform.dtype == np.float32


def test_amplitude_bounds() -> None:
    waveform = generate_white_noise(seed=7, batch_size=4, channel_count=1, sample_count=50_000)
    assert waveform.min() >= -1.0
    assert waveform.max() < 1.0


def test_shape() -> None:
    waveform = generate_white_noise(seed=1, batch_size=3, channel_count=1, sample_count=500)
    assert waveform.shape == (3, 1, 500)


def test_rejects_non_positive_dimensions() -> None:
    import pytest

    with pytest.raises(ValueError):
        generate_white_noise(seed=1, batch_size=0, channel_count=1, sample_count=10)


def test_waveform_sha256_deterministic() -> None:
    waveform = generate_white_noise(seed=5, batch_size=1, channel_count=1, sample_count=100)
    assert waveform_sha256(waveform) == waveform_sha256(waveform.copy())


def test_provenance_records_seed_and_shape() -> None:
    waveform = generate_white_noise(seed=9, batch_size=1, channel_count=1, sample_count=16_000)
    provenance = build_provenance(
        seed=9,
        sample_rate=16_000,
        batch_size=1,
        channel_count=1,
        sample_count=16_000,
        waveform=waveform,
    )
    assert provenance.seed == 9
    assert provenance.sample_count == 16_000
    assert provenance.dtype == "float32"
    assert provenance.tensor_sha256 == waveform_sha256(waveform)
