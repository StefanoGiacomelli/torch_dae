"""Deterministic synthetic white-noise input for Profiling v1 (Section 9 / protocol.md).

Profiling v1 uses only deterministic seeded IID uniform white noise. No speech, music,
soundscape, dataset, or external audio asset participates in protocol v1. Generation and
allocation happen outside canonical timed inference.
"""

from __future__ import annotations

import hashlib

import numpy as np

from torch_dae.profiling.contracts import SyntheticInputProvenance

PRNG_IMPLEMENTATION = "numpy.random.Generator(PCG64)"


def generate_white_noise(
    *,
    seed: int,
    batch_size: int,
    channel_count: int,
    sample_count: int,
) -> np.ndarray:
    """Return deterministic ``float32`` IID uniform samples in ``[-1, 1)`` with shape [B,C,T].

    Uses ``numpy.random.Generator(PCG64)`` seeded explicitly; the same seed and shape always
    produce byte-identical output (verified by ``tests/profiling/test_synthetic_input.py``).
    """

    if batch_size < 1 or channel_count < 1 or sample_count < 1:
        raise ValueError("batch_size, channel_count, and sample_count must all be >= 1")
    rng = np.random.Generator(np.random.PCG64(seed))
    waveform = rng.uniform(low=-1.0, high=1.0, size=(batch_size, channel_count, sample_count))
    return waveform.astype(np.float32, copy=False)


def waveform_sha256(waveform: np.ndarray) -> str:
    """Return the SHA-256 identity of a generated waveform's raw bytes."""

    return hashlib.sha256(np.ascontiguousarray(waveform).tobytes()).hexdigest()


def build_provenance(
    *,
    seed: int,
    sample_rate: int,
    batch_size: int,
    channel_count: int,
    sample_count: int,
    waveform: np.ndarray | None = None,
) -> SyntheticInputProvenance:
    """Build the recorded provenance for one generated synthetic input tensor."""

    return SyntheticInputProvenance(
        distribution="uniform",
        low=-1.0,
        high=1.0,
        dtype="float32",
        prng_implementation=PRNG_IMPLEMENTATION,
        seed=seed,
        sample_rate=sample_rate,
        sample_count=sample_count,
        batch_size=batch_size,
        channel_count=channel_count,
        tensor_sha256=waveform_sha256(waveform) if waveform is not None else None,
    )
