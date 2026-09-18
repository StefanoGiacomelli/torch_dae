"""Timing statistics engine for Profiling v1 (Section 14).

Raw observations are 50 integer-nanosecond steady-state latencies. Summary statistics must be
exactly recomputable from those raw values; this module is the single source of truth for that
recomputation so the worker (measurement) and the validator (re-check) share identical arithmetic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from torch_dae.profiling.contracts import TimingSummary

CANONICAL_WARMUP_COUNT = 10
CANONICAL_MEASURED_COUNT = 50


def _percentile(sorted_values: Sequence[int], fraction: float) -> float:
    """Linear-interpolated percentile, matching ``numpy.percentile`` default behavior."""

    if not sorted_values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[int(position)])
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def summarize_timing(
    raw_ns: Sequence[int],
    *,
    batch_size: int,
    input_duration_seconds: float,
) -> TimingSummary:
    """Compute the canonical summary statistics for one condition's 50 raw observations.

    Parameters
    ----------
    raw_ns
        Exactly ``CANONICAL_MEASURED_COUNT`` integer-nanosecond latency observations.
    batch_size
        Number of items processed per measured inference call.
    input_duration_seconds
        Wall-clock audio duration represented by one item in the batch.
    """

    if len(raw_ns) != CANONICAL_MEASURED_COUNT:
        raise ValueError(
            f"canonical Profiling v1 requires exactly {CANONICAL_MEASURED_COUNT} measured "
            f"observations; received {len(raw_ns)}"
        )
    if any(value <= 0 for value in raw_ns):
        raise ValueError("all raw timing observations must be strictly positive nanoseconds")

    ordered = sorted(raw_ns)
    n = len(ordered)
    mean = sum(ordered) / n
    variance = sum((value - mean) ** 2 for value in ordered) / n
    stddev = math.sqrt(variance)
    coefficient_of_variation = stddev / mean if mean else 0.0

    mean_seconds = mean / 1e9
    processed_audio_seconds = batch_size * input_duration_seconds
    throughput_items_per_second = (batch_size / mean_seconds) if mean_seconds > 0 else 0.0
    forward_calls_per_second = (1.0 / mean_seconds) if mean_seconds > 0 else 0.0
    real_time_factor = (
        (mean_seconds / processed_audio_seconds) if processed_audio_seconds > 0 else 0.0
    )
    speed_factor = (processed_audio_seconds / mean_seconds) if mean_seconds > 0 else 0.0

    return TimingSummary(
        sample_count=50,
        mean_ns=mean,
        stddev_ns=stddev,
        min_ns=ordered[0],
        max_ns=ordered[-1],
        p50_ns=_percentile(ordered, 0.50),
        p90_ns=_percentile(ordered, 0.90),
        p95_ns=_percentile(ordered, 0.95),
        p99_ns=_percentile(ordered, 0.99),
        coefficient_of_variation=coefficient_of_variation,
        throughput_items_per_second=throughput_items_per_second,
        forward_calls_per_second=forward_calls_per_second,
        real_time_factor=real_time_factor,
        speed_factor=speed_factor,
    )


def verify_timing_summary(
    summary: TimingSummary,
    raw_ns: Sequence[int],
    *,
    batch_size: int,
    input_duration_seconds: float,
    relative_tolerance: float = 1e-9,
) -> bool:
    """Return whether ``summary`` is consistent with a fresh recomputation from ``raw_ns``."""

    recomputed = summarize_timing(
        raw_ns, batch_size=batch_size, input_duration_seconds=input_duration_seconds
    )
    if not math.isclose(
        recomputed.speed_factor,
        1.0 / recomputed.real_time_factor,
        rel_tol=relative_tolerance,
        abs_tol=1e-6,
    ):
        raise AssertionError(
            "internal invariant violated: speed_factor must be the reciprocal of "
            "real_time_factor for a successful positive-duration condition"
        )
    for field_name in TimingSummary.model_fields:
        expected = getattr(recomputed, field_name)
        actual = getattr(summary, field_name)
        if isinstance(expected, float):
            if not math.isclose(expected, actual, rel_tol=relative_tolerance, abs_tol=1e-6):
                return False
        elif expected != actual:
            return False
    return True
