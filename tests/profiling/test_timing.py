from __future__ import annotations

import pytest

from torch_dae.profiling.timing import (
    CANONICAL_MEASURED_COUNT,
    CANONICAL_WARMUP_COUNT,
    summarize_timing,
    verify_timing_summary,
)


def _raw(base_ns: int = 1_000_000) -> list[int]:
    return [base_ns + i * 1000 for i in range(CANONICAL_MEASURED_COUNT)]


def test_requires_exactly_50_samples() -> None:
    with pytest.raises(ValueError):
        summarize_timing([1, 2, 3], batch_size=1, input_duration_seconds=1.0)


def test_rejects_non_positive_values() -> None:
    raw = _raw()
    raw[0] = 0
    with pytest.raises(ValueError):
        summarize_timing(raw, batch_size=1, input_duration_seconds=1.0)


def test_min_max_correct() -> None:
    raw = _raw()
    summary = summarize_timing(raw, batch_size=1, input_duration_seconds=1.0)
    assert summary.min_ns == min(raw)
    assert summary.max_ns == max(raw)


def test_percentiles_monotonic() -> None:
    raw = _raw()
    summary = summarize_timing(raw, batch_size=1, input_duration_seconds=1.0)
    assert summary.p50_ns <= summary.p90_ns <= summary.p95_ns <= summary.p99_ns <= summary.max_ns
    assert summary.min_ns <= summary.p50_ns


def test_throughput_and_rtf() -> None:
    raw = [10_000_000] * CANONICAL_MEASURED_COUNT  # 10ms per call
    summary = summarize_timing(raw, batch_size=2, input_duration_seconds=1.0)
    # L = 0.01s, B = 2, D = 1.0s -> processed_audio_seconds = 2.0
    assert summary.throughput_items_per_second == pytest.approx(200.0)  # B / L
    assert summary.forward_calls_per_second == pytest.approx(100.0)  # 1 / L
    assert summary.real_time_factor == pytest.approx(0.005)  # L / (B * D)
    assert summary.speed_factor == pytest.approx(200.0)  # (B * D) / L


def test_speed_factor_is_reciprocal_of_real_time_factor() -> None:
    raw = [7_654_321 + i * 913 for i in range(CANONICAL_MEASURED_COUNT)]
    summary = summarize_timing(raw, batch_size=3, input_duration_seconds=2.5)
    assert summary.speed_factor == pytest.approx(1.0 / summary.real_time_factor, rel=1e-9)
    assert summary.real_time_factor == pytest.approx(1.0 / summary.speed_factor, rel=1e-9)


def test_speed_factor_differs_from_throughput_when_duration_is_not_unity() -> None:
    raw = [10_000_000] * CANONICAL_MEASURED_COUNT
    summary = summarize_timing(raw, batch_size=4, input_duration_seconds=2.0)
    assert summary.speed_factor != summary.throughput_items_per_second


def test_constant_values_zero_stddev() -> None:
    raw = [5_000_000] * CANONICAL_MEASURED_COUNT
    summary = summarize_timing(raw, batch_size=1, input_duration_seconds=1.0)
    assert summary.stddev_ns == 0.0
    assert summary.coefficient_of_variation == 0.0


def test_recomputation_matches() -> None:
    raw = _raw()
    summary = summarize_timing(raw, batch_size=1, input_duration_seconds=1.0)
    assert verify_timing_summary(summary, raw, batch_size=1, input_duration_seconds=1.0)


def test_recomputation_detects_tamper() -> None:
    raw = _raw()
    summary = summarize_timing(raw, batch_size=1, input_duration_seconds=1.0)
    tampered = summary.model_copy(update={"mean_ns": summary.mean_ns * 2})
    assert not verify_timing_summary(tampered, raw, batch_size=1, input_duration_seconds=1.0)


def test_protocol_constants() -> None:
    assert CANONICAL_WARMUP_COUNT == 10
    assert CANONICAL_MEASURED_COUNT == 50
