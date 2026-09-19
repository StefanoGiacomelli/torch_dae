from __future__ import annotations

from torch_dae.profiling.contracts import MinimumInputStatus
from torch_dae.profiling.minimum_input import (
    PROTOCOL_V1_MAX_PROBE_SECONDS,
    search_minimum_supported_samples,
)

SAMPLE_RATE = 1000


def test_monotonic_finds_exact_minimum() -> None:
    threshold = 3_500

    def accepts(n: int) -> bool:
        return n >= threshold

    result = search_minimum_supported_samples(accepts=accepts, sample_rate=SAMPLE_RATE)
    assert result.status == MinimumInputStatus.FOUND
    assert result.minimum_sample_count == threshold
    assert result.boundary_verified
    assert result.below_boundary_verified


def test_initial_probe_already_succeeds_low_threshold() -> None:
    threshold = 100  # below the initial 10s probe (10_000 samples at 1kHz)

    def accepts(n: int) -> bool:
        return n >= threshold

    result = search_minimum_supported_samples(accepts=accepts, sample_rate=SAMPLE_RATE)
    assert result.status == MinimumInputStatus.FOUND
    assert result.minimum_sample_count == threshold


def test_initial_probe_fails_requires_doubling() -> None:
    threshold = 50_000  # above the initial 10s probe (10_000 samples)

    def accepts(n: int) -> bool:
        return n >= threshold

    result = search_minimum_supported_samples(accepts=accepts, sample_rate=SAMPLE_RATE)
    assert result.status == MinimumInputStatus.FOUND
    assert result.minimum_sample_count == threshold


def test_minimum_of_one_sample() -> None:
    def accepts(n: int) -> bool:
        return n >= 1

    result = search_minimum_supported_samples(accepts=accepts, sample_rate=SAMPLE_RATE)
    assert result.status == MinimumInputStatus.FOUND
    assert result.minimum_sample_count == 1
    assert result.below_boundary_verified


def test_always_unsupported_up_to_maximum() -> None:
    def accepts(n: int) -> bool:
        return False

    result = search_minimum_supported_samples(accepts=accepts, sample_rate=SAMPLE_RATE)
    assert result.status in (
        MinimumInputStatus.ALWAYS_UNSUPPORTED,
        MinimumInputStatus.MAX_DURATION_EXHAUSTED,
    )


def test_maximum_duration_exhausted() -> None:
    max_samples = int(PROTOCOL_V1_MAX_PROBE_SECONDS * SAMPLE_RATE)

    def accepts(n: int) -> bool:
        return n > max_samples  # never true within the probe ceiling

    result = search_minimum_supported_samples(accepts=accepts, sample_rate=SAMPLE_RATE)
    assert result.status in (
        MinimumInputStatus.MAX_DURATION_EXHAUSTED,
        MinimumInputStatus.ALWAYS_UNSUPPORTED,
    )


def test_non_monotonic_support_is_reported_explicitly() -> None:
    # Accepts a narrow band only: n in [1000, 1500), rejects both below and above.
    def accepts(n: int) -> bool:
        return 1_000 <= n < 1_500

    result = search_minimum_supported_samples(accepts=accepts, sample_rate=SAMPLE_RATE)
    assert result.status == MinimumInputStatus.NON_MONOTONIC


def test_boundary_and_below_boundary_are_independently_verified() -> None:
    threshold = 777

    calls: list[int] = []

    def accepts(n: int) -> bool:
        calls.append(n)
        return n >= threshold

    result = search_minimum_supported_samples(accepts=accepts, sample_rate=SAMPLE_RATE)
    assert result.minimum_sample_count == threshold
    assert threshold in calls
    assert (threshold - 1) in calls


def test_immediate_lower_failure_at_minimum_of_two() -> None:
    threshold = 2

    def accepts(n: int) -> bool:
        return n >= threshold

    result = search_minimum_supported_samples(accepts=accepts, sample_rate=SAMPLE_RATE)
    assert result.status == MinimumInputStatus.FOUND
    assert result.minimum_sample_count == threshold
    assert result.boundary_verified
    assert result.below_boundary_verified


def test_lower_success_higher_failure_is_non_monotonic() -> None:
    # A small band succeeds, then everything above it fails -- the opposite of the expected
    # "bigger is more likely to be supported" shape.
    def accepts(n: int) -> bool:
        return n < 500

    result = search_minimum_supported_samples(accepts=accepts, sample_rate=SAMPLE_RATE)
    assert result.status == MinimumInputStatus.NON_MONOTONIC


def test_multiple_disconnected_valid_regions_is_non_monotonic() -> None:
    def accepts(n: int) -> bool:
        return (1_000 <= n < 1_500) or n >= 5_000

    result = search_minimum_supported_samples(accepts=accepts, sample_rate=SAMPLE_RATE)
    assert result.status == MinimumInputStatus.NON_MONOTONIC
    assert result.notes and "fabricated minimum" in result.notes


def test_probe_records_are_ordered_and_match_tested_set() -> None:
    threshold = 3_500

    def accepts(n: int) -> bool:
        return n >= threshold

    result = search_minimum_supported_samples(accepts=accepts, sample_rate=SAMPLE_RATE)
    assert [record.sample_count for record in result.probe_records] != []
    assert [record.probe_order for record in result.probe_records] == list(
        range(len(result.probe_records))
    )
    recorded_counts = {record.sample_count for record in result.probe_records}
    assert recorded_counts == set(result.probe_sample_counts_tested)
    for record in result.probe_records:
        assert record.succeeded == (record.sample_count >= threshold)


def test_probe_order_is_deterministic_across_repeated_runs() -> None:
    threshold = 12_345

    def accepts(n: int) -> bool:
        return n >= threshold

    first = search_minimum_supported_samples(accepts=accepts, sample_rate=SAMPLE_RATE)
    second = search_minimum_supported_samples(accepts=accepts, sample_rate=SAMPLE_RATE)
    first_order = [record.sample_count for record in first.probe_records]
    second_order = [record.sample_count for record in second.probe_records]
    assert first_order == second_order
