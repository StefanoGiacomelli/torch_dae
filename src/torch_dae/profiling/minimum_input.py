"""Empirical minimum supported input search (Section 11 / protocol.md).

The search is expressed purely in terms of an injected acceptance callback, so it is testable
with synthetic mocked acceptance functions independent of any real model (Section 11 explicitly
requires this before relying on real models).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from torch_dae.profiling.contracts import (
    MinimumInputProbeRecord,
    MinimumInputSearchResult,
    MinimumInputStatus,
)

PROTOCOL_V1_MAX_PROBE_SECONDS = 120.0
_INITIAL_PROBE_SECONDS = 10.0

AcceptanceFn = Callable[[int], bool]


@dataclass
class _SearchTrace:
    tested: dict[int, bool] = field(default_factory=dict)
    order: list[int] = field(default_factory=list)

    def probe(self, accepts: AcceptanceFn, sample_count: int) -> bool:
        if sample_count not in self.tested:
            self.tested[sample_count] = accepts(sample_count)
            self.order.append(sample_count)
        return self.tested[sample_count]

    def records(self) -> tuple[MinimumInputProbeRecord, ...]:
        return tuple(
            MinimumInputProbeRecord(
                probe_order=index, sample_count=sample_count, succeeded=self.tested[sample_count]
            )
            for index, sample_count in enumerate(self.order)
        )


def _is_monotonic(trace: _SearchTrace) -> bool:
    """A support boundary is monotonic if failure never follows success in sample-count order.

    Larger sample counts are expected to succeed once the minimum is reached, so a failure at a
    sample count is only a violation if a *smaller* sample count already succeeded.
    """

    ordered = sorted(trace.tested.items())
    seen_success = False
    for _, ok in ordered:
        if ok:
            seen_success = True
        elif seen_success:
            return False
    return True


def search_minimum_supported_samples(
    *,
    accepts: AcceptanceFn,
    sample_rate: int,
    batch_size: int = 1,
) -> MinimumInputSearchResult:
    """Empirically discover the minimum accepted sample count for one device/backend.

    Parameters
    ----------
    accepts
        Callback returning whether the model accepts (constructs and runs) the given sample
        count at ``batch_size``. Must be deterministic for a stable search.
    sample_rate
        Sample rate used to convert seconds probes into integer sample counts.
    batch_size
        Fixed at 1 for the canonical minimum-input search (Section 11).
    """

    trace = _SearchTrace()
    max_samples = int(PROTOCOL_V1_MAX_PROBE_SECONDS * sample_rate)

    def to_samples(seconds: float) -> int:
        return max(1, min(max_samples, round(seconds * sample_rate)))

    initial = to_samples(_INITIAL_PROBE_SECONDS)
    initial_ok = trace.probe(accepts, initial)

    if initial_ok:
        low_fail = None
        high_ok = initial
        candidate = initial
        while True:
            candidate = max(1, candidate // 2)
            if candidate == high_ok:
                low_fail = 0
                break
            ok = trace.probe(accepts, candidate)
            if ok:
                high_ok = candidate
                if candidate == 1:
                    low_fail = 0
                    break
            else:
                low_fail = candidate
                break
    else:
        sweep_point = initial
        while sweep_point > 1:
            sweep_point = max(1, sweep_point // 2)
            if trace.probe(accepts, sweep_point):
                return MinimumInputSearchResult(
                    status=MinimumInputStatus.NON_MONOTONIC,
                    probe_sample_counts_tested=tuple(sorted(trace.tested)),
                    probe_records=trace.records(),
                    notes=(
                        f"Sample count {sweep_point} succeeded despite the larger initial probe "
                        f"at {initial} failing; refusing to report a fabricated minimum."
                    ),
                )

        low_fail = initial
        high_ok = None
        candidate = initial
        while True:
            next_candidate = candidate * 2
            if next_candidate > max_samples:
                return MinimumInputSearchResult(
                    status=MinimumInputStatus.MAX_DURATION_EXHAUSTED,
                    probe_sample_counts_tested=tuple(sorted(trace.tested)),
                    probe_records=trace.records(),
                    notes=(
                        "No accepted sample count was found up to the protocol-v1 maximum probe "
                        f"duration of {PROTOCOL_V1_MAX_PROBE_SECONDS}s."
                    ),
                )
            ok = trace.probe(accepts, next_candidate)
            if ok:
                high_ok = next_candidate
                break
            low_fail = next_candidate
            candidate = next_candidate

    if high_ok is None:
        return MinimumInputSearchResult(
            status=MinimumInputStatus.ALWAYS_UNSUPPORTED,
            probe_sample_counts_tested=tuple(sorted(trace.tested)),
            probe_records=trace.records(),
            notes="No sample count up to the protocol maximum was accepted.",
        )

    assert low_fail is not None

    # Bisection alone can only ever converge to a self-consistent bracket: it trusts each
    # answer and narrows accordingly, so it can never observe a contradiction on its own path.
    # A downward geometric sweep below the established failure point is what actually gives
    # non-monotonic (flaky) support a chance to be caught, by directly testing whether anything
    # smaller than a known failure nonetheless succeeds.
    sweep_point = low_fail
    while sweep_point > 1:
        sweep_point = max(1, sweep_point // 2)
        if trace.probe(accepts, sweep_point):
            return MinimumInputSearchResult(
                status=MinimumInputStatus.NON_MONOTONIC,
                probe_sample_counts_tested=tuple(sorted(trace.tested)),
                probe_records=trace.records(),
                notes=(
                    f"Sample count {sweep_point} succeeded even though the larger-or-equal "
                    f"sample count {low_fail} previously failed; refusing to report a "
                    f"fabricated minimum."
                ),
            )

    lower = low_fail
    upper = high_ok
    while upper - lower > 1:
        mid = (lower + upper) // 2
        ok = trace.probe(accepts, mid)
        if ok:
            upper = mid
        else:
            lower = mid

    if not _is_monotonic(trace):
        return MinimumInputSearchResult(
            status=MinimumInputStatus.NON_MONOTONIC,
            probe_sample_counts_tested=tuple(sorted(trace.tested)),
            probe_records=trace.records(),
            notes=(
                "Support was non-monotonic across tested sample counts; refusing to report a "
                "fabricated minimum."
            ),
        )

    minimum = upper
    boundary_verified = trace.probe(accepts, minimum) is True
    below_verified = False
    if minimum > 1:
        below_verified = trace.probe(accepts, minimum - 1) is False
    elif minimum == 1:
        below_verified = True

    if not _is_monotonic(trace):
        return MinimumInputSearchResult(
            status=MinimumInputStatus.NON_MONOTONIC,
            probe_sample_counts_tested=tuple(sorted(trace.tested)),
            probe_records=trace.records(),
            notes="Non-monotonic support was detected during boundary verification.",
        )

    return MinimumInputSearchResult(
        status=MinimumInputStatus.FOUND,
        minimum_sample_count=minimum,
        minimum_duration_seconds=minimum / sample_rate,
        probe_sample_counts_tested=tuple(sorted(trace.tested)),
        probe_records=trace.records(),
        boundary_verified=boundary_verified,
        below_boundary_verified=below_verified,
    )
