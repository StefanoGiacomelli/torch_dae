"""Backend-aware synchronized timing execution (Section 14).

Imported only inside a model-environment worker process. Uses a high-resolution monotonic clock
(`time.perf_counter_ns`) and correct device synchronization so accelerator kernels queued
asynchronously are not mistimed.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch  # type: ignore[import-not-found]


def synchronize(device: torch.device) -> None:
    """Block until all queued work on ``device`` has completed."""

    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    # CPU execution is already synchronous.


def timed_call(fn: Callable[[], object], device: torch.device) -> int:
    """Run ``fn`` once and return elapsed integer nanoseconds, correctly synchronized.

    The canonical timing scope is: prepared waveform tensor -> public wrapper -> requested public
    output. ``fn`` must contain exactly that call; input generation, allocation, and file I/O must
    happen before this function is invoked.
    """

    synchronize(device)
    start = time.perf_counter_ns()
    fn()
    synchronize(device)
    end = time.perf_counter_ns()
    return end - start


def run_warmup_and_measured(
    fn: Callable[[], object],
    device: torch.device,
    *,
    warmup_count: int,
    measured_count: int,
) -> list[int]:
    """Run ``warmup_count`` untimed calls, then return ``measured_count`` raw nanosecond timings."""

    for _ in range(warmup_count):
        fn()
    synchronize(device)
    return [timed_call(fn, device) for _ in range(measured_count)]
