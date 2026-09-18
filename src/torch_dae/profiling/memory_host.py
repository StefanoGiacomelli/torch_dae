"""Host RAM / process RSS sampling (Section 19).

Runs in the root control-plane process (where `psutil` is an explicit `profiling` extra) and
samples the *child* model-environment subprocess's RSS while it performs the resource pass. The
model-runtime environment itself never needs `psutil` (Section 25 dependency isolation).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from torch_dae.profiling.contracts import HostMemoryEvidence


def psutil_available() -> tuple[bool, str | None]:
    """Return whether `psutil` is importable in this (root) environment, and its version."""

    try:
        import psutil
    except ImportError:
        return False, None
    return True, getattr(psutil, "__version__", None)


def total_host_ram_bytes() -> int | None:
    """Return total host RAM in bytes, or ``None`` when `psutil` is unavailable."""

    available, _ = psutil_available()
    if not available:
        return None
    import psutil

    return int(psutil.virtual_memory().total)


@dataclass
class ChildProcessRssSampler:
    """Periodically samples a child process's RSS in a background thread while it runs."""

    pid: int
    interval_seconds: float = 0.05
    _peak_bytes: int | None = field(default=None, init=False, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> ChildProcessRssSampler:
        available, _ = psutil_available()
        if not available:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _sample_loop(self) -> None:
        import psutil

        try:
            process = psutil.Process(self.pid)
        except psutil.Error:
            return
        while not self._stop.is_set():
            try:
                rss = int(process.memory_info().rss)
            except psutil.Error:
                break
            if self._peak_bytes is None or rss > self._peak_bytes:
                self._peak_bytes = rss
            time.sleep(self.interval_seconds)

    @property
    def peak_bytes(self) -> int | None:
        return self._peak_bytes


def build_host_memory_evidence(
    *,
    rss_before_model_load_bytes: int | None,
    rss_after_model_load_bytes: int | None,
    rss_before_resource_pass_bytes: int | None,
    rss_sampled_peak_bytes: int | None,
    rss_after_resource_pass_bytes: int | None,
) -> HostMemoryEvidence:
    """Assemble host-RAM evidence, labeling the peak explicitly as sampled, not exact."""

    return HostMemoryEvidence(
        total_ram_bytes=total_host_ram_bytes(),
        rss_before_model_load_bytes=rss_before_model_load_bytes,
        rss_after_model_load_bytes=rss_after_model_load_bytes,
        rss_before_resource_pass_bytes=rss_before_resource_pass_bytes,
        rss_sampled_peak_bytes=rss_sampled_peak_bytes,
        rss_sampled_peak_is_sampled=True,
        rss_after_resource_pass_bytes=rss_after_resource_pass_bytes,
        uss_bytes=None,
        pss_bytes=None,
        sampling_method=(
            "periodic parent-process psutil.Process(child_pid).memory_info().rss polling "
            f"(interval={0.05}s); a sampled maximum, not an exact allocator peak"
        ),
    )
