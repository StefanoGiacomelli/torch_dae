from __future__ import annotations

import time
from unittest.mock import patch

from torch_dae.profiling.memory_host import (
    ChildProcessRssSampler,
    build_host_memory_evidence,
    psutil_available,
    total_host_ram_bytes,
)


def test_psutil_available_reports_version() -> None:
    available, version = psutil_available()
    assert available is True
    assert version is not None


def test_total_host_ram_bytes_positive() -> None:
    assert total_host_ram_bytes() is not None
    assert total_host_ram_bytes() > 0


def test_total_host_ram_unavailable_without_psutil() -> None:
    with patch("torch_dae.profiling.memory_host.psutil_available", return_value=(False, None)):
        assert total_host_ram_bytes() is None


def test_sampler_tracks_own_process_peak() -> None:
    import os

    with ChildProcessRssSampler(pid=os.getpid(), interval_seconds=0.01) as sampler:
        # Allocate to encourage RSS growth while the sampler polls.
        _blob = bytearray(10_000_000)
        time.sleep(0.05)
        del _blob
    assert sampler.peak_bytes is not None
    assert sampler.peak_bytes > 0


def test_sampler_handles_missing_process_gracefully() -> None:
    # An invalid PID must not crash the sampler; peak stays None.
    with ChildProcessRssSampler(pid=999_999_999, interval_seconds=0.01) as sampler:
        time.sleep(0.05)
    assert sampler.peak_bytes is None


def test_sampler_unavailable_without_psutil() -> None:
    with patch("torch_dae.profiling.memory_host.psutil_available", return_value=(False, None)):
        with ChildProcessRssSampler(pid=1) as sampler:
            time.sleep(0.02)
        assert sampler.peak_bytes is None


def test_build_host_memory_evidence_labels_peak_as_sampled() -> None:
    evidence = build_host_memory_evidence(
        rss_before_model_load_bytes=100,
        rss_after_model_load_bytes=200,
        rss_before_resource_pass_bytes=250,
        rss_sampled_peak_bytes=300,
        rss_after_resource_pass_bytes=280,
    )
    assert evidence.rss_sampled_peak_is_sampled is True
    assert "sampled" in evidence.sampling_method.lower()
    assert evidence.uss_bytes is None
    assert evidence.pss_bytes is None


def test_build_host_memory_evidence_reports_total_ram() -> None:
    evidence = build_host_memory_evidence(
        rss_before_model_load_bytes=None,
        rss_after_model_load_bytes=None,
        rss_before_resource_pass_bytes=None,
        rss_sampled_peak_bytes=None,
        rss_after_resource_pass_bytes=None,
    )
    assert evidence.total_ram_bytes is not None
