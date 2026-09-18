from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_torch():
    module = types.ModuleType("torch")
    module.cuda = MagicMock()
    module.mps = MagicMock()
    original = sys.modules.get("torch")
    sys.modules["torch"] = module
    yield module
    if original is not None:
        sys.modules["torch"] = original
    else:
        del sys.modules["torch"]


def _device(kind: str) -> MagicMock:
    device = MagicMock()
    device.type = kind
    return device


def test_cpu_snapshot_returns_none(fake_torch) -> None:
    from torch_dae.profiling.device_memory_worker import snapshot_accelerator_memory

    assert snapshot_accelerator_memory(_device("cpu")) is None


def test_cuda_snapshot_reports_allocator_evidence(fake_torch) -> None:
    from torch_dae.profiling.device_memory_worker import snapshot_accelerator_memory

    fake_torch.cuda.memory_allocated.return_value = 1_000
    fake_torch.cuda.max_memory_allocated.return_value = 2_000
    fake_torch.cuda.memory_reserved.return_value = 3_000
    fake_torch.cuda.max_memory_reserved.return_value = 4_000

    evidence = snapshot_accelerator_memory(_device("cuda"))
    assert evidence is not None
    assert evidence.backend.value == "cuda"
    assert evidence.cuda_current_allocated_bytes == 1_000
    assert evidence.cuda_peak_allocated_bytes == 2_000
    assert evidence.cuda_current_reserved_bytes == 3_000
    assert evidence.cuda_peak_reserved_bytes == 4_000
    assert evidence.mps_current_allocated_bytes is None


def test_cuda_reset_peak_counters_called(fake_torch) -> None:
    from torch_dae.profiling.device_memory_worker import reset_peak_counters

    device = _device("cuda")
    reset_peak_counters(device)
    fake_torch.cuda.reset_peak_memory_stats.assert_called_once_with(device)


def test_mps_reset_is_a_no_op(fake_torch) -> None:
    from torch_dae.profiling.device_memory_worker import reset_peak_counters

    reset_peak_counters(_device("mps"))
    fake_torch.cuda.reset_peak_memory_stats.assert_not_called()


def test_mps_snapshot_reports_allocation_without_summing(fake_torch) -> None:
    from torch_dae.profiling.device_memory_worker import snapshot_accelerator_memory

    fake_torch.mps.current_allocated_memory.return_value = 5_000
    fake_torch.mps.driver_allocated_memory.return_value = 8_000

    evidence = snapshot_accelerator_memory(_device("mps"))
    assert evidence is not None
    assert evidence.backend.value == "mps"
    assert evidence.mps_current_allocated_bytes == 5_000
    assert evidence.mps_driver_allocated_bytes == 8_000
    assert evidence.cuda_current_allocated_bytes is None
    assert "never summed" in evidence.unified_memory_note.lower()


def test_mps_snapshot_handles_missing_apis_gracefully(fake_torch) -> None:
    from torch_dae.profiling.device_memory_worker import snapshot_accelerator_memory

    # Simulate an older torch.mps without current_allocated_memory/driver_allocated_memory.
    del fake_torch.mps.current_allocated_memory
    del fake_torch.mps.driver_allocated_memory

    evidence = snapshot_accelerator_memory(_device("mps"))
    assert evidence is not None
    assert evidence.mps_current_allocated_bytes is None
    assert evidence.mps_driver_allocated_bytes is None
