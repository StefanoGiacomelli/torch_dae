from __future__ import annotations

import pytest

from torch_dae.profiling.contracts import DeviceBackend
from torch_dae.profiling.devices import parse_device_selector, resolve_auto_device_labels


def test_parse_auto() -> None:
    selector = parse_device_selector("auto")
    assert selector.auto
    assert selector.backend is None


def test_parse_cpu() -> None:
    selector = parse_device_selector("cpu")
    assert selector.backend == DeviceBackend.CPU
    assert selector.index is None


def test_parse_mps() -> None:
    selector = parse_device_selector("mps")
    assert selector.backend == DeviceBackend.MPS


def test_parse_cuda_bare() -> None:
    selector = parse_device_selector("cuda")
    assert selector.backend == DeviceBackend.CUDA
    assert selector.index is None


def test_parse_cuda_indexed() -> None:
    selector = parse_device_selector("cuda:2")
    assert selector.backend == DeviceBackend.CUDA
    assert selector.index == 2
    assert selector.label == "cuda:2"


def test_parse_is_case_insensitive() -> None:
    assert parse_device_selector("CPU").backend == DeviceBackend.CPU


def test_parse_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        parse_device_selector("tpu")
    with pytest.raises(ValueError):
        parse_device_selector("cuda:")
    with pytest.raises(ValueError):
        parse_device_selector("cuda:abc")


def test_auto_expands_cpu_only() -> None:
    labels = resolve_auto_device_labels(mps_available=False, cuda_device_count=0)
    assert labels == ("cpu",)


def test_auto_expands_with_mps() -> None:
    labels = resolve_auto_device_labels(mps_available=True, cuda_device_count=0)
    assert labels == ("cpu", "mps")


def test_auto_expands_with_multiple_cuda_devices() -> None:
    labels = resolve_auto_device_labels(mps_available=False, cuda_device_count=2)
    assert labels == ("cpu", "cuda:0", "cuda:1")


def test_auto_expands_with_everything() -> None:
    labels = resolve_auto_device_labels(mps_available=True, cuda_device_count=1)
    assert labels == ("cpu", "mps", "cuda:0")
