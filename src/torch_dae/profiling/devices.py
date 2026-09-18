"""Device selector parsing for `--device` (Section 30 / protocol.md)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from torch_dae.profiling.contracts import DeviceBackend

_CUDA_INDEXED = re.compile(r"^cuda:(\d+)$")


@dataclass(frozen=True)
class DeviceSelector:
    """One parsed `--device` selector."""

    backend: DeviceBackend | None
    index: int | None
    auto: bool = False
    label: str = ""


def parse_device_selector(raw: str) -> DeviceSelector:
    """Parse one `--device` value: `auto`, `cpu`, `mps`, `cuda`, or `cuda:<index>`."""

    value = raw.strip().lower()
    if value == "auto":
        return DeviceSelector(backend=None, index=None, auto=True, label="auto")
    if value == "cpu":
        return DeviceSelector(backend=DeviceBackend.CPU, index=None, label="cpu")
    if value == "mps":
        return DeviceSelector(backend=DeviceBackend.MPS, index=None, label="mps")
    if value == "cuda":
        return DeviceSelector(backend=DeviceBackend.CUDA, index=None, label="cuda")
    match = _CUDA_INDEXED.match(value)
    if match:
        index = int(match.group(1))
        return DeviceSelector(backend=DeviceBackend.CUDA, index=index, label=f"cuda:{index}")
    raise ValueError(
        f"invalid device selector {raw!r}; expected one of auto, cpu, mps, cuda, cuda:<index>"
    )


def resolve_auto_device_labels(*, mps_available: bool, cuda_device_count: int) -> tuple[str, ...]:
    """Return the concrete device labels `auto` expands to given local capability discovery.

    CPU is always included. MPS is included only when genuinely usable. Every discovered CUDA
    index is included individually so a smoke-test failure on one index does not hide others.
    """

    labels = ["cpu"]
    if mps_available:
        labels.append("mps")
    labels.extend(f"cuda:{i}" for i in range(cuda_device_count))
    return tuple(labels)
