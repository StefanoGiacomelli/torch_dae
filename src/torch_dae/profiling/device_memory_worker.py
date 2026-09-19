"""Native accelerator-memory evidence (Sections 20-21). Worker-process only (needs `torch`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch_dae.profiling.contracts import AcceleratorMemoryEvidence, DeviceBackend

if TYPE_CHECKING:
    import torch  # type: ignore[import-not-found]


def reset_peak_counters(device: torch.device) -> None:
    """Reset native peak-memory counters before a resource-pass measurement window."""

    import torch

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # MPS exposes no reset-peak API as of the PyTorch versions targeted by Profiling v1.


def snapshot_accelerator_memory(device: torch.device) -> AcceleratorMemoryEvidence | None:
    """Return native allocator evidence for CUDA/MPS, or ``None`` on CPU."""

    import torch

    if device.type == "cuda":
        return AcceleratorMemoryEvidence(
            backend=DeviceBackend.CUDA,
            cuda_current_allocated_bytes=int(torch.cuda.memory_allocated(device)),
            cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            cuda_current_reserved_bytes=int(torch.cuda.memory_reserved(device)),
            cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        )
    if device.type == "mps":
        current = None
        driver = None
        if hasattr(torch, "mps"):
            if hasattr(torch.mps, "current_allocated_memory"):
                current = int(torch.mps.current_allocated_memory())
            if hasattr(torch.mps, "driver_allocated_memory"):
                driver = int(torch.mps.driver_allocated_memory())
        return AcceleratorMemoryEvidence(
            backend=DeviceBackend.MPS,
            mps_current_allocated_bytes=current,
            mps_driver_allocated_bytes=driver,
            unified_memory_note=(
                "Apple unified memory: host RSS, MPS tensor allocation, and Metal/driver "
                "allocation are separate measurement surfaces and are never summed."
            ),
        )
    return None
