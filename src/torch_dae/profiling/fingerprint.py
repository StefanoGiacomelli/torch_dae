"""Privacy-safe hardware and execution-context fingerprints (Sections 6-7).

These fingerprints describe a *configuration class*, never a physical machine: no hostname,
username, IP, MAC address, serial number, machine UUID, credentials, tokens, or home-directory
path ever participates. `privacy.validate_privacy_safe` is run over every candidate Technical
Card payload as an additional structural safeguard.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from typing import Literal

from torch_dae.contracts import canonical_json_bytes
from torch_dae.profiling.contracts import (
    DeviceBackend,
    ExecutionContextMetadata,
    HardwareMetadata,
    SoftwareMetadata,
    ThreadRegime,
)


def _cpu_architecture() -> str:
    return platform.machine().lower()


def _cpu_model() -> str | None:
    """Return a safe, portable, human-meaningful CPU/SoC model string when available.

    ``platform.processor()`` alone is not good enough for benchmark comparison: on Apple Silicon
    it returns the generic architecture family (e.g. ``"arm"``) rather than the actual chip
    (e.g. ``"Apple M4 Pro"``). This queries OS-provided, non-identifying hardware/SoC model
    strings -- never a hostname, username, serial number, MAC address, or any other
    machine-unique identifier -- and falls back to ``platform.processor()`` when no richer safe
    model can be obtained portably.
    """

    system = platform.system()
    if system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass
        else:
            brand = result.stdout.strip()
            if result.returncode == 0 and brand:
                return brand
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        _, _, value = line.partition(":")
                        model = value.strip()
                        if model:
                            return model
        except OSError:
            pass

    return platform.processor() or None


def gather_hardware_metadata(
    *,
    accelerator_vendor: str | None = None,
    accelerator_model: str | None = None,
    accelerator_memory_bytes: int | None = None,
    backend_capabilities: tuple[str, ...] = (),
) -> HardwareMetadata:
    """Gather privacy-safe local hardware evidence using only configuration-class signals."""

    from torch_dae.profiling.memory_host import psutil_available, total_host_ram_bytes

    physical_cores = None
    logical_cores = os.cpu_count()
    available, _ = psutil_available()
    if available:
        import psutil

        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True) or logical_cores

    return HardwareMetadata(
        cpu_model=_cpu_model(),
        cpu_architecture=_cpu_architecture(),
        physical_cores=physical_cores,
        logical_cores=logical_cores,
        total_ram_bytes=total_host_ram_bytes(),
        accelerator_vendor=accelerator_vendor,
        accelerator_model=accelerator_model,
        accelerator_memory_bytes=accelerator_memory_bytes,
        backend_capabilities=backend_capabilities,
    )


def hardware_fingerprint(hardware: HardwareMetadata) -> str:
    """Return a stable SHA-256 fingerprint describing the hardware *configuration class*.

    RAM and accelerator-memory are bucketed to nearest GiB so trivially different reported
    totals across otherwise-identical machines still collapse to the same configuration class.
    """

    def _bucket(value: int | None) -> int | None:
        return None if value is None else round(value / (1 << 30))

    payload = {
        "cpu_model": hardware.cpu_model,
        "cpu_architecture": hardware.cpu_architecture,
        "physical_cores": hardware.physical_cores,
        "logical_cores": hardware.logical_cores,
        "total_ram_gib": _bucket(hardware.total_ram_bytes),
        "accelerator_vendor": hardware.accelerator_vendor,
        "accelerator_model": hardware.accelerator_model,
        "accelerator_memory_gib": _bucket(hardware.accelerator_memory_bytes),
        "backend_capabilities": sorted(hardware.backend_capabilities),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def execution_context_fingerprint(
    *,
    hardware_fingerprint_value: str,
    software: SoftwareMetadata,
    torch_dae_content_identity: str,
    device_backend: DeviceBackend,
    native_precision: str,
    thread_regime: ThreadRegime | None,
    profiling_protocol_id: Literal["audio-inference-v1"],
    profiling_protocol_version: str,
    profiler_implementation_version: str,
) -> str:
    """Return a stable SHA-256 fingerprint for one reproducible execution context."""

    payload = {
        "hardware_fingerprint": hardware_fingerprint_value,
        "os_name": software.os_name,
        "os_version": software.os_version,
        "python_version": software.python_version,
        "torch_version": software.torch_version,
        "torch_dae_content_identity": torch_dae_content_identity,
        "device_backend": device_backend.value,
        "native_precision": native_precision,
        "thread_regime": thread_regime.value if thread_regime is not None else None,
        "profiling_protocol_id": profiling_protocol_id,
        "profiling_protocol_version": profiling_protocol_version,
        "profiler_implementation_version": profiler_implementation_version,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def build_execution_context_metadata(
    *,
    hardware: HardwareMetadata,
    software: SoftwareMetadata,
    torch_dae_content_identity: str,
    device_backend: DeviceBackend,
    native_precision: str,
    thread_regime: ThreadRegime | None,
    profiling_protocol_id: Literal["audio-inference-v1"],
    profiling_protocol_version: str,
    profiler_implementation_version: str,
) -> ExecutionContextMetadata:
    """Build the full recorded execution-context metadata for one Technical Card."""

    hw_fp = hardware_fingerprint(hardware)
    ctx_fp = execution_context_fingerprint(
        hardware_fingerprint_value=hw_fp,
        software=software,
        torch_dae_content_identity=torch_dae_content_identity,
        device_backend=device_backend,
        native_precision=native_precision,
        thread_regime=thread_regime,
        profiling_protocol_id=profiling_protocol_id,
        profiling_protocol_version=profiling_protocol_version,
        profiler_implementation_version=profiler_implementation_version,
    )
    return ExecutionContextMetadata(
        hardware_fingerprint=hw_fp,
        execution_context_fingerprint=ctx_fp,
        os_name=software.os_name,
        os_version=software.os_version,
        python_version=software.python_version,
        torch_version=software.torch_version,
        torch_dae_content_identity=torch_dae_content_identity,
        device_backend=device_backend,
        native_precision=native_precision,
        thread_regime=thread_regime,
        profiling_protocol_id=profiling_protocol_id,
        profiling_protocol_version=profiling_protocol_version,
        profiler_implementation_version=profiler_implementation_version,
    )
