from __future__ import annotations

import platform
import subprocess
from unittest.mock import patch

from torch_dae.profiling.contracts import DeviceBackend, SoftwareMetadata, ThreadRegime
from torch_dae.profiling.fingerprint import (
    _cpu_model,
    execution_context_fingerprint,
    gather_hardware_metadata,
    hardware_fingerprint,
)


def _software() -> SoftwareMetadata:
    return SoftwareMetadata(
        os_name="Darwin",
        os_version="25.5.0",
        python_implementation="CPython",
        python_version="3.12.13",
        torch_version="2.13.0",
    )


def test_hardware_fingerprint_deterministic() -> None:
    hw = gather_hardware_metadata()
    assert hardware_fingerprint(hw) == hardware_fingerprint(hw)


def test_hardware_metadata_never_leaks_prohibited_fields() -> None:
    hw = gather_hardware_metadata()
    dumped = hw.model_dump()
    for key in ("hostname", "username", "ip_address", "mac_address", "machine_uuid"):
        assert key not in dumped


def test_execution_context_fingerprint_changes_with_device_backend() -> None:
    hw_fp = "a" * 64
    a = execution_context_fingerprint(
        hardware_fingerprint_value=hw_fp,
        software=_software(),
        torch_dae_content_identity="content-sha256:" + "b" * 64,
        device_backend=DeviceBackend.CPU,
        native_precision="float32",
        thread_regime=ThreadRegime.SINGLE_THREAD,
        profiling_protocol_id="audio-inference-v1",
        profiling_protocol_version="1.0.0",
        profiler_implementation_version="1.0.0",
    )
    b = execution_context_fingerprint(
        hardware_fingerprint_value=hw_fp,
        software=_software(),
        torch_dae_content_identity="content-sha256:" + "b" * 64,
        device_backend=DeviceBackend.MPS,
        native_precision="float32",
        thread_regime=None,
        profiling_protocol_id="audio-inference-v1",
        profiling_protocol_version="1.0.0",
        profiler_implementation_version="1.0.0",
    )
    assert a != b


def test_cpu_model_on_macos_uses_sysctl_brand_string_not_generic_arch() -> None:
    fake_result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="Apple M4 Pro\n", stderr=""
    )
    with (
        patch("torch_dae.profiling.fingerprint.platform.system", return_value="Darwin"),
        patch("torch_dae.profiling.fingerprint.subprocess.run", return_value=fake_result) as run,
    ):
        model = _cpu_model()
    assert model == "Apple M4 Pro"
    assert model != "arm"
    run.assert_called_once()


def test_cpu_model_on_linux_uses_proc_cpuinfo_model_name() -> None:
    cpuinfo = "processor\t: 0\nmodel name\t: AMD Ryzen 9 7950X\nflags\t: fpu vme\n"
    with (
        patch("torch_dae.profiling.fingerprint.platform.system", return_value="Linux"),
        patch("builtins.open", side_effect=lambda *a, **k: __import__("io").StringIO(cpuinfo)),
    ):
        model = _cpu_model()
    assert model == "AMD Ryzen 9 7950X"


def test_cpu_model_falls_back_to_platform_processor_when_nothing_richer_is_available() -> None:
    with (
        patch("torch_dae.profiling.fingerprint.platform.system", return_value="Windows"),
        patch("torch_dae.profiling.fingerprint.platform.processor", return_value="AMD64 Family"),
    ):
        assert _cpu_model() == "AMD64 Family"


def test_cpu_model_falls_back_when_sysctl_fails() -> None:
    with (
        patch("torch_dae.profiling.fingerprint.platform.system", return_value="Darwin"),
        patch("torch_dae.profiling.fingerprint.subprocess.run", side_effect=OSError("no sysctl")),
        patch("torch_dae.profiling.fingerprint.platform.processor", return_value="arm"),
    ):
        assert _cpu_model() == "arm"


def test_cpu_model_never_leaks_prohibited_identity() -> None:
    # Whatever the real local machine reports, it must never be a hostname/user/serial-shaped
    # value; this is a smoke check against this test machine's actual reported model.
    model = _cpu_model()
    if model is not None:
        assert platform.node() not in model
        import getpass

        assert getpass.getuser() not in model


def test_execution_context_fingerprint_changes_with_thread_regime() -> None:
    kwargs = dict(
        hardware_fingerprint_value="a" * 64,
        software=_software(),
        torch_dae_content_identity="content-sha256:" + "b" * 64,
        device_backend=DeviceBackend.CPU,
        native_precision="float32",
        profiling_protocol_id="audio-inference-v1",
        profiling_protocol_version="1.0.0",
        profiler_implementation_version="1.0.0",
    )
    single = execution_context_fingerprint(thread_regime=ThreadRegime.SINGLE_THREAD, **kwargs)
    native = execution_context_fingerprint(thread_regime=ThreadRegime.NATIVE_DEFAULT, **kwargs)
    assert single != native
