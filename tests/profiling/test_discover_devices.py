from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from torch_dae.profiling_executor import discover_devices


def _fake_process(*, returncode: int = 0, stderr: str = "") -> MagicMock:
    process = MagicMock()
    process.communicate.return_value = ("", stderr)
    process.returncode = returncode
    return process


def _install_worker_results(workspace: Path, results: dict[str, dict]) -> None:
    """Pre-write result.json for each device label so the fake subprocess 'produces' it."""

    for label, result in results.items():
        session_dir = workspace / "smoke" / label.replace(":", "_")
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "result.json").write_text(json.dumps(result))


def test_cpu_only_all_succeed(tmp_path: Path) -> None:
    _install_worker_results(tmp_path, {"cpu": {"smoke_passed": True, "smoke_error": None}})
    with patch(
        "torch_dae.profiling_executor._run_worker_subprocess",
        return_value=(_fake_process(), 1234),
    ):
        attempted, successful, failed = discover_devices(
            python_executable=Path("/bin/true"),
            wrapper_entry_point="pkg:Model",
            checkpoint_path=Path("/tmp/x"),
            sample_rate=16_000,
            requested_labels=("cpu",),
            workspace=tmp_path,
        )
    assert attempted == ("cpu",)
    assert successful == ("cpu",)
    assert failed == ()


def test_unavailable_accelerator_reported_as_failed_not_raised(tmp_path: Path) -> None:
    _install_worker_results(
        tmp_path,
        {
            "cpu": {"smoke_passed": True, "smoke_error": None},
            "mps": {"smoke_passed": False, "smoke_error": "MPS backend out of memory"},
        },
    )
    with patch(
        "torch_dae.profiling_executor._run_worker_subprocess",
        return_value=(_fake_process(), 1234),
    ):
        _attempted, successful, failed = discover_devices(
            python_executable=Path("/bin/true"),
            wrapper_entry_point="pkg:Model",
            checkpoint_path=Path("/tmp/x"),
            sample_rate=16_000,
            requested_labels=("cpu", "mps"),
            workspace=tmp_path,
        )
    assert successful == ("cpu",)
    assert len(failed) == 1
    assert failed[0].device_label == "mps"
    assert "out of memory" in failed[0].error


def test_multiple_cuda_devices_independent_outcomes(tmp_path: Path) -> None:
    _install_worker_results(
        tmp_path,
        {
            "cpu": {"smoke_passed": True, "smoke_error": None},
            "cuda:0": {"smoke_passed": True, "smoke_error": None},
            "cuda:1": {"smoke_passed": False, "smoke_error": "device-side assert"},
        },
    )
    with patch(
        "torch_dae.profiling_executor._run_worker_subprocess",
        return_value=(_fake_process(), 1234),
    ):
        _attempted, successful, failed = discover_devices(
            python_executable=Path("/bin/true"),
            wrapper_entry_point="pkg:Model",
            checkpoint_path=Path("/tmp/x"),
            sample_rate=16_000,
            requested_labels=("cpu", "cuda:0", "cuda:1"),
            workspace=tmp_path,
        )
    assert set(successful) == {"cpu", "cuda:0"}
    assert {d.device_label for d in failed} == {"cuda:1"}


def test_explicit_device_failure_does_not_fall_back_silently(tmp_path: Path) -> None:
    # An explicitly requested device that fails must be reported, never silently swapped for cpu.
    _install_worker_results(tmp_path, {"mps": {"smoke_passed": False, "smoke_error": "no MPS"}})
    with patch(
        "torch_dae.profiling_executor._run_worker_subprocess",
        return_value=(_fake_process(), 1234),
    ):
        attempted, successful, failed = discover_devices(
            python_executable=Path("/bin/true"),
            wrapper_entry_point="pkg:Model",
            checkpoint_path=Path("/tmp/x"),
            sample_rate=16_000,
            requested_labels=("mps",),
            workspace=tmp_path,
        )
    assert attempted == ("mps",)
    assert successful == ()
    assert "cpu" not in successful
    assert failed[0].device_label == "mps"


def test_worker_produces_no_result_is_reported_as_failure(tmp_path: Path) -> None:
    # No result.json written at all (crash before writing) -- must not raise, must be diagnosed.
    with patch(
        "torch_dae.profiling_executor._run_worker_subprocess",
        return_value=(_fake_process(stderr="segfault"), 1234),
    ):
        _attempted, successful, failed = discover_devices(
            python_executable=Path("/bin/true"),
            wrapper_entry_point="pkg:Model",
            checkpoint_path=Path("/tmp/x"),
            sample_rate=16_000,
            requested_labels=("cuda:0",),
            workspace=tmp_path,
        )
    assert successful == ()
    assert failed[0].device_label == "cuda:0"
    assert "no result" in failed[0].error
