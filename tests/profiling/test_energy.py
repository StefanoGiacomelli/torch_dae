from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from torch_dae.profiling.contracts import DeviceBackend, EnergyMeasurementKind
from torch_dae.profiling.energy import (
    CodeCarbonEnergyBackend,
    OffEnergyBackend,
    select_energy_backend,
)


def test_off_backend_reports_unavailable() -> None:
    backend = OffEnergyBackend()
    result, evidence = backend.measure_block(lambda: "done", device_backend=DeviceBackend.CPU)
    assert result == "done"
    assert evidence.measurement_kind == EnergyMeasurementKind.UNAVAILABLE
    assert evidence.coverage_complete is False
    assert evidence.unaccounted_components == ()


def test_select_energy_backend_off() -> None:
    assert isinstance(select_energy_backend("off"), OffEnergyBackend)


def test_select_energy_backend_auto() -> None:
    assert isinstance(select_energy_backend("auto"), CodeCarbonEnergyBackend)


def test_select_energy_backend_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        select_energy_backend("bogus")


def _fake_emissions_data(**overrides: object) -> MagicMock:
    data = MagicMock()
    data.cpu_energy = overrides.get("cpu_energy", 1e-5)
    data.gpu_energy = overrides.get("gpu_energy", 0.0)
    data.ram_energy = overrides.get("ram_energy", 2e-6)
    data.energy_consumed = overrides.get("energy_consumed", 1.2e-5)
    data.duration = overrides.get("duration", 3.0)
    return data


def test_codecarbon_unavailable_when_not_installed() -> None:
    backend = CodeCarbonEnergyBackend(allow_privileged_energy=True)
    with patch("torch_dae.profiling.energy.codecarbon_available", return_value=(False, None)):
        result, evidence = backend.measure_block(lambda: "ok", device_backend=DeviceBackend.CPU)
    assert result == "ok"
    assert evidence.measurement_kind == EnergyMeasurementKind.UNAVAILABLE
    assert evidence.coverage_complete is False


def test_codecarbon_apple_silicon_requires_explicit_privilege_consent() -> None:
    backend = CodeCarbonEnergyBackend()
    fake_module = MagicMock()

    with (
        patch(
            "torch_dae.profiling.energy.codecarbon_available",
            return_value=(True, "2.8.4"),
        ),
        patch("torch_dae.profiling.energy.platform.system", return_value="Darwin"),
        patch("torch_dae.profiling.energy.platform.machine", return_value="arm64"),
        patch.dict("sys.modules", {"codecarbon": fake_module}),
    ):
        result, evidence = backend.measure_block(
            lambda: "ok",
            device_backend=DeviceBackend.MPS,
        )

    assert result == "ok"
    assert evidence.measurement_kind == EnergyMeasurementKind.UNAVAILABLE
    assert evidence.privilege_used is False
    assert any("not explicitly authorized" in note for note in evidence.limitations)
    fake_module.OfflineEmissionsTracker.assert_not_called()


def _patched_codecarbon(tracker: MagicMock):
    fake_module = MagicMock()
    fake_module.OfflineEmissionsTracker.return_value = tracker
    return (
        patch("torch_dae.profiling.energy.codecarbon_available", return_value=(True, "2.8.4")),
        patch.dict("sys.modules", {"codecarbon": fake_module}),
    )


def test_codecarbon_software_estimated_path_cpu_only_is_complete() -> None:
    """A CPU-only run (no accelerator in play) whose CPU/RAM are both estimated has complete
    coverage -- there is no unmeasured active component to flag."""

    backend = CodeCarbonEnergyBackend(allow_privileged_energy=True)
    tracker = MagicMock()
    tracker.final_emissions_data = _fake_emissions_data()
    tracker._conf = {"hardware": ["CPU(Apple M4 Pro > 85W [generic])", "RAM()"]}
    patch_a, patch_b = _patched_codecarbon(tracker)
    with patch_a, patch_b:
        result, evidence = backend.measure_block(lambda: "ok", device_backend=DeviceBackend.CPU)

    assert result == "ok"
    assert evidence.measurement_kind == EnergyMeasurementKind.SOFTWARE_ESTIMATED
    assert evidence.codecarbon_version == "2.8.4"
    assert evidence.total_energy_kwh is not None
    assert evidence.coverage_complete is True
    assert evidence.unaccounted_components == ()
    tracker.start.assert_called_once()
    tracker.stop.assert_called_once()


def test_codecarbon_hardware_measured_path_cpu_only_is_complete() -> None:
    backend = CodeCarbonEnergyBackend(allow_privileged_energy=True)
    tracker = MagicMock()
    tracker.final_emissions_data = _fake_emissions_data()
    tracker._conf = {"hardware": ["CPU(Intel RAPL)", "RAM()"]}
    patch_a, patch_b = _patched_codecarbon(tracker)
    with patch_a, patch_b:
        _, evidence = backend.measure_block(lambda: "ok", device_backend=DeviceBackend.CPU)

    assert evidence.measurement_kind == EnergyMeasurementKind.HARDWARE_MEASURED
    assert evidence.privilege_used is False
    assert evidence.coverage_complete is True
    assert evidence.unaccounted_components == ()


def test_codecarbon_mps_run_has_incomplete_coverage() -> None:
    """CodeCarbon cannot measure Apple MPS accelerator energy: an MPS run must be marked
    incomplete and must not present its CPU+RAM total as if it were the full run's energy."""

    backend = CodeCarbonEnergyBackend(allow_privileged_energy=True)
    tracker = MagicMock()
    tracker.final_emissions_data = _fake_emissions_data(gpu_energy=0.0)
    tracker._conf = {"hardware": ["CPU(Apple M4 Pro > 85W [generic])", "RAM()"]}
    patch_a, patch_b = _patched_codecarbon(tracker)
    with patch_a, patch_b:
        _, evidence = backend.measure_block(lambda: "ok", device_backend=DeviceBackend.MPS)

    assert evidence.accelerator_energy_kwh is None
    assert evidence.total_energy_kwh is not None  # CPU+RAM total is still reported...
    assert evidence.coverage_complete is False  # ...but explicitly marked incomplete
    assert "accelerator" in evidence.unaccounted_components
    assert any("accelerator energy" in note for note in evidence.limitations)


def test_codecarbon_cuda_run_with_measured_gpu_energy_is_complete() -> None:
    backend = CodeCarbonEnergyBackend(allow_privileged_energy=True)
    tracker = MagicMock()
    tracker.final_emissions_data = _fake_emissions_data(gpu_energy=3e-6)
    tracker._conf = {"hardware": ["CPU(Intel RAPL)", "GPU(NVIDIA)", "RAM()"]}
    patch_a, patch_b = _patched_codecarbon(tracker)
    with patch_a, patch_b:
        _, evidence = backend.measure_block(lambda: "ok", device_backend=DeviceBackend.CUDA)

    assert evidence.accelerator_energy_kwh == pytest.approx(3e-6)
    assert evidence.coverage_complete is True
    assert evidence.unaccounted_components == ()


def test_codecarbon_cuda_run_without_measured_gpu_energy_is_incomplete() -> None:
    """A mocked CUDA run where CodeCarbon reports no GPU energy (e.g. unsupported card) must
    also be flagged incomplete, exactly like the MPS case."""

    backend = CodeCarbonEnergyBackend(allow_privileged_energy=True)
    tracker = MagicMock()
    tracker.final_emissions_data = _fake_emissions_data(gpu_energy=0.0)
    tracker._conf = {"hardware": ["CPU(Intel RAPL)", "RAM()"]}
    patch_a, patch_b = _patched_codecarbon(tracker)
    with patch_a, patch_b:
        _, evidence = backend.measure_block(lambda: "ok", device_backend=DeviceBackend.CUDA)

    assert evidence.accelerator_energy_kwh is None
    assert evidence.coverage_complete is False
    assert "accelerator" in evidence.unaccounted_components


def test_codecarbon_powermetrics_marks_privilege_used() -> None:
    backend = CodeCarbonEnergyBackend(allow_privileged_energy=True)
    tracker = MagicMock()
    tracker.final_emissions_data = _fake_emissions_data(gpu_energy=3e-6)
    tracker._conf = {"hardware": ["Apple PowerMetrics"]}
    tracker._hardware = [type("AppleSiliconChip", (), {})()]

    patch_a, patch_b = _patched_codecarbon(tracker)
    with patch_a, patch_b:
        _, evidence = backend.measure_block(
            lambda: "ok",
            device_backend=DeviceBackend.MPS,
        )

    assert evidence.measurement_kind == EnergyMeasurementKind.HARDWARE_MEASURED
    assert evidence.privilege_used is True
    assert evidence.accelerator_energy_kwh == pytest.approx(3e-6)
    assert any("PowerMetrics" in note for note in evidence.limitations)


def test_codecarbon_powermetrics_zero_gpu_energy_remains_finite_zero() -> None:
    backend = CodeCarbonEnergyBackend(allow_privileged_energy=True)
    tracker = MagicMock()
    tracker.final_emissions_data = _fake_emissions_data(gpu_energy=0.0)
    tracker._conf = {"hardware": ["Apple PowerMetrics"]}
    tracker._hardware = [type("AppleSiliconChip", (), {})()]

    patch_a, patch_b = _patched_codecarbon(tracker)
    with patch_a, patch_b:
        _, evidence = backend.measure_block(
            lambda: "ok",
            device_backend=DeviceBackend.MPS,
        )

    assert evidence.accelerator_energy_kwh == 0.0
    assert evidence.coverage_complete is True
    assert evidence.unaccounted_components == ()


def test_codecarbon_uses_temporary_output_dir() -> None:
    backend = CodeCarbonEnergyBackend(allow_privileged_energy=True)
    tracker = MagicMock()
    tracker.final_emissions_data = _fake_emissions_data()
    tracker._conf = {"hardware": ["CPU(Intel RAPL)", "RAM()"]}

    fake_module = MagicMock()
    fake_module.OfflineEmissionsTracker.return_value = tracker

    with (
        patch(
            "torch_dae.profiling.energy.codecarbon_available",
            return_value=(True, "2.8.4"),
        ),
        patch.dict("sys.modules", {"codecarbon": fake_module}),
    ):
        backend.measure_block(
            lambda: "ok",
            device_backend=DeviceBackend.CPU,
        )

    _, kwargs = fake_module.OfflineEmissionsTracker.call_args
    assert kwargs["output_dir"]
    assert "torch-dae-codecarbon-" in kwargs["output_dir"]


def test_codecarbon_non_finite_cpu_energy_is_unaccounted() -> None:
    backend = CodeCarbonEnergyBackend(allow_privileged_energy=True)
    tracker = MagicMock()
    tracker.final_emissions_data = _fake_emissions_data(
        cpu_energy=float("nan"),
        gpu_energy=3e-6,
        energy_consumed=float("nan"),
    )
    tracker._conf = {"hardware": ["Apple PowerMetrics"]}
    tracker._hardware = [type("AppleSiliconChip", (), {})()]

    patch_a, patch_b = _patched_codecarbon(tracker)
    with patch_a, patch_b:
        _, evidence = backend.measure_block(
            lambda: "ok",
            device_backend=DeviceBackend.MPS,
        )

    assert evidence.measurement_kind == EnergyMeasurementKind.HARDWARE_MEASURED
    assert evidence.cpu_energy_kwh is None
    assert evidence.total_energy_kwh is None
    assert evidence.coverage_complete is False
    assert "cpu" in evidence.unaccounted_components
    assert any("finite CPU energy" in note for note in evidence.limitations)


def test_codecarbon_zero_energy_remains_finite_zero() -> None:
    backend = CodeCarbonEnergyBackend(allow_privileged_energy=True)
    tracker = MagicMock()
    tracker.final_emissions_data = _fake_emissions_data(
        cpu_energy=0.0,
        gpu_energy=0.0,
        ram_energy=0.0,
        energy_consumed=0.0,
        duration=1.0,
    )
    tracker._conf = {"hardware": ["CPU(Intel RAPL)", "RAM()"]}

    patch_a, patch_b = _patched_codecarbon(tracker)
    with patch_a, patch_b:
        _, evidence = backend.measure_block(
            lambda: "ok",
            device_backend=DeviceBackend.CPU,
        )

    assert evidence.cpu_energy_kwh == 0.0
    assert evidence.ram_energy_kwh == 0.0
    assert evidence.total_energy_kwh == 0.0
    assert evidence.coverage_complete is True


def test_codecarbon_start_failure_reported_as_failed() -> None:
    backend = CodeCarbonEnergyBackend(allow_privileged_energy=True)
    tracker = MagicMock()
    tracker.start.side_effect = RuntimeError("boom")
    patch_a, patch_b = _patched_codecarbon(tracker)
    with patch_a, patch_b:
        result, evidence = backend.measure_block(lambda: "ok", device_backend=DeviceBackend.MPS)

    assert result == "ok"
    assert evidence.measurement_kind == EnergyMeasurementKind.FAILED
    assert evidence.failure_reason is not None
    assert evidence.coverage_complete is False


def test_codecarbon_never_sets_geolocation_fields() -> None:
    # OfflineEmissionsTracker must be constructed with an explicit country code and never call
    # the network geo-lookup path; asserting the call kwargs is the unit-level proxy for that.
    backend = CodeCarbonEnergyBackend(allow_privileged_energy=True)
    tracker = MagicMock()
    tracker.final_emissions_data = _fake_emissions_data()
    tracker._conf = {"hardware": []}
    fake_module = MagicMock()
    fake_module.OfflineEmissionsTracker.return_value = tracker

    with (
        patch("torch_dae.profiling.energy.codecarbon_available", return_value=(True, "2.8.4")),
        patch.dict("sys.modules", {"codecarbon": fake_module}),
    ):
        backend.measure_block(lambda: "ok", device_backend=DeviceBackend.CPU)

    _, kwargs = fake_module.OfflineEmissionsTracker.call_args
    assert kwargs["country_iso_code"]
    assert "latitude" not in kwargs
    assert "longitude" not in kwargs
