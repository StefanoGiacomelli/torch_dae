"""Energy backend abstraction and CodeCarbon integration (Sections 22-24).

`EnergyBackend` is a real abstraction so additional backends can be added later without changing
Technical Card semantics. CodeCarbon and its `psutil` companion are profiling-tooling
dependencies installed only in the root control-plane environment (`profiling` extra); they are
never required inside an accepted model-runtime environment (Section 25).

Geolocation is explicitly avoided: this module always uses CodeCarbon's `OfflineEmissionsTracker`
with a fixed placeholder ISO country code and never reports CO2 emissions, so no IP-based
geolocation network call is made and no location is recorded (project_spec.md Section 3 and the
implementation prompt's prohibition on silent geolocation).
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from torch_dae.profiling.contracts import DeviceBackend, EnergyEvidence, EnergyMeasurementKind

_ACCELERATOR_BACKENDS = (DeviceBackend.MPS, DeviceBackend.CUDA)

T = TypeVar("T")

# Never geolocated: a fixed placeholder used only to keep CodeCarbon's offline tracker from
# requiring network access. Technical Cards never report CO2 emissions or location.
_PLACEHOLDER_COUNTRY_ISO_CODE = "USA"
_DEFAULT_MEASUREMENT_INTERVAL_SECONDS = 1.0


class EnergyBackend(abc.ABC, Generic[T]):
    """Abstract energy-measurement backend for the profiling resource pass."""

    @abc.abstractmethod
    def measure_block(
        self, block: Callable[[], T], *, device_backend: DeviceBackend
    ) -> tuple[T, EnergyEvidence]:
        """Run ``block`` once, returning its result alongside energy evidence for its duration.

        ``device_backend`` identifies the accelerator (if any) actually under profile, so the
        returned evidence can state explicitly whether accelerator energy is accounted for.
        """


@dataclass(frozen=True)
class OffEnergyBackend(EnergyBackend[T]):
    """``--energy off``: energy instrumentation is disabled entirely."""

    def measure_block(
        self, block: Callable[[], T], *, device_backend: DeviceBackend
    ) -> tuple[T, EnergyEvidence]:
        result = block()
        return result, EnergyEvidence(measurement_kind=EnergyMeasurementKind.UNAVAILABLE)


def codecarbon_available() -> tuple[bool, str | None]:
    """Return whether CodeCarbon is importable in this (root) environment, and its version."""

    try:
        import codecarbon
    except ImportError:
        return False, None
    return True, getattr(codecarbon, "__version__", None)


class CodeCarbonEnergyBackend(EnergyBackend[T]):
    """``--energy auto``: CodeCarbon-backed measurement/estimation without geolocation.

    Classifies evidence as ``hardware_measured`` when CodeCarbon's selected hardware backend is
    not a generic constant-TDP fallback, otherwise ``software_estimated``. Never invokes
    privileged measurement paths (e.g. Apple ``powermetrics``) on its own; privileged elevation is
    handled only by an explicit, separately-gated interactive step (Section 24).
    """

    def __init__(
        self, *, measurement_interval_seconds: float = _DEFAULT_MEASUREMENT_INTERVAL_SECONDS
    ) -> None:
        self._measurement_interval_seconds = measurement_interval_seconds

    def measure_block(
        self, block: Callable[[], T], *, device_backend: DeviceBackend
    ) -> tuple[T, EnergyEvidence]:
        available, version = codecarbon_available()
        if not available:
            result = block()
            return result, EnergyEvidence(
                measurement_kind=EnergyMeasurementKind.UNAVAILABLE,
                limitations=("CodeCarbon is not installed in the root profiling environment.",),
            )

        from codecarbon import OfflineEmissionsTracker

        tracker = OfflineEmissionsTracker(
            country_iso_code=_PLACEHOLDER_COUNTRY_ISO_CODE,
            measure_power_secs=self._measurement_interval_seconds,
            save_to_file=False,
            save_to_api=False,
            save_to_logger=False,
            log_level="error",
            allow_multiple_runs=True,
        )
        try:
            tracker.start()
        except Exception as exc:  # pragma: no cover - defensive; codecarbon internals vary
            result = block()
            return result, EnergyEvidence(
                measurement_kind=EnergyMeasurementKind.FAILED,
                codecarbon_version=version,
                failure_reason=f"tracker.start() failed: {type(exc).__name__}: {exc}",
            )

        result = block()

        try:
            tracker.stop()
        except Exception as exc:  # pragma: no cover - defensive; codecarbon internals vary
            return result, EnergyEvidence(
                measurement_kind=EnergyMeasurementKind.FAILED,
                codecarbon_version=version,
                failure_reason=f"tracker.stop() failed: {type(exc).__name__}: {exc}",
            )

        data = tracker.final_emissions_data
        hardware_reprs = tuple(str(item) for item in tracker._conf.get("hardware", ()))
        hardware_measured = bool(hardware_reprs) and not any(
            "generic" in repr_ or repr_.startswith("CPU(Constant") for repr_ in hardware_reprs
        )
        kind = (
            EnergyMeasurementKind.HARDWARE_MEASURED
            if hardware_measured
            else EnergyMeasurementKind.SOFTWARE_ESTIMATED
        )
        cpu_energy = float(data.cpu_energy) if data.cpu_energy else None
        gpu_energy = float(data.gpu_energy) if data.gpu_energy else None
        ram_energy = float(data.ram_energy) if data.ram_energy else None
        total_energy = float(data.energy_consumed) if data.energy_consumed else None
        duration = float(data.duration) if data.duration else None
        average_power = (
            (total_energy * 1000.0 / (duration / 3600.0)) if total_energy and duration else None
        )

        accelerator_active = device_backend in _ACCELERATOR_BACKENDS
        unaccounted: list[str] = []
        if accelerator_active and gpu_energy is None:
            unaccounted.append("accelerator")
        if cpu_energy is None:
            unaccounted.append("cpu")
        if ram_energy is None:
            unaccounted.append("ram")
        coverage_complete = not unaccounted

        limitations = list(
            (
                "No privileged hardware counters were used; CPU energy is a generic "
                "constant-TDP estimate.",
            )
            if kind == EnergyMeasurementKind.SOFTWARE_ESTIMATED
            else ()
        )
        if accelerator_active and gpu_energy is None:
            limitations.append(
                f"{device_backend.value} accelerator energy could not be measured by CodeCarbon; "
                "total_energy_kwh (when present) covers only the components listed as accounted "
                "for and must not be read as complete run energy."
            )

        evidence = EnergyEvidence(
            measurement_kind=kind,
            codecarbon_version=version,
            cpu_energy_kwh=cpu_energy,
            accelerator_energy_kwh=gpu_energy,
            ram_energy_kwh=ram_energy,
            total_energy_kwh=total_energy,
            average_power_watts=average_power,
            measurement_duration_seconds=duration,
            measurement_interval_seconds=self._measurement_interval_seconds,
            measurement_scope=(
                "wall-clock duration of the isolated resource-pass subprocess, including model "
                "construction and checkpoint loading overhead; not scoped to individual inferences"
            ),
            privilege_used=False,
            coverage_complete=coverage_complete,
            unaccounted_components=tuple(unaccounted),
            limitations=tuple(limitations),
        )
        return result, evidence


def select_energy_backend(mode: str) -> EnergyBackend[object]:
    """Return the backend for `--energy auto|off`."""

    if mode == "off":
        return OffEnergyBackend()
    if mode == "auto":
        return CodeCarbonEnergyBackend()
    raise ValueError(f"unknown energy mode {mode!r}; expected 'auto' or 'off'")
