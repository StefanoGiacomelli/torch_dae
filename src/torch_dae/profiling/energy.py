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
import math
import platform
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, SupportsFloat, SupportsIndex, TypeVar

from torch_dae.profiling.contracts import DeviceBackend, EnergyEvidence, EnergyMeasurementKind

_ACCELERATOR_BACKENDS = (DeviceBackend.MPS, DeviceBackend.CUDA)

T = TypeVar("T")

# Never geolocated: a fixed placeholder used only to keep CodeCarbon's offline tracker from
# requiring network access. Technical Cards never report CO2 emissions or location.
_PLACEHOLDER_COUNTRY_ISO_CODE = "USA"
_DEFAULT_MEASUREMENT_INTERVAL_SECONDS = 1.0


def _finite_float_or_none(
    value: str | SupportsFloat | SupportsIndex | None,
) -> float | None:
    """Normalize optional upstream numeric evidence to a finite float.

    CodeCarbon may expose NaN when a hardware counter has no usable samples.
    Non-finite values must be treated as unavailable evidence rather than
    allowed to survive until JSON serialization.
    """

    if value is None:
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    return numeric if math.isfinite(numeric) else None


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
    not a generic constant-TDP fallback, otherwise ``software_estimated``. On Apple Silicon,
    CodeCarbon 2.x may use the privileged ``powermetrics`` backend; torch-dae permits that path
    only when the caller has explicitly authorized privileged energy measurement (Section 24).
    """

    def __init__(
        self,
        *,
        measurement_interval_seconds: float = _DEFAULT_MEASUREMENT_INTERVAL_SECONDS,
        allow_privileged_energy: bool = False,
    ) -> None:
        self._measurement_interval_seconds = measurement_interval_seconds
        self._allow_privileged_energy = allow_privileged_energy

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

        apple_silicon = platform.system() == "Darwin" and platform.machine().lower() in {
            "arm64",
            "aarch64",
        }
        if apple_silicon and not self._allow_privileged_energy:
            result = block()
            return result, EnergyEvidence(
                measurement_kind=EnergyMeasurementKind.UNAVAILABLE,
                codecarbon_version=version,
                limitations=(
                    "Apple Silicon hardware energy measurement through CodeCarbon may invoke "
                    "privileged powermetrics; privileged energy measurement was not explicitly "
                    "authorized for this run.",
                ),
            )

        from codecarbon import OfflineEmissionsTracker

        with tempfile.TemporaryDirectory(prefix="torch-dae-codecarbon-") as tracker_output_dir:
            tracker = OfflineEmissionsTracker(
                country_iso_code=_PLACEHOLDER_COUNTRY_ISO_CODE,
                measure_power_secs=self._measurement_interval_seconds,
                save_to_file=False,
                save_to_api=False,
                save_to_logger=False,
                log_level="error",
                allow_multiple_runs=True,
                output_dir=tracker_output_dir,
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
            hardware_objects_raw = getattr(tracker, "_hardware", ())
            hardware_objects = (
                tuple(hardware_objects_raw)
                if isinstance(hardware_objects_raw, (list, tuple))
                else ()
            )

        privilege_used = any(
            type(item).__name__ == "AppleSiliconChip" or "powermetrics" in str(item).lower()
            for item in hardware_objects
        ) or any("powermetrics" in repr_.lower() for repr_ in hardware_reprs)

        hardware_measured = privilege_used or (
            bool(hardware_reprs)
            and not any(
                "generic" in repr_ or repr_.startswith("CPU(Constant") for repr_ in hardware_reprs
            )
        )
        kind = (
            EnergyMeasurementKind.HARDWARE_MEASURED
            if hardware_measured
            else EnergyMeasurementKind.SOFTWARE_ESTIMATED
        )
        accelerator_backend_observed = privilege_used or any(
            token in repr_.lower()
            for repr_ in hardware_reprs
            for token in ("gpu", "cuda", "nvidia", "rocm")
        )
        cpu_energy = _finite_float_or_none(data.cpu_energy)
        gpu_energy = _finite_float_or_none(data.gpu_energy)
        ram_energy = _finite_float_or_none(data.ram_energy)
        total_energy = _finite_float_or_none(data.energy_consumed)
        duration = _finite_float_or_none(data.duration)
        average_power = (
            total_energy * 1000.0 / (duration / 3600.0)
            if total_energy is not None and duration is not None and duration > 0.0
            else None
        )

        accelerator_active = device_backend in _ACCELERATOR_BACKENDS
        if accelerator_active and gpu_energy == 0.0 and not accelerator_backend_observed:
            gpu_energy = None

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
        if privilege_used:
            limitations.append(
                "CodeCarbon selected Apple PowerMetrics; local sudo privilege was used for "
                "hardware-counter access. torch-dae does not read, store, or persist the sudo "
                "credential."
            )
        if accelerator_active and gpu_energy is None:
            limitations.append(
                f"{device_backend.value} accelerator energy could not be measured by CodeCarbon; "
                "total_energy_kwh (when present) covers only the components listed as accounted "
                "for and must not be read as complete run energy."
            )
        if cpu_energy is None:
            limitations.append(
                "CodeCarbon did not provide a finite CPU energy value for this resource pass."
            )
        if ram_energy is None:
            limitations.append(
                "CodeCarbon did not provide a finite RAM energy value for this resource pass."
            )
        if total_energy is None:
            limitations.append(
                "CodeCarbon did not provide a finite aggregate energy value; component-level "
                "energy evidence is retained where available."
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
            privilege_used=privilege_used,
            coverage_complete=coverage_complete,
            unaccounted_components=tuple(unaccounted),
            limitations=tuple(limitations),
        )
        return result, evidence


def select_energy_backend(
    mode: str, *, allow_privileged_energy: bool = False
) -> EnergyBackend[object]:
    """Return the backend for `--energy auto|off`."""

    if mode == "off":
        return OffEnergyBackend()
    if mode == "auto":
        return CodeCarbonEnergyBackend(allow_privileged_energy=allow_privileged_energy)
    raise ValueError(f"unknown energy mode {mode!r}; expected 'auto' or 'off'")
