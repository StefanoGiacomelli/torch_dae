"""Strict typed contracts for Profiling v1 / Technical Card evidence.

Technical Cards are independent, immutable profiling evidence for one accepted Model Card,
one profiling protocol, one device/backend, one execution context, and one complete profiling
session (`project_spec.md` Section 27; `docs/profiling/technical-cards.md`). They never mutate,
enrich, or lifecycle-promote the referenced Model Card.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from torch_dae.contracts import (
    REPO_RELATIVE_PATTERN,
    SHA256_PATTERN,
    CanonicalId,
    StrictBaseModel,
    ensure_repository_relative,
)

TECHNICAL_CARD_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"
PROFILING_PROTOCOL_VERSION: Literal["1.0.0"] = "1.0.0"
PROFILING_PROTOCOL_ID: Literal["audio-inference-v1"] = "audio-inference-v1"
PROFILER_IMPLEMENTATION_VERSION: Literal["1.0.0"] = "1.0.0"

Sha256Str = Annotated[str, Field(pattern=SHA256_PATTERN)]


class RuntimeClassification(StrEnum):
    """Whether a profiling run used unmodified, modified, or unknown torch-dae source."""

    CANONICAL = "canonical"
    MODIFIED_RUNTIME = "modified_runtime"
    UNKNOWN_RUNTIME = "unknown_runtime"


class DeviceBackend(StrEnum):
    """Supported device/backend selectors."""

    CPU = "cpu"
    MPS = "mps"
    CUDA = "cuda"


class ThreadRegime(StrEnum):
    """CPU intra-op thread regime."""

    SINGLE_THREAD = "single_thread"
    NATIVE_DEFAULT = "native_default"


class MinimumInputStatus(StrEnum):
    """Outcome of the empirical minimum-input search (Section 11)."""

    FOUND = "found"
    NON_MONOTONIC = "non_monotonic"
    ALWAYS_UNSUPPORTED = "always_unsupported"
    MAX_DURATION_EXHAUSTED = "max_duration_exhausted"


class ConditionStatus(StrEnum):
    """Outcome of one benchmarked (device, thread-regime, duration, batch) condition."""

    SUCCESS = "success"
    UNSUPPORTED = "unsupported"
    NOT_TESTED = "not_tested"


class ArchitectureEvidenceStatus(StrEnum):
    """Coverage status for architecture/FLOPs evidence (Section 18)."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class EnergyMeasurementKind(StrEnum):
    """CodeCarbon measurement classification (Section 23)."""

    HARDWARE_MEASURED = "hardware_measured"
    SOFTWARE_ESTIMATED = "software_estimated"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class RawArrayDescriptor(StrictBaseModel):
    """One named array inside the raw `.npz` measurement manifest (Section 26)."""

    name: str
    dtype: str
    shape: tuple[int, ...]


class RawMeasurementManifest(StrictBaseModel):
    """Reference and manifest for the bounded lossless raw `.npz` asset.

    ``path`` is relative to the Technical Card JSON's own directory, never to the repository
    root, so a reviewed JSON+`.npz` pair validates identically before and after being relocated
    together (e.g. candidate workspace -> `technical_cards/<model-id>/`). Absolute paths and any
    `..` traversal or directory-escape are rejected.
    """

    path: Annotated[str, Field(pattern=REPO_RELATIVE_PATTERN)]
    sha256: Sha256Str
    media_type: Literal["application/x-npz"] = "application/x-npz"
    arrays: tuple[RawArrayDescriptor, ...]

    @field_validator("path")
    @classmethod
    def _relative(cls, value: str) -> str:
        result = ensure_repository_relative(value)
        assert result is not None
        return result


class ModelReference(StrictBaseModel):
    """Immutable reference to the accepted Model Card and checkpoint being profiled."""

    model_card_id: CanonicalId
    model_card_path: Annotated[str, Field(pattern=REPO_RELATIVE_PATTERN)]
    model_card_sha256: Sha256Str
    model_card_status: str
    checkpoint_id: CanonicalId
    checkpoint_sha256: Sha256Str
    wrapper_entry_point: str


class ProfilerImplementationProvenance(StrictBaseModel):
    """Identity of the profiling tooling that produced this evidence, distinct from the model."""

    profiler_implementation_version: str
    profiling_protocol_id: Literal["audio-inference-v1"]
    profiling_protocol_version: str
    torch_dae_package_version: str
    torch_dae_content_identity: str
    torch_dae_repository_head: str | None
    torch_dae_repository_dirty: bool | None
    runtime_classification: RuntimeClassification
    codecarbon_version: str | None = None
    psutil_version: str | None = None


class ContributorMetadata(StrictBaseModel):
    """Optional contributor identity; never participates in technical validity."""

    display_name: str | None = None
    github_handle: str | None = None


class HardwareMetadata(StrictBaseModel):
    """Privacy-safe hardware-configuration evidence (never a physical-machine identifier)."""

    cpu_model: str | None = None
    cpu_architecture: str
    physical_cores: int | None = None
    logical_cores: int | None = None
    total_ram_bytes: int | None = None
    accelerator_vendor: str | None = None
    accelerator_model: str | None = None
    accelerator_memory_bytes: int | None = None
    backend_capabilities: tuple[str, ...] = ()


class SoftwareMetadata(StrictBaseModel):
    """Software stack evidence relevant to reproducibility."""

    os_name: str
    os_version: str
    python_implementation: str
    python_version: str
    torch_version: str
    cuda_runtime_version: str | None = None
    cuda_driver_version: str | None = None
    cudnn_version: str | None = None
    mps_backend_info: str | None = None


class DeviceBackendMetadata(StrictBaseModel):
    """Concrete device/backend identity for one Technical Card."""

    backend: DeviceBackend
    device_index: int | None = None
    device_label: str


class ExecutionContextMetadata(StrictBaseModel):
    """Privacy-safe fingerprint binding reproducibility-relevant execution conditions."""

    hardware_fingerprint: Sha256Str
    execution_context_fingerprint: Sha256Str
    os_name: str
    os_version: str
    python_version: str
    torch_version: str
    torch_dae_content_identity: str
    device_backend: DeviceBackend
    native_precision: str
    thread_regime: ThreadRegime | None = None
    profiling_protocol_id: Literal["audio-inference-v1"]
    profiling_protocol_version: str
    profiler_implementation_version: str


class PrecisionMetadata(StrictBaseModel):
    """Native execution precision (only verified/native precision is benchmarked in v1)."""

    dtype: str
    autocast: bool = False


class SyntheticInputProvenance(StrictBaseModel):
    """Deterministic synthetic white-noise input provenance (Section 9)."""

    distribution: Literal["uniform"] = "uniform"
    low: float = -1.0
    high: float = 1.0
    dtype: Literal["float32"] = "float32"
    prng_implementation: str
    seed: int
    sample_rate: int
    sample_count: int
    batch_size: int
    channel_count: int
    tensor_sha256: Sha256Str | None = None


class ArchitectureEvidence(StrictBaseModel):
    """Mandatory + optional architecture profiling evidence (Section 18)."""

    total_parameters: int
    trainable_parameters: int
    non_trainable_parameters: int
    parameter_bytes: int
    buffer_bytes: int
    state_dict_tensor_bytes: int
    dtype_distribution: dict[str, int]
    module_count: int
    flops_macs_status: ArchitectureEvidenceStatus = ArchitectureEvidenceStatus.UNAVAILABLE
    flops: int | None = None
    macs: int | None = None
    counting_backend: str | None = None
    counting_backend_version: str | None = None
    counting_convention: str | None = None
    unaccounted_operations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _flops_status_consistency(self) -> ArchitectureEvidence:
        if self.flops_macs_status == ArchitectureEvidenceStatus.UNAVAILABLE and (
            self.flops is not None or self.macs is not None
        ):
            raise ValueError("unavailable FLOPs/MACs status must not carry values")
        return self


class ColdStartEvidence(StrictBaseModel):
    """Cold initialization/loading evidence, kept separate from steady-state timing."""

    initialization_ns: int | None = None
    checkpoint_loading_ns: int | None = None
    device_placement_ns: int | None = None
    first_inference_ns: int | None = None
    limitations: tuple[str, ...] = ()


class TimingSummary(StrictBaseModel):
    """Recomputable summary statistics over 50 raw steady-state latency observations."""

    sample_count: Literal[50]
    mean_ns: float
    stddev_ns: float
    min_ns: int
    max_ns: int
    p50_ns: float
    p90_ns: float
    p95_ns: float
    p99_ns: float
    coefficient_of_variation: float
    throughput_items_per_second: float
    forward_calls_per_second: float
    real_time_factor: float
    speed_factor: float


class ProfilingConditions(StrictBaseModel):
    """One benchmarked (duration, batch) condition and its outcome."""

    condition_id: str
    duration_seconds: float
    duration_source: Literal["canonical_contract", "protocol_default", "empirical_minimum"]
    batch_size: int
    sample_count: int
    status: ConditionStatus
    unsupported_reason: str | None = None
    timing: TimingSummary | None = None
    raw_timing_array: str | None = None

    @model_validator(mode="after")
    def _status_consistency(self) -> ProfilingConditions:
        if self.status == ConditionStatus.SUCCESS and self.timing is None:
            raise ValueError("success conditions require a timing summary")
        if self.status == ConditionStatus.UNSUPPORTED and not self.unsupported_reason:
            raise ValueError("unsupported conditions require a reason")
        if self.status != ConditionStatus.SUCCESS and self.timing is not None:
            raise ValueError("only success conditions carry a timing summary")
        return self


class MinimumInputProbeRecord(StrictBaseModel):
    """One ordered probe observation in the minimum-input search (Section 11).

    Persisted so the search can be audited after the fact: which sample counts were tried, in
    what order, and whether each one succeeded.
    """

    probe_order: int
    sample_count: int
    succeeded: bool


class MinimumInputSearchResult(StrictBaseModel):
    """Empirical minimum supported input search evidence (Section 11)."""

    status: MinimumInputStatus
    minimum_sample_count: int | None = None
    minimum_duration_seconds: float | None = None
    probe_sample_counts_tested: tuple[int, ...] = ()
    probe_records: tuple[MinimumInputProbeRecord, ...] = ()
    boundary_verified: bool = False
    below_boundary_verified: bool = False
    notes: str | None = None

    @model_validator(mode="after")
    def _found_requires_value(self) -> MinimumInputSearchResult:
        if self.status == MinimumInputStatus.FOUND and self.minimum_sample_count is None:
            raise ValueError("found status requires minimum_sample_count")
        return self


class HostMemoryEvidence(StrictBaseModel):
    """Host RAM / process RSS evidence (Section 19)."""

    total_ram_bytes: int | None = None
    rss_before_model_load_bytes: int | None = None
    rss_after_model_load_bytes: int | None = None
    rss_before_resource_pass_bytes: int | None = None
    rss_sampled_peak_bytes: int | None = None
    rss_sampled_peak_is_sampled: bool = True
    rss_after_resource_pass_bytes: int | None = None
    uss_bytes: int | None = None
    pss_bytes: int | None = None
    sampling_method: str | None = None


class AcceleratorMemoryEvidence(StrictBaseModel):
    """Device-native accelerator memory evidence (CUDA allocator or MPS surfaces)."""

    backend: DeviceBackend
    cuda_current_allocated_bytes: int | None = None
    cuda_peak_allocated_bytes: int | None = None
    cuda_current_reserved_bytes: int | None = None
    cuda_peak_reserved_bytes: int | None = None
    mps_current_allocated_bytes: int | None = None
    mps_driver_allocated_bytes: int | None = None
    unified_memory_note: str | None = None


class EnergyEvidence(StrictBaseModel):
    """CodeCarbon-backed energy evidence for the separate resource pass (Section 23).

    ``total_energy_kwh`` (when present) is only ever the sum of the components this evidence
    actually measured or estimated; it is never a claim of complete run/system energy. Whether it
    represents the *whole* active run is stated explicitly by ``coverage_complete`` and
    ``unaccounted_components`` -- e.g. an MPS run whose CPU/RAM energy is software-estimated but
    whose accelerator energy CodeCarbon cannot measure MUST set
    ``coverage_complete=False`` and ``unaccounted_components=("accelerator",)`` rather than
    silently presenting a CPU+RAM-only total as if it were the full run's energy.
    """

    measurement_kind: EnergyMeasurementKind
    codecarbon_version: str | None = None
    cpu_energy_kwh: float | None = None
    accelerator_energy_kwh: float | None = None
    ram_energy_kwh: float | None = None
    total_energy_kwh: float | None = None
    average_power_watts: float | None = None
    measurement_duration_seconds: float | None = None
    measurement_interval_seconds: float | None = None
    measurement_scope: str | None = None
    privilege_used: bool = False
    coverage_complete: bool = False
    unaccounted_components: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    failure_reason: str | None = None

    @field_validator(
        "cpu_energy_kwh",
        "accelerator_energy_kwh",
        "ram_energy_kwh",
        "total_energy_kwh",
        "average_power_watts",
        "measurement_duration_seconds",
        "measurement_interval_seconds",
    )
    @classmethod
    def _finite_numeric_evidence(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("energy numeric evidence must be finite when present")
        return value

    @model_validator(mode="after")
    def _kind_consistency(self) -> EnergyEvidence:
        if self.measurement_kind in (
            EnergyMeasurementKind.UNAVAILABLE,
            EnergyMeasurementKind.FAILED,
        ):
            for field_name in (
                "cpu_energy_kwh",
                "accelerator_energy_kwh",
                "ram_energy_kwh",
                "total_energy_kwh",
                "average_power_watts",
            ):
                if getattr(self, field_name) is not None:
                    raise ValueError(f"{self.measurement_kind} energy must not carry {field_name}")
            if self.coverage_complete:
                raise ValueError(f"{self.measurement_kind} energy cannot claim complete coverage")
        if self.measurement_kind == EnergyMeasurementKind.FAILED and not self.failure_reason:
            raise ValueError("failed energy evidence requires failure_reason")
        if self.coverage_complete and self.unaccounted_components:
            raise ValueError("coverage_complete=True must not list unaccounted_components")
        if self.coverage_complete and (self.cpu_energy_kwh is None or self.ram_energy_kwh is None):
            raise ValueError("coverage_complete=True requires finite CPU and RAM energy evidence")
        if (
            not self.coverage_complete
            and not self.unaccounted_components
            and self.measurement_kind
            in (EnergyMeasurementKind.HARDWARE_MEASURED, EnergyMeasurementKind.SOFTWARE_ESTIMATED)
        ):
            raise ValueError(
                "incomplete coverage must name at least one unaccounted_components entry"
            )
        return self


class MeasurementCoverage(StrictBaseModel):
    """Explicit tested/untested/unsupported coverage summary distinguishing absence types."""

    tested_conditions: tuple[str, ...]
    unsupported_conditions: tuple[str, ...] = ()
    not_tested_conditions: tuple[str, ...] = ()
    architecture_status: ArchitectureEvidenceStatus
    energy_status: EnergyMeasurementKind
    host_memory_available: bool
    accelerator_memory_available: bool


class ComparabilityMetadata(StrictBaseModel):
    """Explicit metadata for future cross-card comparability analytics."""

    protocol_id: Literal["audio-inference-v1"]
    protocol_version: str
    execution_context_fingerprint: Sha256Str
    runtime_classification: RuntimeClassification
    comparable_only_within_same_execution_context: bool = True


class SupersessionMetadata(StrictBaseModel):
    """Optional supersession pointer for a corrected later Technical Card."""

    supersedes: str | None = None
    supersede_reason: str | None = None

    @model_validator(mode="after")
    def _reason_requires_target(self) -> SupersessionMetadata:
        if self.supersede_reason and not self.supersedes:
            raise ValueError("supersede_reason requires supersedes")
        return self


class ProfilingCampaignReference(StrictBaseModel):
    """Reference back to the orchestrating campaign that produced this card."""

    campaign_id: str
    run_id: str


class TechnicalCardIdentity(StrictBaseModel):
    """Technical Card identity: a hash digest plus its stable human-facing form."""

    technical_card_id: str
    identity_digest_sha256: Sha256Str
    nonce: str

    @field_validator("technical_card_id")
    @classmethod
    def _human_form(cls, value: str) -> str:
        if not value.startswith("tc-"):
            raise ValueError("technical_card_id must use the 'tc-<digest prefix>' form")
        return value


class TechnicalCard(StrictBaseModel):
    """Complete immutable Technical Card: one Model Card, one protocol, one device, one session."""

    technical_card_schema_version: Literal["1.0.0"]
    identity: TechnicalCardIdentity
    model: ModelReference
    profiler: ProfilerImplementationProvenance
    contributor: ContributorMetadata = ContributorMetadata()
    hardware: HardwareMetadata
    software: SoftwareMetadata
    device: DeviceBackendMetadata
    execution_context: ExecutionContextMetadata
    precision: PrecisionMetadata
    synthetic_input: SyntheticInputProvenance
    canonical_duration_seconds: float
    canonical_duration_source: Literal["explicit_model_contract", "protocol_default"]
    architecture: ArchitectureEvidence
    cold_start: ColdStartEvidence
    minimum_input_search: MinimumInputSearchResult
    conditions: tuple[ProfilingConditions, ...]
    host_memory: HostMemoryEvidence
    accelerator_memory: AcceleratorMemoryEvidence | None = None
    energy: EnergyEvidence
    raw_measurements: RawMeasurementManifest
    coverage: MeasurementCoverage
    comparability: ComparabilityMetadata
    limitations: tuple[str, ...] = ()
    campaign: ProfilingCampaignReference
    supersession: SupersessionMetadata = SupersessionMetadata()
    created_at: str

    @model_validator(mode="after")
    def _at_least_one_condition(self) -> TechnicalCard:
        if not self.conditions:
            raise ValueError("a Technical Card must record at least one benchmarked condition")
        ids = [item.condition_id for item in self.conditions]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate condition_id values")
        return self


class DeviceSmokeDiagnostic(StrictBaseModel):
    """Bounded diagnostics for one device that failed its smoke inference (Section 30)."""

    device_label: str
    error: str


class SuccessfulDeviceRun(StrictBaseModel):
    """One successful device run's produced candidate evidence (Section 31)."""

    device_label: str
    thread_regime: ThreadRegime | None = None
    technical_card_id: str
    technical_card_path: Annotated[str, Field(pattern=REPO_RELATIVE_PATTERN)]
    raw_asset_path: Annotated[str, Field(pattern=REPO_RELATIVE_PATTERN)]
    validation_passed: bool
    validation_errors: tuple[str, ...] = ()


class ProfilingCampaignResult(StrictBaseModel):
    """Orchestration metadata for one profiling campaign; not itself a Technical Card."""

    campaign_id: str
    model_card_id: CanonicalId
    requested_devices: tuple[str, ...]
    detected_devices: tuple[str, ...]
    attempted_devices: tuple[str, ...]
    successful_device_runs: tuple[SuccessfulDeviceRun, ...]
    failed_device_diagnostics: tuple[DeviceSmokeDiagnostic, ...]
    profiling_protocol_id: Literal["audio-inference-v1"]
    profiling_protocol_version: str
    profiler_implementation_version: str
    energy_mode: Literal["auto", "off"]
    created_at: str
    # Informational only: points into the ignored `.torch-dae/` runtime-state tree, not a
    # committed repository artifact, so the strict dotfile-excluding path pattern does not apply.
    workspace_directory: str | None = None
