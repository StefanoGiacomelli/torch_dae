"""Strict model-card contracts for Phase 00."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, HttpUrl, field_validator, model_validator

from torch_dae.contracts import (
    GIT_REVISION_PATTERN,
    REPO_RELATIVE_PATTERN,
    CanonicalId,
    StrictBaseModel,
    ensure_repository_relative,
    ensure_wrapper_entry_point,
)
from torch_dae.core.checkpoint import CheckpointSpec
from torch_dae.core.embeddings import EmbeddingSpec


class ModelCardLifecycle(StrEnum):
    """Lifecycle states defined by project_spec.md."""

    DRAFT = "draft"
    ANALYZED = "analyzed"
    ENVIRONMENT_RESOLVED = "environment_resolved"
    CHECKPOINT_VERIFIED = "checkpoint_verified"
    RUNTIME_VERIFIED = "runtime_verified"
    PROFILED = "profiled"


class EvidenceStatus(StrEnum):
    """Evidence provenance status."""

    OFFICIALLY_REPORTED = "officially_reported"
    OBSERVED = "observed"
    INFERRED = "inferred"
    UNRESOLVED = "unresolved"
    NOT_REPORTED = "not_reported"
    NOT_APPLICABLE = "not_applicable"


class IssueStatus(StrEnum):
    """Issue lifecycle state."""

    OPEN = "open"
    RESOLVED = "resolved"
    ACCEPTED = "accepted"
    NOT_APPLICABLE = "not_applicable"


class ProfilingStatus(StrEnum):
    """Profiling status reserved for later phases."""

    NOT_PROFILED = "not_profiled"
    PROFILED = "profiled"


class EvidenceRecord(StrictBaseModel):
    """Concise evidence record for a material claim."""

    evidence_id: CanonicalId
    kind: str
    status: EvidenceStatus
    url: HttpUrl | None = None
    revision: Annotated[str | None, Field(pattern=GIT_REVISION_PATTERN)] = None
    path: str | None = None
    symbol: str | None = None
    description: str
    rationale: str | None = None

    @model_validator(mode="after")
    def inferred_requires_rationale(self) -> EvidenceRecord:
        if self.status == EvidenceStatus.INFERRED and not self.rationale:
            raise ValueError("inferred evidence requires rationale")
        return self

    @field_validator("path")
    @classmethod
    def path_repository_relative(cls, value: str | None) -> str | None:
        return ensure_repository_relative(value)


class IssueRecord(StrictBaseModel):
    """Explicit issue record; lifecycle state is not used for every problem."""

    issue_id: CanonicalId
    kind: str
    status: IssueStatus
    description: str
    impact: str


class Identity(StrictBaseModel):
    """Checkpoint-specific card identity."""

    model_name: str
    model_family: str
    variant: str
    checkpoint_name: str
    framework: Literal["pytorch"]
    wrapper_entry_point: Annotated[
        str,
        Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Z][A-Za-z0-9_]*$"),
    ]

    @field_validator("wrapper_entry_point")
    @classmethod
    def validate_entry_point(cls, value: str) -> str:
        return ensure_wrapper_entry_point(value)


class SourceRecord(StrictBaseModel):
    """Repository, package, or asset source evidence."""

    source_id: CanonicalId
    kind: str
    url: HttpUrl | None = None
    package: str | None = None
    revision: Annotated[str | None, Field(pattern=GIT_REVISION_PATTERN)] = None
    path: Annotated[str | None, Field(pattern=REPO_RELATIVE_PATTERN)] = None
    evidence_status: EvidenceStatus

    @field_validator("path")
    @classmethod
    def path_repository_relative(cls, value: str | None) -> str | None:
        return ensure_repository_relative(value)


class Sources(StrictBaseModel):
    """Required source roles."""

    official_repository: SourceRecord
    implementation: SourceRecord
    checkpoint: SourceRecord
    wrapper: SourceRecord


class ScientificReference(StrictBaseModel):
    """Publication metadata."""

    title: str
    doi: str | None = None
    official_publication: HttpUrl | None = None
    authors: tuple[str, ...]
    year: int | None = Field(default=None, ge=1900, le=2100)


class Description(StrictBaseModel):
    """Separated descriptive claims."""

    architecture: str
    preprocessing: str
    training_objective: str
    checkpoint_behavior: str
    implementation: str


class Tasks(StrictBaseModel):
    """Task roles separated by phase."""

    pretraining: tuple[str, ...]
    finetuning: tuple[str, ...]
    official_evaluation: tuple[str, ...]
    supported_inference: tuple[str, ...]


class DatasetRecord(StrictBaseModel):
    """Dataset usage record."""

    name: str
    version: str | None = None
    subset: str | None = None
    split: str | None = None
    role: str
    source_status: EvidenceStatus
    evidence_ids: tuple[str, ...] = ()


class Datasets(StrictBaseModel):
    """Training, validation, and testing dataset partitions."""

    training: tuple[DatasetRecord, ...]
    validation: tuple[DatasetRecord, ...]
    testing: tuple[DatasetRecord, ...]


class MetricRecord(StrictBaseModel):
    """Reported metric record."""

    task: str
    dataset: str
    split: str
    metric: str
    value: float | str | None
    unit: str | None = None
    protocol: str
    checkpoint_specific: bool
    source_status: EvidenceStatus
    evidence_ids: tuple[str, ...] = ()


class RecommendedEnvironment(StrictBaseModel):
    """Reference to committed environment artifacts."""

    environment_id: CanonicalId
    specification: Annotated[str, Field(pattern=REPO_RELATIVE_PATTERN)]
    lockfile: Annotated[str, Field(pattern=REPO_RELATIVE_PATTERN)]
    verified: bool

    @field_validator("specification", "lockfile")
    @classmethod
    def paths_repository_relative(cls, value: str) -> str:
        return ensure_repository_relative(value) or value


class Usage(StrictBaseModel):
    """Usage references and commands."""

    recommended_environment: RecommendedEnvironment
    installation_commands: tuple[str, ...]
    checkpoint_loading: tuple[str, ...]
    smoke_test_command: str


class WaveformInput(StrictBaseModel):
    """Canonical public waveform input contract."""

    shape: Literal["B,C,T"]
    sample_rate_hz: int = Field(gt=0)
    dtype: str = "float32"
    valid_lengths_shape: Literal["B"] | None = "B"
    channels: str
    resampling: str
    padding: str
    normalization: str


class OutputComponent(StrictBaseModel):
    """Stable output component description."""

    name: str
    kind: str
    semantic_kind: str
    rank: int = Field(ge=0)
    layout: str
    dimensions: tuple[str, ...]
    dtype: str | None = None
    granularity: str | None = None
    task_head_relation: str | None = None

    @model_validator(mode="after")
    def rank_matches_dimensions(self) -> OutputComponent:
        if len(self.dimensions) != self.rank:
            raise ValueError("output component rank must match dimensions length")
        return self


class Outputs(StrictBaseModel):
    """Forward and probability output declarations."""

    primary: OutputComponent
    components: tuple[OutputComponent, ...]
    probability_output: OutputComponent | None = None


class EmbeddingsSection(StrictBaseModel):
    """Embedding declarations with one selected default."""

    default_embedding_id: CanonicalId
    items: tuple[EmbeddingSpec, ...]

    @model_validator(mode="after")
    def validate_default(self) -> EmbeddingsSection:
        ids = [item.embedding_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("embedding IDs must be unique")
        defaults = [item for item in self.items if item.default]
        if len(defaults) != 1:
            raise ValueError("exactly one embedding must be marked default")
        if self.default_embedding_id != defaults[0].embedding_id:
            raise ValueError("default_embedding_id must refer to the declared default embedding")
        if self.default_embedding_id not in {item.embedding_id for item in self.items}:
            raise ValueError("default_embedding_id must refer to a declared embedding")
        return self


class BooleanCapability(StrictBaseModel):
    """Capability support plus optional reason."""

    supported: bool
    reason: str | None = None

    @model_validator(mode="after")
    def reason_for_unsupported(self) -> BooleanCapability:
        if not self.supported and not self.reason:
            raise ValueError("unsupported capabilities require a reason")
        return self


class CapabilitiesSection(StrictBaseModel):
    """Model-independent capability declarations."""

    random_initialization: BooleanCapability
    checkpoint_loading: BooleanCapability
    probabilities: BooleanCapability
    embeddings: BooleanCapability


class DeviceSupport(StrictBaseModel):
    """Declared and observed device support."""

    upstream_declared: tuple[str, ...]
    locally_tested: tuple[str, ...]
    known_limitations: tuple[str, ...]


class ProfilingSection(StrictBaseModel):
    """Profiling placeholder or completed profiling reference."""

    status: ProfilingStatus
    report: Annotated[str | None, Field(pattern=REPO_RELATIVE_PATTERN)] = None

    @field_validator("report")
    @classmethod
    def report_repository_relative(cls, value: str | None) -> str | None:
        return ensure_repository_relative(value)

    @model_validator(mode="after")
    def profiled_requires_report(self) -> ProfilingSection:
        if self.status == ProfilingStatus.PROFILED and self.report is None:
            raise ValueError("profiled sections require a report reference")
        return self


class ModelCard(StrictBaseModel):
    """Complete checkpoint-specific model card contract."""

    schema_version: Literal["1.0.0"]
    card_id: CanonicalId
    card_status: ModelCardLifecycle
    identity: Identity
    checkpoint: CheckpointSpec
    sources: Sources
    scientific_reference: ScientificReference
    description: Description
    tasks: Tasks
    datasets: Datasets
    reported_metrics: tuple[MetricRecord, ...]
    usage: Usage
    input: WaveformInput
    outputs: Outputs
    embeddings: EmbeddingsSection
    capabilities: CapabilitiesSection
    device_support: DeviceSupport
    verification_report: Annotated[str | None, Field(pattern=REPO_RELATIVE_PATTERN)] = None
    architectural_profiling: ProfilingSection
    inference_profiling: ProfilingSection
    energy_profiling: ProfilingSection
    limitations: tuple[str, ...]
    issues: tuple[IssueRecord, ...]
    evidence: tuple[EvidenceRecord, ...]

    @field_validator("verification_report")
    @classmethod
    def verification_report_repository_relative(cls, value: str | None) -> str | None:
        return ensure_repository_relative(value)

    @model_validator(mode="after")
    def validate_lifecycle(self) -> ModelCard:
        if (
            self.card_status
            in {
                ModelCardLifecycle.ENVIRONMENT_RESOLVED,
                ModelCardLifecycle.CHECKPOINT_VERIFIED,
                ModelCardLifecycle.RUNTIME_VERIFIED,
                ModelCardLifecycle.PROFILED,
            }
            and not self.usage.recommended_environment.verified
        ):
            raise ValueError("environment_resolved or later cards require verified environment")
        if (
            self.card_status
            in {
                ModelCardLifecycle.CHECKPOINT_VERIFIED,
                ModelCardLifecycle.RUNTIME_VERIFIED,
                ModelCardLifecycle.PROFILED,
            }
            and self.checkpoint.observed_sha256 is None
        ):
            raise ValueError(
                "checkpoint_verified or later cards require observed checkpoint SHA-256"
            )
        if (
            self.card_status
            in {
                ModelCardLifecycle.RUNTIME_VERIFIED,
                ModelCardLifecycle.PROFILED,
            }
            and self.verification_report is None
        ):
            raise ValueError("runtime_verified and profiled cards require verification_report")
        if self.card_status == ModelCardLifecycle.PROFILED:
            for section in (
                self.architectural_profiling,
                self.inference_profiling,
                self.energy_profiling,
            ):
                if section.status != ProfilingStatus.PROFILED:
                    raise ValueError(
                        "profiled cards require all profiling sections marked profiled"
                    )
        return self

    @model_validator(mode="after")
    def validate_capabilities_and_evidence(self) -> ModelCard:
        if self.capabilities.probabilities.supported != (
            self.outputs.probability_output is not None
        ):
            raise ValueError("probability capability must agree with probability output")
        if self.capabilities.embeddings.supported != bool(self.embeddings.items):
            raise ValueError("embedding capability must agree with embedding declarations")

        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence IDs must be unique")
        declared = set(evidence_ids)
        references: list[str] = []
        for dataset in self.datasets.training + self.datasets.validation + self.datasets.testing:
            references.extend(dataset.evidence_ids)
        for metric in self.reported_metrics:
            references.extend(metric.evidence_ids)
        for embedding in self.embeddings.items:
            references.extend(embedding.evidence_ids)
        missing = sorted(set(references) - declared)
        if missing:
            raise ValueError(f"evidence references are unresolved: {missing}")
        return self
