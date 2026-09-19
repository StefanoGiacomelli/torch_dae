"""Strict results for card-independent environment lifecycle operations."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from torch_dae.contracts import SHA256_PATTERN, CanonicalId, StrictBaseModel
from torch_dae.environment.specification import EnvironmentSourcesManifest, EnvironmentSpecification


class EnvironmentLifecycleState(StrEnum):
    """States owned by an environment rather than a model card."""

    DRAFT = "draft"
    MATERIALIZED = "materialized"
    VERIFIED = "environment_verified"


class EnvironmentFailureClassification(StrEnum):
    """Stable infrastructure failure classes for environment operations."""

    INVALID_SPECIFICATION = "invalid_specification"
    HASH_MISMATCH = "hash_mismatch"
    INTERPRETER_UNAVAILABLE = "interpreter_unavailable"
    PLATFORM_INCOMPATIBILITY = "platform_incompatibility"
    DEPENDENCY_INSTALLATION = "dependency_installation"
    SOURCE_PREPARATION = "source_preparation"
    DIRECT_DEPENDENCY_MISMATCH = "direct_dependency_mismatch"
    DEPENDENCY_CLOSURE = "dependency_closure"
    VERIFICATION_SCRIPT = "verification_script"
    SANDBOX_OR_EXECUTION_POLICY = "sandbox_or_execution_policy"
    EXTERNAL_COMMAND = "external_command"


class ArtifactEvidence(StrictBaseModel):
    """Repository or managed-runtime artifact plus its observed SHA-256."""

    path: str
    sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]


class VerificationObservation(StrictBaseModel):
    """One bounded environment-level import or smoke observation."""

    name: str
    status: Literal["passed", "failed", "unsupported"]
    details: str | None = None


class RuntimeRequirementEvidence(StrictBaseModel):
    """One active local-wheel requirement and its reachable accepted-lock versions."""

    requirement: str
    normalized_name: str
    specifier: str | None = None
    marker: str | None = None
    reachable_versions: tuple[str, ...] = ()


class EnvironmentDependencyClosureResult(StrictBaseModel):
    """Offline proof that a lock can satisfy the local package wheel runtime metadata."""

    schema_version: Literal["1.0.0"]
    environment_id: CanonicalId
    package_wheel_identity: str
    package_wheel_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    python_version: str
    platform: str
    installation_policy: Literal["locked-project-closure-then-local-wheel-no-deps"]
    active_runtime_requirements: tuple[str, ...]
    satisfied_requirements: tuple[RuntimeRequirementEvidence, ...]
    missing_requirements: tuple[RuntimeRequirementEvidence, ...]
    incompatible_requirements: tuple[RuntimeRequirementEvidence, ...]
    lockfile_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    status: Literal["passed", "failed"]
    failure_classification: EnvironmentFailureClassification | None = None
    result_path: str

    @model_validator(mode="after")
    def status_matches_evidence(self) -> EnvironmentDependencyClosureResult:
        active = tuple(
            item.requirement
            for item in (
                *self.satisfied_requirements,
                *self.missing_requirements,
                *self.incompatible_requirements,
            )
        )
        if sorted(active) != sorted(self.active_runtime_requirements):
            raise ValueError("preflight classifications must cover every active requirement")
        if len(active) != len(set(active)):
            raise ValueError("active runtime requirements must be classified exactly once")
        if self.status == "passed":
            if self.missing_requirements or self.incompatible_requirements:
                raise ValueError("passed preflight cannot contain unsatisfied requirements")
            if self.failure_classification is not None:
                raise ValueError("passed preflight cannot carry failure_classification")
        else:
            if not self.missing_requirements and not self.incompatible_requirements:
                raise ValueError("failed preflight requires missing or incompatible requirements")
            if self.failure_classification != EnvironmentFailureClassification.DEPENDENCY_CLOSURE:
                raise ValueError("failed preflight requires dependency_closure classification")
        return self


class ResolvedEnvironmentDefinition(StrictBaseModel):
    """Validated canonical inputs for one logical environment identity."""

    schema_version: Literal["1.0.0"]
    environment_id: CanonicalId
    specification: EnvironmentSpecification
    sources_manifest: EnvironmentSourcesManifest
    specification_path: str
    project_path: str
    lock_path: str
    sources_path: str
    verification_script_path: str
    environment_spec_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    project_file_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    lockfile_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    source_manifest_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    verification_script_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    python_implementation: Literal["CPython"]
    python_version: str
    platform: str
    direct_dependencies: dict[str, str]
    referenced_source_hashes: dict[str, Annotated[str, Field(pattern=SHA256_PATTERN)]]
    local_package_identity: str
    package_content_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    repository_head: str | None
    repository_dirty: bool | None
    environment_fingerprint: Annotated[str, Field(pattern=SHA256_PATTERN)]

    @model_validator(mode="after")
    def identities_agree(self) -> ResolvedEnvironmentDefinition:
        if self.specification.environment_id != self.environment_id:
            raise ValueError("resolved environment identity disagrees with specification")
        if self.sources_manifest.environment_id != self.environment_id:
            raise ValueError("resolved environment identity disagrees with source manifest")
        return self


class EnvironmentMaterializationResult(StrictBaseModel):
    """Result of creating or reusing environment infrastructure without verification."""

    schema_version: Literal["1.0.0"]
    environment_id: CanonicalId
    environment_spec_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    project_file_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    lockfile_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    source_manifest_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    verification_script_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    environment_fingerprint: Annotated[str, Field(pattern=SHA256_PATTERN)]
    local_package_identity: str
    package_content_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    repository_head: str | None
    repository_dirty: bool | None
    local_package_wheel_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    interpreter: str
    platform: str
    managed_identifier: str
    status: Literal["created", "reused", "failed"]
    dependency_installation_status: Literal["complete", "failed", "not_started"]
    source_preparation_status: Literal["complete", "failed", "not_started"]
    lifecycle_state: Literal[EnvironmentLifecycleState.MATERIALIZED]
    warnings: tuple[str, ...] = ()
    failure_classification: EnvironmentFailureClassification | None = None
    started_at: datetime
    completed_at: datetime
    materialization_record: ArtifactEvidence
    dependency_closure_preflight: ArtifactEvidence

    @model_validator(mode="after")
    def success_has_no_failure(self) -> EnvironmentMaterializationResult:
        if self.status in {"created", "reused"} and self.failure_classification is not None:
            raise ValueError("successful materialization cannot carry failure_classification")
        if self.status == "failed" and self.failure_classification is None:
            raise ValueError("failed materialization requires failure_classification")
        return self


class EnvironmentVerificationResult(StrictBaseModel):
    """Infrastructure-only compatibility evidence for a materialized environment."""

    schema_version: Literal["1.0.0"]
    environment_id: CanonicalId
    environment_spec_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    materialization_result_reference: ArtifactEvidence
    verification_script_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    exact_interpreter: str
    platform: str
    direct_dependency_versions: dict[str, str]
    import_results: tuple[VerificationObservation, ...]
    environment_smoke_results: tuple[VerificationObservation, ...]
    verification_status: Literal["passed", "failed"]
    lifecycle_state: Literal[
        EnvironmentLifecycleState.MATERIALIZED,
        EnvironmentLifecycleState.VERIFIED,
    ]
    environment_fingerprint: Annotated[str, Field(pattern=SHA256_PATTERN)]
    warnings: tuple[str, ...] = ()
    failure_classification: EnvironmentFailureClassification | None = None
    result_path: str
    evidence: tuple[ArtifactEvidence, ...]
    started_at: datetime
    completed_at: datetime

    @model_validator(mode="after")
    def status_matches_lifecycle(self) -> EnvironmentVerificationResult:
        if self.verification_status == "passed":
            if self.lifecycle_state != EnvironmentLifecycleState.VERIFIED:
                raise ValueError(
                    "successful verification requires the environment_verified lifecycle state"
                )
            if self.failure_classification is not None:
                raise ValueError("successful verification cannot carry failure_classification")
            if not self.import_results:
                raise ValueError("successful verification requires import observations")
            if not self.environment_smoke_results:
                raise ValueError("successful verification requires smoke observations")
            observations = (*self.import_results, *self.environment_smoke_results)
            unsuccessful = [item.name for item in observations if item.status != "passed"]
            if unsuccessful:
                raise ValueError(
                    f"successful verification requires every observation to pass: {unsuccessful}"
                )
        else:
            if self.lifecycle_state != EnvironmentLifecycleState.MATERIALIZED:
                raise ValueError("failed verification requires the materialized lifecycle state")
            if self.failure_classification is None:
                raise ValueError("failed verification requires failure_classification")
        observation_names = [
            item.name for item in (*self.import_results, *self.environment_smoke_results)
        ]
        if len(observation_names) != len(set(observation_names)):
            raise ValueError("environment observation names must be unique across all collections")
        return self
