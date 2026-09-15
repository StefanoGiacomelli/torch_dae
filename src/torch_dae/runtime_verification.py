"""Explicit requests for checkpoint-specific runtime verification."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from torch_dae.contracts import (
    SHA256_PATTERN,
    CanonicalId,
    StrictBaseModel,
    ensure_repository_relative,
    ensure_wrapper_entry_point,
)
from torch_dae.core.checkpoint import CheckpointAcquisitionPolicy, CheckpointSpec
from torch_dae.environment.results import ArtifactEvidence


class RuntimeInputContract(StrictBaseModel):
    """Expected public waveform contract for a verification request."""

    shape: Literal["B,C,T"]
    valid_lengths_shape: Literal["B"] | None
    dtype: str
    channels: str


class ExpectedRuntimeOutput(StrictBaseModel):
    """Expected named output and dimensions before execution."""

    name: CanonicalId
    rank: int = Field(ge=0)
    dimensions: tuple[str, ...]

    @model_validator(mode="after")
    def rank_matches_dimensions(self) -> ExpectedRuntimeOutput:
        if self.rank != len(self.dimensions):
            raise ValueError("runtime output rank must match dimensions")
        return self


class ExpectedRuntimeEmbedding(StrictBaseModel):
    """Default embedding identity and expected dimension."""

    embedding_id: CanonicalId
    dimension: int = Field(gt=0)


class RuntimeVerificationLimits(StrictBaseModel):
    """Execution bounds for one requested verification run."""

    timeout_seconds: int = Field(gt=0)
    maximum_batch_size: int = Field(gt=0)
    maximum_samples_per_item: int = Field(gt=0)
    repeated_calls: int = Field(ge=1)
    profiling_permitted: Literal[False] = False


class RuntimeVerificationTarget(StrictBaseModel):
    """Strict model/checkpoint/environment request that makes no success claim."""

    schema_version: Literal["1.0.0", "2.0.0"]
    target_id: CanonicalId
    workflow_id: CanonicalId
    integrated_variant_id: CanonicalId
    adapter_id: CanonicalId
    checkpoint: CheckpointSpec
    future_card_id: CanonicalId | None = None
    public_model_entry_point: str
    registry_identity: CanonicalId
    environment_id: CanonicalId
    accepted_integration_handoff: ArtifactEvidence
    environment_spec_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    source_manifest: ArtifactEvidence
    checkpoint_acquisition_policy: CheckpointAcquisitionPolicy
    required_sample_rate_hz: int = Field(gt=0)
    input_contract: RuntimeInputContract
    expected_outputs: tuple[ExpectedRuntimeOutput, ...]
    probability_semantics: Literal["sigmoid", "softmax", "other", "unsupported"]
    default_embedding: ExpectedRuntimeEmbedding
    permitted_devices: tuple[Literal["cpu", "mps", "cuda"], ...]
    required_check_ids: tuple[CanonicalId, ...] = ()
    optional_check_ids: tuple[CanonicalId, ...] = ()
    verification_limits: RuntimeVerificationLimits
    known_unresolved_items: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def completeness_fields_are_explicit(cls, value: object) -> object:
        if isinstance(value, dict) and value.get("schema_version") == "2.0.0":
            missing = sorted(
                name for name in ("required_check_ids", "optional_check_ids") if name not in value
            )
            if missing:
                raise ValueError(f"completeness-aware runtime target is missing: {missing}")
        return value

    @model_validator(mode="after")
    def validate_request(self) -> RuntimeVerificationTarget:
        ensure_wrapper_entry_point(self.public_model_entry_point)
        ensure_repository_relative(self.accepted_integration_handoff.path)
        ensure_repository_relative(self.source_manifest.path)
        if not self.expected_outputs:
            raise ValueError("runtime verification target requires expected outputs")
        output_ids = [item.name for item in self.expected_outputs]
        if len(output_ids) != len(set(output_ids)):
            raise ValueError("runtime verification output names must be unique")
        if not self.permitted_devices or len(self.permitted_devices) != len(
            set(self.permitted_devices)
        ):
            raise ValueError("permitted devices must be nonempty and unique")
        if self.schema_version == "2.0.0":
            if not self.required_check_ids:
                raise ValueError("runtime verification target requires at least one required check")
            if len(self.required_check_ids) != len(set(self.required_check_ids)):
                raise ValueError("required runtime check IDs must be unique")
            if len(self.optional_check_ids) != len(set(self.optional_check_ids)):
                raise ValueError("optional runtime check IDs must be unique")
            overlap = sorted(set(self.required_check_ids) & set(self.optional_check_ids))
            if overlap:
                raise ValueError(
                    f"required and optional runtime checks must be disjoint: {overlap}"
                )
        elif self.required_check_ids or self.optional_check_ids:
            raise ValueError(
                "legacy runtime targets cannot declare completeness-aware check contracts; "
                "migrate to schema 2.0.0"
            )
        policy = self.checkpoint_acquisition_policy
        if self.schema_version == "2.0.0" and policy.require_authority:
            if self.checkpoint.authority is None:
                raise ValueError("authority-complete runtime target requires checkpoint authority")
            if not (
                policy.require_exact_size
                and policy.require_published_checksums
                and policy.require_observed_sha256
            ):
                raise ValueError(
                    "authority-complete runtime target must require exact size, published "
                    "checksums, and observed SHA-256"
                )
            if (
                policy.maximum_bytes is not None
                and policy.maximum_bytes < self.checkpoint.authority.expected_size_bytes
            ):
                raise ValueError("maximum_bytes cannot be smaller than authoritative exact size")
        return self


def validate_runtime_verification_target(
    target: RuntimeVerificationTarget,
    repository_root: Path,
) -> RuntimeVerificationTarget:
    """Validate repository associations without acquiring or inspecting a checkpoint."""

    from torch_dae.environment.manager import EnvironmentManager
    from torch_dae.environment.sources import sha256_file
    from torch_dae.onboarding.contracts import HandoffStatus, OnboardingPhase
    from torch_dae.onboarding.handoff import load_handoff, validate_workflow

    root = Path(repository_root).resolve()
    expected_handoff = f"onboarding_reports/{target.workflow_id}/integrate/handoff.json"
    if target.accepted_integration_handoff.path != expected_handoff:
        raise ValueError("runtime target must reference the canonical integrate handoff")
    validate_workflow(root, target.workflow_id, phase=OnboardingPhase.INTEGRATE)
    handoff_path = root / expected_handoff
    if sha256_file(handoff_path) != target.accepted_integration_handoff.sha256:
        raise ValueError("runtime target integration handoff SHA-256 mismatch")
    handoff = load_handoff(handoff_path)
    if handoff.workflow_id != target.workflow_id or handoff.phase != OnboardingPhase.INTEGRATE:
        raise ValueError("runtime target integration handoff identity mismatch")
    if handoff.handoff_status != HandoffStatus.ACCEPTED:
        raise ValueError("runtime target requires an accepted integration handoff")
    if target.integrated_variant_id not in handoff.target_variant_ids:
        raise ValueError("runtime target variant is outside the integration handoff")
    if target.checkpoint.checkpoint_id not in handoff.target_checkpoint_ids:
        raise ValueError("runtime target checkpoint is outside the integration handoff")
    if target.future_card_id and target.future_card_id not in handoff.target_card_ids:
        raise ValueError("runtime target future card is outside the integration handoff")
    definition = EnvironmentManager(root).resolve_environment(target.environment_id)
    if definition.environment_spec_sha256 != target.environment_spec_sha256:
        raise ValueError("runtime target environment specification SHA-256 mismatch")
    if (
        definition.sources_path != target.source_manifest.path
        or definition.source_manifest_sha256 != target.source_manifest.sha256
    ):
        raise ValueError("runtime target source manifest mismatch")
    return target
