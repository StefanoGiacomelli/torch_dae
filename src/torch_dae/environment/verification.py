"""Environment verification contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, model_validator

from torch_dae.contracts import (
    SHA256_PATTERN,
    CanonicalId,
    StrictBaseModel,
    ensure_canonical_id,
)
from torch_dae.environment.results import ArtifactEvidence


class VerificationCheck(StrictBaseModel):
    """Single runtime verification check."""

    name: str
    status: Literal["passed", "failed", "unsupported"]
    details: str | None = None


class TensorDimension(StrictBaseModel):
    """Structured tensor dimension observation."""

    name: str
    size: int | None = Field(default=None, ge=0)
    dynamic: bool = False
    description: str | None = None


class TensorObservation(StrictBaseModel):
    """Structured tensor observation from runtime verification."""

    name: str
    role: str
    component_path: str
    shape: tuple[TensorDimension, ...]
    rank: int = Field(ge=0)
    dtype: str
    device: str
    lengths: str | None = None
    temporal_metadata: str | None = None

    @model_validator(mode="after")
    def rank_matches_shape(self) -> TensorObservation:
        if len(self.shape) != self.rank:
            raise ValueError("tensor observation rank must match shape length")
        return self


class VerificationReport(StrictBaseModel):
    """Runtime verification report contract."""

    schema_version: Literal["1.0.0", "2.0.0"]
    report_id: CanonicalId
    runtime_target_id: CanonicalId | None = None
    workflow_id: CanonicalId | None = None
    integrated_variant_id: CanonicalId | None = None
    checkpoint_id: CanonicalId | None = None
    public_model_entry_point: str | None = None
    integration_handoff_sha256: Annotated[str | None, Field(pattern=SHA256_PATTERN)] = None
    environment_spec_sha256: Annotated[str | None, Field(pattern=SHA256_PATTERN)] = None
    source_manifest_sha256: Annotated[str | None, Field(pattern=SHA256_PATTERN)] = None
    model_card_id: CanonicalId
    environment_id: CanonicalId
    environment_fingerprint: Annotated[str, Field(pattern=SHA256_PATTERN)]
    created_at: datetime
    platform: str
    device: str
    checkpoint_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    checkpoint_specification_fingerprint: Annotated[str | None, Field(pattern=SHA256_PATTERN)] = (
        None
    )
    checkpoint_materialization: ArtifactEvidence | None = None
    input_contracts: tuple[str, ...]
    tensor_observations: tuple[TensorObservation, ...]
    embedding_results: tuple[str, ...]
    passed_capabilities: tuple[str, ...]
    unsupported_capabilities: tuple[str, ...]
    known_limitations: tuple[str, ...]
    required_check_ids: tuple[CanonicalId, ...] = ()
    optional_check_ids: tuple[CanonicalId, ...] = ()
    checks: tuple[VerificationCheck, ...]
    verification_status: Literal["passed", "failed"] | None = None

    @model_validator(mode="before")
    @classmethod
    def completeness_fields_are_explicit(cls, value: object) -> object:
        if isinstance(value, dict) and value.get("schema_version") == "2.0.0":
            missing = sorted(
                name for name in ("required_check_ids", "optional_check_ids") if name not in value
            )
            if missing:
                raise ValueError(f"target-aware verification report is missing: {missing}")
        return value

    @model_validator(mode="after")
    def target_association_required_for_v2(self) -> VerificationReport:
        if self.schema_version == "2.0.0":
            required = {
                "runtime_target_id": self.runtime_target_id,
                "workflow_id": self.workflow_id,
                "integrated_variant_id": self.integrated_variant_id,
                "checkpoint_id": self.checkpoint_id,
                "public_model_entry_point": self.public_model_entry_point,
                "integration_handoff_sha256": self.integration_handoff_sha256,
                "environment_spec_sha256": self.environment_spec_sha256,
                "source_manifest_sha256": self.source_manifest_sha256,
            }
            missing = sorted(name for name, value in required.items() if value is None)
            if missing:
                raise ValueError(f"target-aware verification report is missing: {missing}")
            if self.verification_status is None:
                raise ValueError("target-aware verification report requires verification_status")
            if not self.required_check_ids:
                raise ValueError("target-aware verification report requires required checks")
            if len(self.required_check_ids) != len(set(self.required_check_ids)):
                raise ValueError("required runtime check IDs must be unique")
            if len(self.optional_check_ids) != len(set(self.optional_check_ids)):
                raise ValueError("optional runtime check IDs must be unique")
            overlap = sorted(set(self.required_check_ids) & set(self.optional_check_ids))
            if overlap:
                raise ValueError(
                    f"required and optional runtime checks must be disjoint: {overlap}"
                )
            if not self.checks:
                raise ValueError("target-aware verification report requires nonempty checks")
            check_names = [check.name for check in self.checks]
            for name in check_names:
                ensure_canonical_id(name)
            if len(check_names) != len(set(check_names)):
                raise ValueError("target-aware verification check names must be unique")
            declared = set(self.required_check_ids) | set(self.optional_check_ids)
            undeclared_checks = sorted(set(check_names) - declared)
            if undeclared_checks:
                raise ValueError(
                    f"verification report contains undeclared checks: {undeclared_checks}"
                )
            missing_required = sorted(set(self.required_check_ids) - set(check_names))
            if missing_required:
                raise ValueError(
                    f"verification report is missing required checks: {missing_required}"
                )
            status_by_name = {check.name: check.status for check in self.checks}
            unsupported_required = sorted(
                name for name in self.required_check_ids if status_by_name[name] == "unsupported"
            )
            if unsupported_required:
                raise ValueError(
                    f"required runtime checks cannot be unsupported: {unsupported_required}"
                )
            failed_checks = [check.name for check in self.checks if check.status == "failed"]
            if self.verification_status == "passed" and failed_checks:
                raise ValueError("passed verification report cannot contain failed checks")
            if self.verification_status == "passed":
                unpassed_required = sorted(
                    name for name in self.required_check_ids if status_by_name[name] != "passed"
                )
                if unpassed_required:
                    raise ValueError(
                        "passed verification report requires every required check to pass: "
                        f"{unpassed_required}"
                    )
            if self.verification_status == "failed" and not failed_checks:
                raise ValueError("failed verification report requires a failed check")
            unsupported_checks = [check for check in self.checks if check.status == "unsupported"]
            unsupported_names = {check.name for check in unsupported_checks}
            nonoptional_unsupported = sorted(unsupported_names - set(self.optional_check_ids))
            if nonoptional_unsupported:
                raise ValueError(
                    f"only declared optional checks may be unsupported: {nonoptional_unsupported}"
                )
            undeclared = sorted(
                check.name
                for check in unsupported_checks
                if check.name not in self.unsupported_capabilities
            )
            if undeclared:
                raise ValueError(
                    "unsupported checks require matching unsupported_capabilities entries: "
                    f"{undeclared}"
                )
            if unsupported_checks and any(check.details is None for check in unsupported_checks):
                raise ValueError("unsupported checks require limitation details")
            stale_unsupported = sorted(set(self.unsupported_capabilities) - unsupported_names)
            if stale_unsupported:
                raise ValueError(
                    "unsupported_capabilities contains checks without unsupported observations: "
                    f"{stale_unsupported}"
                )
            if self.unsupported_capabilities and not self.known_limitations:
                raise ValueError("unsupported capabilities require a recorded known limitation")
        else:
            if self.verification_status is not None:
                raise ValueError("legacy verification reports cannot declare verification_status")
            if self.required_check_ids or self.optional_check_ids:
                raise ValueError(
                    "legacy verification reports cannot declare target-aware check contracts"
                )
        return self

    @property
    def successful(self) -> bool:
        """Return explicit v2 success or conservative legacy compatibility success."""

        if any(check.status == "failed" for check in self.checks):
            return False
        if self.schema_version == "2.0.0":
            return self.verification_status == "passed"
        return True
