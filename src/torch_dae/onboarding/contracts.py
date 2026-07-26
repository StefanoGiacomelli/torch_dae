"""Strict Phase 02 onboarding artifact contracts."""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version
from pydantic import Field, HttpUrl, field_validator, model_validator

from torch_dae.cards.models import ModelCardLifecycle
from torch_dae.contracts import (
    GIT_REVISION_PATTERN,
    REPO_RELATIVE_PATTERN,
    CanonicalId,
    StrictBaseModel,
    ensure_repository_relative,
)

ONBOARDING_FORBIDDEN_PATH_PREFIXES = (".git", ".venv", "venv")
GENERATED_PROJECT_EVIDENCE_PATH_PREFIXES = frozenset(
    {
        ".torch-dae",
        "environments",
        "model_cards",
        "reports",
        "verification_reports",
    }
)


class EvidenceItemKind(StrEnum):
    """Accepted evidence item sources for onboarding analysis."""

    SOURCE_FILE = "source_file"
    SOURCE_LINE_OR_SYMBOL = "source_line_or_symbol"
    PACKAGE_METADATA = "package_metadata"
    CONFIGURATION_FILE = "configuration_file"
    OFFICIAL_DOCUMENTATION = "official_documentation"
    PAPER = "paper"
    RUNTIME_OBSERVATION = "runtime_observation"
    AGENT_INFERENCE = "agent_inference"
    USER_DECISION = "user_decision"


AUTHORITATIVE_UPSTREAM_EVIDENCE_KINDS = frozenset(
    {
        EvidenceItemKind.SOURCE_FILE,
        EvidenceItemKind.SOURCE_LINE_OR_SYMBOL,
        EvidenceItemKind.PACKAGE_METADATA,
        EvidenceItemKind.CONFIGURATION_FILE,
        EvidenceItemKind.OFFICIAL_DOCUMENTATION,
        EvidenceItemKind.PAPER,
    }
)
UPSTREAM_PACKAGE_METADATA_FILENAMES = frozenset(
    {"METADATA", "PKG-INFO", "pyproject.toml", "setup.cfg", "setup.py"}
)
ENVIRONMENT_PACKAGE_IDENTITY_FILENAMES = frozenset(
    {"environment.json", "pyproject.toml", "uv.lock"}
)


class ClaimStatus(StrEnum):
    """Scientific status of a claim or candidate."""

    VERIFIED_UPSTREAM_FACT = "verified_upstream_fact"
    LOCALLY_OBSERVED_BEHAVIOR = "locally_observed_behavior"
    REASONED_INFERENCE = "reasoned_inference"
    USER_PROVIDED_DECISION = "user_provided_decision"
    UNRESOLVED_AMBIGUITY = "unresolved_ambiguity"
    UNSUPPORTED_CLAIM = "unsupported_claim"


class OpenQuestionClassification(StrEnum):
    """Required open-question classifications."""

    NEEDS_MORE_EVIDENCE = "needs_more_evidence"
    NEEDS_RUNTIME_PROBE = "needs_runtime_probe"
    NEEDS_ENVIRONMENT_RESOLUTION = "needs_environment_resolution"
    NEEDS_USER_DECISION = "needs_user_decision"
    UNSUPPORTED_UPSTREAM_CLAIM = "unsupported_upstream_claim"
    OUT_OF_SCOPE = "out_of_scope"


class SourceStrategy(StrEnum):
    """Source strategy decisions evaluated by the onboarding workflow."""

    OFFICIAL_PACKAGE = "official_package"
    PINNED_OFFICIAL_GIT_REPOSITORY = "pinned_official_git_repository"
    MINIMAL_VENDORED_ADAPTATION = "minimal_vendored_adaptation"
    EXTERNAL_PYTORCH_IMPLEMENTATION = "external_pytorch_implementation"
    UNSUPPORTED_OR_NON_EQUIVALENT_IMPLEMENTATION = "unsupported_or_non_equivalent_implementation"


class RecommendedNextMode(StrEnum):
    """Allowed next skill modes."""

    ANALYZE = "analyze"
    RESOLVE_ENVIRONMENT = "resolve-environment"
    INTEGRATE = "integrate"
    VERIFY = "verify"
    CARD = "card"
    PROFILE = "profile"


class FailureClassification(StrEnum):
    """Environment-resolution failure classifications required by Phase 02."""

    PYTHON_CONSTRAINT = "python_constraint"
    DEPENDENCY_CONFLICT = "dependency_conflict"
    RESOLUTION_FAILURE = "resolution_failure"
    REMOVED_API = "removed_api"
    DEPRECATED_API = "deprecated_api"
    BINARY_OR_ABI_INCOMPATIBILITY = "binary_or_abi_incompatibility"
    MISSING_BINARY_WHEEL = "missing_binary_wheel"
    TORCH_TORCHAUDIO_MISMATCH = "torch_torchaudio_mismatch"
    NUMPY_COMPATIBILITY = "numpy_compatibility"
    CHECKPOINT_INCOMPATIBILITY = "checkpoint_incompatibility"
    SOURCE_BUILD_FAILURE = "source_build_failure"
    IMPORT_FAILURE = "import_failure"
    RUNTIME_FAILURE = "runtime_failure"
    PLATFORM_INCOMPATIBILITY = "platform_incompatibility"
    ACCESS_OR_AUTHENTICATION_BLOCKER = "access_or_authentication_blocker"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class CandidateTrialStatus(StrEnum):
    """Observed status for an explicitly selected environment candidate."""

    NOT_ATTEMPTED = "not_attempted"
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"


class DependencyKind(StrEnum):
    """Static dependency declaration formats observed during inspection."""

    REQUIREMENT = "requirement"
    CONDA = "conda"
    VCS = "vcs"
    DIRECT_URL = "direct_url"
    EDITABLE = "editable"
    LOCAL_PATH = "local_path"
    LOCKED = "locked"
    UNKNOWN = "unknown"


ONBOARDING_EVIDENCE_PATH_PATTERN = r"[A-Za-z0-9.][A-Za-z0-9._/-]*"


class EvidenceItem(StrictBaseModel):
    """Atomic evidence item used by Phase 02 reports."""

    evidence_id: CanonicalId
    kind: EvidenceItemKind
    claim_status: ClaimStatus
    description: str
    source_file: Annotated[str | None, Field(pattern=ONBOARDING_EVIDENCE_PATH_PATTERN)] = None
    source_line_or_symbol: str | None = None
    url: HttpUrl | None = None
    revision: Annotated[str | None, Field(pattern=GIT_REVISION_PATTERN)] = None
    package_name: str | None = None
    package_version: str | None = None
    rationale: str | None = None

    @field_validator("source_file")
    @classmethod
    def source_file_onboarding_evidence_path(cls, value: str | None) -> str | None:
        return ensure_onboarding_evidence_path(value)

    @field_validator("package_name")
    @classmethod
    def normalize_package_name(cls, value: str | None) -> str | None:
        return canonicalize_name(value) if value is not None else None

    @field_validator("package_version")
    @classmethod
    def validate_package_version(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                Version(value)
            except InvalidVersion as exc:
                raise ValueError(f"invalid package version: {value}") from exc
        return value

    @model_validator(mode="after")
    def validate_status_kind_pair(self) -> EvidenceItem:
        if self.kind == EvidenceItemKind.AGENT_INFERENCE:
            if self.claim_status != ClaimStatus.REASONED_INFERENCE or not self.rationale:
                raise ValueError("agent inference evidence requires reasoned status and rationale")
        if self.kind == EvidenceItemKind.USER_DECISION and (
            self.claim_status != ClaimStatus.USER_PROVIDED_DECISION
        ):
            raise ValueError("user decision evidence requires user_provided_decision status")
        if self.claim_status == ClaimStatus.REASONED_INFERENCE and not self.rationale:
            raise ValueError("reasoned inferences require rationale")
        return self


class EvidenceBackedClaim(StrictBaseModel):
    """A claim that cannot silently promote inference to verified fact."""

    statement: str
    status: ClaimStatus
    evidence_ids: tuple[CanonicalId, ...] = ()
    rationale: str | None = None

    @model_validator(mode="after")
    def validate_evidence_policy(self) -> EvidenceBackedClaim:
        if (
            self.status
            in {
                ClaimStatus.VERIFIED_UPSTREAM_FACT,
                ClaimStatus.LOCALLY_OBSERVED_BEHAVIOR,
                ClaimStatus.REASONED_INFERENCE,
                ClaimStatus.USER_PROVIDED_DECISION,
            }
            and not self.evidence_ids
        ):
            raise ValueError(f"{self.status.value} claims require evidence references")
        if self.status == ClaimStatus.REASONED_INFERENCE and not self.rationale:
            raise ValueError("reasoned inference claims require rationale")
        return self


class ReportSection(StrictBaseModel):
    """Named report section rendered in machine and Markdown outputs."""

    summary: str
    claims: tuple[EvidenceBackedClaim, ...] = ()


class RepositoryIdentity(StrictBaseModel):
    """Repository identity fields required by analyze mode."""

    canonical_repository_url: str | None
    repository_owner: str | None
    repository_name: str | None
    revision_inspected: str | None
    license_evidence: tuple[EvidenceBackedClaim, ...]
    package_name: str | None = None
    release_or_tag_evidence: tuple[EvidenceBackedClaim, ...] = ()
    maintenance_status_evidence: tuple[EvidenceBackedClaim, ...] = ()
    official_status: EvidenceBackedClaim


class VariantCandidate(StrictBaseModel):
    """Candidate model variant found during static analysis."""

    variant_id: CanonicalId
    name: str
    status: ClaimStatus
    evidence_ids: tuple[CanonicalId, ...]
    unresolved_reason: str | None = None

    @model_validator(mode="after")
    def candidate_requires_evidence_or_reason(self) -> VariantCandidate:
        _validate_candidate_evidence(self.status, self.evidence_ids, self.unresolved_reason)
        return self


class CheckpointCandidate(StrictBaseModel):
    """Candidate checkpoint metadata."""

    checkpoint_id: CanonicalId
    source_type: str
    filename: str | None = None
    url: str | None = None
    model_variant: str | None = None
    loader: str | None = None
    hash_evidence: str | None = None
    access_or_license_notes: str | None = None
    helper_symbol: str | None = None
    expression_status: str | None = None
    unresolved_components: tuple[str, ...] | None = None
    evidence_ids: tuple[CanonicalId, ...]
    status: ClaimStatus = ClaimStatus.LOCALLY_OBSERVED_BEHAVIOR
    unresolved_reason: str | None = None

    @model_validator(mode="after")
    def candidate_requires_evidence_or_reason(self) -> CheckpointCandidate:
        _validate_candidate_evidence(self.status, self.evidence_ids, self.unresolved_reason)
        if self.source_type == "https" and self.helper_symbol and not self.expression_status:
            raise ValueError("checkpoint helper candidates require expression_status")
        return self


class SourceStrategyCandidate(StrictBaseModel):
    """Evidence-supported source strategy candidate."""

    strategy: SourceStrategy
    status: ClaimStatus
    rationale: str
    evidence_ids: tuple[CanonicalId, ...]
    user_decision_required: bool = False
    unresolved_reason: str | None = None

    @model_validator(mode="after")
    def candidate_requires_evidence_or_reason(self) -> SourceStrategyCandidate:
        _validate_candidate_evidence(self.status, self.evidence_ids, self.unresolved_reason)
        return self


class EmbeddingCandidate(StrictBaseModel):
    """Candidate embedding tensor whose semantics require evidence."""

    embedding_id: CanonicalId
    tensor_origin: str
    semantic_kind: Literal[
        "architectural_intermediate_tensor",
        "pooled_representation",
        "task_head_input",
        "pre_logit_representation",
        "post_activation_output",
        "sequence_level_embedding",
        "frame_level_embedding",
        "latent_code",
    ]
    shape_semantics: str | None
    batch_dimension: str | None
    time_dimension: str | None
    status: ClaimStatus
    evidence_ids: tuple[CanonicalId, ...]
    requires_user_decision: bool = False
    unresolved_reason: str | None = None

    @model_validator(mode="after")
    def candidate_requires_evidence_or_reason(self) -> EmbeddingCandidate:
        _validate_candidate_evidence(self.status, self.evidence_ids, self.unresolved_reason)
        return self


class OpenQuestion(StrictBaseModel):
    """Explicit unresolved item carried forward by the workflow."""

    question_id: CanonicalId
    classification: OpenQuestionClassification
    description: str
    alternatives: tuple[str, ...] = ()
    evidence_ids: tuple[CanonicalId, ...] = ()
    default_if_deferred: str | None = None
    failure_classification: FailureClassification | None = None

    @model_validator(mode="after")
    def user_decisions_need_alternatives(self) -> OpenQuestion:
        if (
            self.classification == OpenQuestionClassification.NEEDS_USER_DECISION
            and len(self.alternatives) < 2
        ):
            raise ValueError("user decision questions require at least two alternatives")
        return self


class DecisionRecord(StrictBaseModel):
    """A user-provided or evidence-determined decision."""

    decision_id: CanonicalId
    decision: str
    selected_option: str | None = None
    status: ClaimStatus
    evidence_ids: tuple[CanonicalId, ...] = ()
    rationale: str | None = None

    @model_validator(mode="after")
    def decision_requires_evidence_or_reason(self) -> DecisionRecord:
        _validate_candidate_evidence(self.status, self.evidence_ids, self.rationale)
        return self


class DependencyEvidenceRecord(StrictBaseModel):
    """Normalized dependency evidence with explicit provenance."""

    normalized_name: str | None = None
    raw_declaration: str
    constraint: str | None = None
    exact_version: str | None = None
    source_file: Annotated[str, Field(pattern=ONBOARDING_EVIDENCE_PATH_PATTERN)]
    source_section: str
    dependency_kind: DependencyKind
    editable: bool = False
    direct_url: bool = False
    vcs: str | None = None
    local_path: bool = False
    valid: bool
    evidence_id: CanonicalId

    @field_validator("source_file")
    @classmethod
    def source_file_onboarding_evidence_path(cls, value: str) -> str:
        return ensure_onboarding_evidence_path(value) or value

    @model_validator(mode="after")
    def validate_versions(self) -> DependencyEvidenceRecord:
        if self.constraint:
            try:
                SpecifierSet(self.constraint)
            except InvalidSpecifier as exc:
                raise ValueError(f"invalid dependency constraint: {self.constraint}") from exc
        if self.exact_version:
            try:
                version = Version(self.exact_version)
            except InvalidVersion as exc:
                raise ValueError(f"invalid dependency exact version: {self.exact_version}") from exc
            if self.constraint and version not in SpecifierSet(self.constraint):
                raise ValueError(
                    f"exact dependency version {self.exact_version} is outside {self.constraint}"
                )
        return self


class ConfidenceSummary(StrictBaseModel):
    """Summary counts used to expose uncertainty."""

    verified_fact_count: int = Field(ge=0)
    locally_observed_count: int = Field(ge=0)
    inference_count: int = Field(ge=0)
    unresolved_count: int = Field(ge=0)
    unsupported_claim_count: int = Field(ge=0)


class AnalysisReport(StrictBaseModel):
    """Machine-readable Phase 02 technical analysis report."""

    schema_version: Literal["1.0.0"]
    report_id: CanonicalId
    created_at: datetime
    repository: RepositoryIdentity
    revision: str | None
    official_status: EvidenceBackedClaim
    license_evidence: tuple[EvidenceBackedClaim, ...]
    scientific_identity: ReportSection
    architecture: ReportSection
    variants: tuple[VariantCandidate, ...]
    runtime_interface: ReportSection
    preprocessing: ReportSection
    outputs: ReportSection
    embedding_candidates: tuple[EmbeddingCandidate, ...]
    checkpoint_candidates: tuple[CheckpointCandidate, ...]
    dependency_evidence: ReportSection
    source_strategy_candidates: tuple[SourceStrategyCandidate, ...]
    environment_evidence: ReportSection
    open_questions: tuple[OpenQuestion, ...]
    decisions: tuple[DecisionRecord, ...]
    evidence_items: tuple[EvidenceItem, ...]
    confidence_summary: ConfidenceSummary
    recommended_next_mode: RecommendedNextMode

    @model_validator(mode="after")
    def validate_evidence_references(self) -> AnalysisReport:
        evidence_by_id = {item.evidence_id: item for item in self.evidence_items}
        if len(evidence_by_id) != len(self.evidence_items):
            raise ValueError("evidence item IDs must be unique")
        missing = sorted(set(_collect_evidence_references(self)) - set(evidence_by_id))
        if missing:
            raise ValueError(f"evidence references are unresolved: {missing}")

        if self.repository.official_status != self.official_status:
            raise ValueError("repository.official_status and official_status must agree")
        if self.repository.license_evidence != self.license_evidence:
            raise ValueError("repository.license_evidence and license_evidence must agree")
        if self.repository.revision_inspected != self.revision:
            raise ValueError("repository.revision_inspected and revision must agree")

        duplicate_errors = _duplicate_ids(
            ("variant", [item.variant_id for item in self.variants]),
            ("checkpoint", [item.checkpoint_id for item in self.checkpoint_candidates]),
            ("embedding", [item.embedding_id for item in self.embedding_candidates]),
            ("open question", [item.question_id for item in self.open_questions]),
            ("decision", [item.decision_id for item in self.decisions]),
        )
        if duplicate_errors:
            raise ValueError("; ".join(duplicate_errors))

        compatibility_errors: list[str] = []
        for claim in _iter_claims(self):
            compatibility_errors.extend(_evidence_compatibility_errors(claim, evidence_by_id))
        for variant in self.variants:
            compatibility_errors.extend(
                _evidence_status_compatibility_errors(
                    "variant",
                    variant.status,
                    variant.evidence_ids,
                    evidence_by_id,
                    variant.unresolved_reason,
                )
            )
        for checkpoint in self.checkpoint_candidates:
            compatibility_errors.extend(
                _evidence_status_compatibility_errors(
                    "checkpoint",
                    checkpoint.status,
                    checkpoint.evidence_ids,
                    evidence_by_id,
                    checkpoint.unresolved_reason,
                )
            )
        for source_strategy in self.source_strategy_candidates:
            compatibility_errors.extend(
                _evidence_status_compatibility_errors(
                    "source strategy",
                    source_strategy.status,
                    source_strategy.evidence_ids,
                    evidence_by_id,
                    source_strategy.rationale,
                )
            )
        for embedding in self.embedding_candidates:
            compatibility_errors.extend(
                _evidence_status_compatibility_errors(
                    "embedding",
                    embedding.status,
                    embedding.evidence_ids,
                    evidence_by_id,
                    embedding.unresolved_reason,
                )
            )
        for decision in self.decisions:
            compatibility_errors.extend(
                _evidence_status_compatibility_errors(
                    "decision",
                    decision.status,
                    decision.evidence_ids,
                    evidence_by_id,
                    decision.rationale,
                )
            )
        for question in self.open_questions:
            if question.evidence_ids:
                compatibility_errors.extend(
                    _evidence_status_compatibility_errors(
                        "open question",
                        ClaimStatus.UNRESOLVED_AMBIGUITY,
                        question.evidence_ids,
                        evidence_by_id,
                        question.description,
                        question.classification,
                    )
                )
        if compatibility_errors:
            raise ValueError("; ".join(sorted(set(compatibility_errors))))

        expected_confidence = _confidence_summary(self)
        if self.confidence_summary != expected_confidence:
            raise ValueError(
                "confidence_summary does not match report content: "
                f"expected {expected_confidence.model_dump()}"
            )

        if self.recommended_next_mode == RecommendedNextMode.PROFILE:
            raise ValueError("profile mode is reserved and cannot be a Phase 02 next step")
        return self


class EnvironmentCandidate(StrictBaseModel):
    """Evidence-motivated compatibility candidate; not a verified environment."""

    candidate_id: CanonicalId
    reason_for_selection: str
    python_constraint: str | None = None
    python_version: str | None = None
    pytorch_constraint: str | None = None
    pytorch_version: str | None = None
    torchaudio_constraint: str | None = None
    torchaudio_version: str | None = None
    numpy_constraint: str | None = None
    numpy_version: str | None = None
    other_principal_dependencies: dict[str, str] = Field(default_factory=dict)
    source_revision: str | None = None
    source_package_name: str | None = None
    source_package_version: str | None = None
    installation_strategy: SourceStrategy
    expected_compatibility_evidence: tuple[CanonicalId, ...]
    trial_status: CandidateTrialStatus = CandidateTrialStatus.NOT_ATTEMPTED
    failure_classification: FailureClassification | None = None
    failure_diagnostics: str | None = None
    uncertainty: tuple[str, ...] = ()
    predicted_failure_risks: tuple[FailureClassification, ...] = ()
    trial_command_plan: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_candidate_state(self) -> EnvironmentCandidate:
        if not self.expected_compatibility_evidence:
            raise ValueError("environment candidates require compatibility evidence references")
        if self.source_package_name is not None:
            object.__setattr__(
                self,
                "source_package_name",
                canonicalize_name(self.source_package_name),
            )
        if self.source_package_version is not None:
            try:
                Version(self.source_package_version)
            except InvalidVersion as exc:
                raise ValueError(
                    f"invalid source_package_version: {self.source_package_version}"
                ) from exc
        if self.installation_strategy == SourceStrategy.OFFICIAL_PACKAGE and (
            not self.source_package_name or not self.source_package_version
        ):
            raise ValueError(
                "official_package candidates require source_package_name and source_package_version"
            )
        if self.installation_strategy in {
            SourceStrategy.PINNED_OFFICIAL_GIT_REPOSITORY,
            SourceStrategy.MINIMAL_VENDORED_ADAPTATION,
        } and not _is_exact_lowercase_git_sha(self.source_revision):
            raise ValueError(
                f"{self.installation_strategy.value} requires exact 40-character lowercase "
                "source_revision"
            )
        _validate_version_constraint("python", self.python_version, self.python_constraint)
        _validate_version_constraint("pytorch", self.pytorch_version, self.pytorch_constraint)
        _validate_version_constraint(
            "torchaudio", self.torchaudio_version, self.torchaudio_constraint
        )
        _validate_version_constraint("numpy", self.numpy_version, self.numpy_constraint)
        if self.trial_status == CandidateTrialStatus.FAILED and self.failure_classification is None:
            raise ValueError("failed candidates require failure_classification")
        if self.trial_status == CandidateTrialStatus.PASSED and self.failure_classification:
            raise ValueError("passed candidates must not have failure_classification")
        return self


class EnvironmentCandidateGenerationResult(StrictBaseModel):
    """Strict output from static environment-candidate generation."""

    schema_version: Literal["1.0.0"]
    evidence_items: tuple[EvidenceItem, ...]
    dependency_records: tuple[DependencyEvidenceRecord, ...] = ()
    candidates: tuple[EnvironmentCandidate, ...]
    unresolved_constraints: tuple[str, ...] = ()
    source_strategy_context: tuple[SourceStrategyCandidate, ...]
    decision_gates: tuple[OpenQuestion, ...] = ()
    target_platform: str | None = None

    @model_validator(mode="after")
    def validate_result(self) -> EnvironmentCandidateGenerationResult:
        evidence_by_id = {item.evidence_id: item for item in self.evidence_items}
        if len(evidence_by_id) != len(self.evidence_items):
            raise ValueError("evidence item IDs must be unique")
        missing = sorted(set(_collect_evidence_references(self)) - set(evidence_by_id))
        if missing:
            raise ValueError(f"evidence references are unresolved: {missing}")
        errors: list[str] = []
        for record in self.dependency_records:
            errors.extend(
                _evidence_status_compatibility_errors(
                    "dependency record",
                    ClaimStatus.LOCALLY_OBSERVED_BEHAVIOR,
                    (record.evidence_id,),
                    evidence_by_id,
                    None,
                )
            )
        for source_strategy in self.source_strategy_context:
            errors.extend(
                _evidence_status_compatibility_errors(
                    "source strategy",
                    source_strategy.status,
                    source_strategy.evidence_ids,
                    evidence_by_id,
                    source_strategy.rationale,
                )
            )
        for candidate in self.candidates:
            errors.extend(
                _evidence_status_compatibility_errors(
                    "environment candidate",
                    ClaimStatus.REASONED_INFERENCE,
                    candidate.expected_compatibility_evidence,
                    evidence_by_id,
                    candidate.reason_for_selection,
                )
            )
        for gate in self.decision_gates:
            errors.extend(
                _evidence_status_compatibility_errors(
                    "decision gate",
                    ClaimStatus.UNRESOLVED_AMBIGUITY,
                    gate.evidence_ids,
                    evidence_by_id,
                    gate.description,
                    gate.classification,
                )
            )
        if errors:
            raise ValueError("; ".join(sorted(set(errors))))
        return self


class EnvironmentResolutionReport(StrictBaseModel):
    """Structured output from resolve-environment mode."""

    schema_version: Literal["1.0.0"]
    report_id: CanonicalId
    created_at: datetime
    analysis_report_id: CanonicalId | None = None
    evidence_items: tuple[EvidenceItem, ...] = ()
    evidence_summary: tuple[EvidenceBackedClaim, ...]
    ordered_candidates: tuple[EnvironmentCandidate, ...]
    attempted_candidates: tuple[CanonicalId, ...] = ()
    selected_candidate_id: CanonicalId | None = None
    unresolved_risks: tuple[str, ...] = ()
    source_strategy_decision_gates: tuple[OpenQuestion, ...] = ()
    phase01_artifact_paths: tuple[Annotated[str, Field(pattern=REPO_RELATIVE_PATTERN)], ...] = ()
    phase01_materialization_succeeded: bool = False
    phase01_verification_succeeded: bool = False
    environment_fingerprint: Annotated[str | None, Field(pattern=r"^[0-9a-f]{64}$")] = None
    verification_report_or_diagnostic_reference: Annotated[
        str | None, Field(pattern=REPO_RELATIVE_PATTERN)
    ] = None
    next_lifecycle_status: Literal[ModelCardLifecycle.ENVIRONMENT_RESOLVED] | None = None

    @field_validator("phase01_artifact_paths")
    @classmethod
    def artifact_paths_repository_relative(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            ensure_repository_relative(item)
        return value

    @field_validator("verification_report_or_diagnostic_reference")
    @classmethod
    def diagnostic_reference_repository_relative(cls, value: str | None) -> str | None:
        return ensure_repository_relative(value)

    @model_validator(mode="after")
    def selected_candidate_must_exist(self) -> EnvironmentResolutionReport:
        candidate_ids = {candidate.candidate_id for candidate in self.ordered_candidates}
        if len(candidate_ids) != len(self.ordered_candidates):
            raise ValueError("environment candidate IDs must be unique")
        if self.selected_candidate_id and self.selected_candidate_id not in candidate_ids:
            raise ValueError("selected_candidate_id must refer to an ordered candidate")
        missing_attempts = sorted(set(self.attempted_candidates) - candidate_ids)
        if missing_attempts:
            raise ValueError(f"attempted candidate IDs are unresolved: {missing_attempts}")
        evidence_by_id = {item.evidence_id: item for item in self.evidence_items}
        if len(evidence_by_id) != len(self.evidence_items):
            raise ValueError("evidence item IDs must be unique")
        missing = sorted(set(_collect_evidence_references(self)) - set(evidence_by_id))
        if missing:
            raise ValueError(f"evidence references are unresolved: {missing}")
        if self.next_lifecycle_status == ModelCardLifecycle.ENVIRONMENT_RESOLVED:
            if self.selected_candidate_id is None:
                raise ValueError("environment_resolved requires selected_candidate_id")
            selected = next(
                candidate
                for candidate in self.ordered_candidates
                if candidate.candidate_id == self.selected_candidate_id
            )
            if self.selected_candidate_id not in self.attempted_candidates:
                raise ValueError("environment_resolved requires selected candidate to be attempted")
            if selected.trial_status != CandidateTrialStatus.PASSED:
                raise ValueError(
                    "environment_resolved requires selected candidate trial_status passed"
                )
            if selected.failure_classification is not None:
                raise ValueError("environment_resolved selected candidate must not have failure")
            if (
                selected.installation_strategy
                == SourceStrategy.UNSUPPORTED_OR_NON_EQUIVALENT_IMPLEMENTATION
            ):
                raise ValueError("environment_resolved rejects unsupported source strategies")
            if selected.installation_strategy in {
                SourceStrategy.PINNED_OFFICIAL_GIT_REPOSITORY,
                SourceStrategy.MINIMAL_VENDORED_ADAPTATION,
            } and not _is_exact_lowercase_git_sha(selected.source_revision):
                raise ValueError(
                    "environment_resolved pinned Git or vendored strategies require exact "
                    "40-character lowercase source_revision"
                )
            if selected.installation_strategy == SourceStrategy.OFFICIAL_PACKAGE:
                if not (
                    selected.source_package_name is not None
                    and selected.source_package_version is not None
                ):
                    raise ValueError(
                        "environment_resolved official-package strategy requires exact "
                        "source package identity"
                    )
                if not _has_exact_package_identity_evidence(
                    selected,
                    evidence_by_id,
                    set(selected.expected_compatibility_evidence),
                    set(self.phase01_artifact_paths),
                ):
                    raise ValueError(
                        "environment_resolved official-package strategy requires exact "
                        "package/version evidence matching the selected candidate and "
                        "environment artifacts"
                    )
            if not (
                selected.python_version
                and selected.pytorch_version
                and selected.torchaudio_version
                and selected.numpy_version
            ):
                raise ValueError("environment_resolved requires exact Python/Torch/NumPy choices")
            for label, version, constraint in (
                ("python", selected.python_version, selected.python_constraint),
                ("pytorch", selected.pytorch_version, selected.pytorch_constraint),
                ("torchaudio", selected.torchaudio_version, selected.torchaudio_constraint),
                ("numpy", selected.numpy_version, selected.numpy_constraint),
            ):
                _validate_version_constraint(label, version, constraint)
            required_names = {
                "environment.json",
                "pyproject.toml",
                "uv.lock",
                "sources.json",
                "verify_environment.py",
            }
            actual_names = {
                item.rsplit("/", 1)[-1]
                for item in self.phase01_artifact_paths
                if item.startswith("environments/")
            }
            actual_dirs = {
                item.rsplit("/", 1)[0]
                for item in self.phase01_artifact_paths
                if item.startswith("environments/")
            }
            if actual_names != required_names or len(self.phase01_artifact_paths) != 5:
                raise ValueError("environment_resolved requires all five Phase 01 artifact paths")
            if len(actual_dirs) != 1:
                raise ValueError("environment_resolved requires one Phase 01 artifact directory")
            artifact_card_id = next(iter(actual_dirs)).split("/", 1)[1]
            if not self.phase01_materialization_succeeded:
                raise ValueError("environment_resolved requires Phase 01 materialization success")
            if not self.phase01_verification_succeeded:
                raise ValueError("environment_resolved requires Phase 01 verification success")
            if self.environment_fingerprint is None:
                raise ValueError("environment_resolved requires environment_fingerprint")
            if self.verification_report_or_diagnostic_reference is None:
                raise ValueError(
                    "environment_resolved requires verification_report_or_diagnostic_reference"
                )
            if not _valid_environment_report_reference(
                self.verification_report_or_diagnostic_reference,
                artifact_card_id,
                self.environment_fingerprint,
            ):
                raise ValueError(
                    "environment_resolved requires a recognized verification or diagnostic "
                    "report reference"
                )
            if self.unresolved_risks:
                raise ValueError("environment_resolved cannot retain unresolved blockers")
            if self.source_strategy_decision_gates:
                raise ValueError(
                    "environment_resolved requires source-strategy decision gates to be resolved"
                )
        compatibility_errors: list[str] = []
        for claim in self.evidence_summary:
            compatibility_errors.extend(_evidence_compatibility_errors(claim, evidence_by_id))
        for candidate in self.ordered_candidates:
            compatibility_errors.extend(
                _evidence_status_compatibility_errors(
                    "environment candidate",
                    ClaimStatus.REASONED_INFERENCE,
                    candidate.expected_compatibility_evidence,
                    evidence_by_id,
                    candidate.reason_for_selection,
                )
            )
        for gate in self.source_strategy_decision_gates:
            compatibility_errors.extend(
                _evidence_status_compatibility_errors(
                    "source strategy decision gate",
                    ClaimStatus.UNRESOLVED_AMBIGUITY,
                    gate.evidence_ids,
                    evidence_by_id,
                    gate.description,
                    gate.classification,
                )
            )
        if compatibility_errors:
            raise ValueError("; ".join(sorted(set(compatibility_errors))))
        return self


class SkillEvaluationScenario(StrictBaseModel):
    """Synthetic scenario evaluation input for deterministic skill harness tests."""

    scenario_id: CanonicalId
    synthetic: Literal[True]
    expected_source_strategy: SourceStrategy | None = None
    requires_user_decision: bool = False
    expected_failure_classification: FailureClassification | None = None
    expected_checkpoint_ids: tuple[CanonicalId, ...] = ()
    expected_embedding_decision: bool = False
    expected_next_mode: RecommendedNextMode | None = None


class ScenarioInspectionResult(StrictBaseModel):
    """Grounded synthetic fixture observations produced only by production inspectors."""

    schema_version: Literal["1.0.0"]
    scenario_id: CanonicalId
    repository_inventory: dict[str, Any]
    packaging_evidence: dict[str, Any]
    dependency_evidence: dict[str, Any]
    import_evidence: dict[str, Any]
    model_candidates: dict[str, Any]
    output_candidates: dict[str, Any]
    checkpoint_candidates: dict[str, Any]
    source_strategy_assessment: dict[str, Any]
    environment_candidates: EnvironmentCandidateGenerationResult
    inspection_warnings: tuple[str, ...] = ()


def _collect_evidence_references(value: Any) -> list[str]:
    references: list[str] = []
    if isinstance(value, EvidenceBackedClaim):
        references.extend(value.evidence_ids)
    elif isinstance(value, EnvironmentCandidateGenerationResult):
        for child in (
            value.dependency_records,
            value.candidates,
            value.source_strategy_context,
            value.decision_gates,
        ):
            references.extend(_collect_evidence_references(child))
    elif isinstance(
        value,
        (
            VariantCandidate,
            CheckpointCandidate,
            SourceStrategyCandidate,
            EmbeddingCandidate,
            OpenQuestion,
            DecisionRecord,
            DependencyEvidenceRecord,
            EnvironmentCandidate,
        ),
    ):
        references.extend(value.evidence_ids if hasattr(value, "evidence_ids") else ())
        references.extend(
            (value.evidence_id,) if isinstance(value, DependencyEvidenceRecord) else ()
        )
        references.extend(
            value.expected_compatibility_evidence if isinstance(value, EnvironmentCandidate) else ()
        )
    elif isinstance(value, StrictBaseModel):
        for child in value.__dict__.values():
            references.extend(_collect_evidence_references(child))
    elif isinstance(value, dict):
        for child in value.values():
            references.extend(_collect_evidence_references(child))
    elif isinstance(value, list | tuple):
        for child in value:
            references.extend(_collect_evidence_references(child))
    return references


def _iter_claims(value: Any) -> list[EvidenceBackedClaim]:
    claims: list[EvidenceBackedClaim] = []
    if isinstance(value, EvidenceBackedClaim):
        claims.append(value)
    elif isinstance(value, StrictBaseModel):
        for child in value.__dict__.values():
            claims.extend(_iter_claims(child))
    elif isinstance(value, list | tuple):
        for child in value:
            claims.extend(_iter_claims(child))
    elif isinstance(value, dict):
        for child in value.values():
            claims.extend(_iter_claims(child))
    return claims


def _validate_candidate_evidence(
    status: ClaimStatus,
    evidence_ids: tuple[str, ...],
    unresolved_reason: str | None,
) -> None:
    if (
        status
        in {
            ClaimStatus.VERIFIED_UPSTREAM_FACT,
            ClaimStatus.LOCALLY_OBSERVED_BEHAVIOR,
            ClaimStatus.REASONED_INFERENCE,
            ClaimStatus.USER_PROVIDED_DECISION,
        }
        and not evidence_ids
    ):
        raise ValueError(f"{status.value} candidates require evidence references")
    if status in {ClaimStatus.UNRESOLVED_AMBIGUITY, ClaimStatus.UNSUPPORTED_CLAIM}:
        if not evidence_ids and not unresolved_reason:
            raise ValueError(
                f"{status.value} candidates require unresolved_reason when unevidenced"
            )


def _duplicate_ids(*groups: tuple[str, list[str]]) -> list[str]:
    errors: list[str] = []
    for label, values in groups:
        if len(values) != len(set(values)):
            errors.append(f"{label} IDs must be unique")
    return errors


def _evidence_compatibility_errors(
    claim: EvidenceBackedClaim,
    evidence_by_id: dict[str, EvidenceItem],
) -> list[str]:
    return _evidence_status_compatibility_errors(
        "claim",
        claim.status,
        claim.evidence_ids,
        evidence_by_id,
        claim.rationale,
    )


def _evidence_status_compatibility_errors(
    label: str,
    status: ClaimStatus,
    evidence_ids: tuple[str, ...],
    evidence_by_id: dict[str, EvidenceItem],
    rationale: str | None,
    gate_classification: OpenQuestionClassification | None = None,
) -> list[str]:
    errors: list[str] = []
    unresolved = sorted(set(evidence_ids) - set(evidence_by_id))
    if unresolved:
        return [f"evidence references are unresolved: {unresolved}"]
    cited = [evidence_by_id[evidence_id] for evidence_id in evidence_ids]
    if status == ClaimStatus.VERIFIED_UPSTREAM_FACT:
        incompatible = [
            item.evidence_id
            for item in cited
            if item.claim_status != ClaimStatus.VERIFIED_UPSTREAM_FACT
            or item.kind not in AUTHORITATIVE_UPSTREAM_EVIDENCE_KINDS
            or _is_generated_project_evidence_path(item.source_file)
        ]
        if incompatible:
            errors.append(f"verified {label}s cite incompatible evidence: {sorted(incompatible)}")
    elif status == ClaimStatus.LOCALLY_OBSERVED_BEHAVIOR:
        if any(
            item.claim_status != ClaimStatus.LOCALLY_OBSERVED_BEHAVIOR
            or item.kind
            not in {
                EvidenceItemKind.SOURCE_FILE,
                EvidenceItemKind.SOURCE_LINE_OR_SYMBOL,
                EvidenceItemKind.PACKAGE_METADATA,
                EvidenceItemKind.CONFIGURATION_FILE,
                EvidenceItemKind.RUNTIME_OBSERVATION,
            }
            for item in cited
        ):
            errors.append(f"locally observed {label}s must cite local or runtime observations")
    elif status == ClaimStatus.USER_PROVIDED_DECISION:
        if any(
            item.kind != EvidenceItemKind.USER_DECISION
            or item.claim_status != ClaimStatus.USER_PROVIDED_DECISION
            for item in cited
        ):
            errors.append(f"user-provided {label}s must cite user_decision evidence")
    elif status == ClaimStatus.REASONED_INFERENCE:
        if not rationale or not rationale.strip():
            errors.append(f"reasoned inference {label}s require rationale")
        if not any(
            item.claim_status
            in {
                ClaimStatus.VERIFIED_UPSTREAM_FACT,
                ClaimStatus.LOCALLY_OBSERVED_BEHAVIOR,
            }
            and item.kind != EvidenceItemKind.AGENT_INFERENCE
            for item in cited
        ):
            errors.append(
                f"reasoned inference {label}s require factual or locally observed evidence"
            )
        if all(
            item.claim_status in {ClaimStatus.UNSUPPORTED_CLAIM, ClaimStatus.UNRESOLVED_AMBIGUITY}
            for item in cited
        ):
            errors.append(
                f"reasoned inference {label}s cannot cite only unsupported or unresolved evidence"
            )
    elif status == ClaimStatus.UNRESOLVED_AMBIGUITY:
        if cited and all(item.claim_status == ClaimStatus.UNSUPPORTED_CLAIM for item in cited):
            if gate_classification != OpenQuestionClassification.UNSUPPORTED_UPSTREAM_CLAIM:
                errors.append(
                    f"unresolved {label}s cannot cite only unsupported evidence unless "
                    "classified unsupported_upstream_claim"
                )
    return errors


def _validate_version_constraint(label: str, version: str | None, constraint: str | None) -> None:
    specifier = SpecifierSet("")
    if constraint and constraint != "unconstrained":
        try:
            specifier = SpecifierSet(constraint)
        except InvalidSpecifier as exc:
            raise ValueError(f"{label} has invalid constraint: {constraint}") from exc
    if version:
        try:
            parsed = Version(version)
        except InvalidVersion as exc:
            raise ValueError(f"{label} has invalid version: {version}") from exc
        if constraint and constraint != "unconstrained" and parsed not in specifier:
            raise ValueError(f"{label} version {version} is outside constraint {constraint}")


def _is_exact_lowercase_git_sha(value: str | None) -> bool:
    return bool(value and re.fullmatch(r"[0-9a-f]{40}", value))


def _valid_environment_report_reference(value: str, card_id: str, fingerprint: str | None) -> bool:
    if re.fullmatch(rf"verification_reports/{re.escape(card_id)}/[^/]+\.json", value):
        return True
    if fingerprint and re.fullmatch(
        rf"reports/environments/{re.escape(card_id)}/{re.escape(fingerprint)}/[^/]+\.json",
        value,
    ):
        return True
    return False


def _has_exact_package_identity_evidence(
    selected: EnvironmentCandidate,
    evidence_by_id: dict[str, EvidenceItem],
    expected_evidence_ids: set[str],
    artifact_paths: set[str],
) -> bool:
    environment_identity_paths = {
        path
        for path in artifact_paths
        if re.fullmatch(
            r"environments/[^/]+/(?:environment\.json|pyproject\.toml|uv\.lock)",
            path,
        )
    }
    for evidence_id, item in evidence_by_id.items():
        if (
            evidence_id not in expected_evidence_ids
            or item.package_name != selected.source_package_name
            or item.package_version != selected.source_package_version
        ):
            continue
        if _is_upstream_package_identity_evidence(item):
            return True
        if (
            item.kind
            in {
                EvidenceItemKind.PACKAGE_METADATA,
                EvidenceItemKind.CONFIGURATION_FILE,
            }
            and item.claim_status == ClaimStatus.LOCALLY_OBSERVED_BEHAVIOR
            and item.source_file in environment_identity_paths
        ):
            return True
    return False


def _is_upstream_package_identity_evidence(item: EvidenceItem) -> bool:
    if (
        item.kind != EvidenceItemKind.PACKAGE_METADATA
        or item.claim_status != ClaimStatus.VERIFIED_UPSTREAM_FACT
    ):
        return False
    if item.source_file is None or _is_generated_project_evidence_path(item.source_file):
        return False
    return item.source_file.rsplit("/", 1)[-1] in UPSTREAM_PACKAGE_METADATA_FILENAMES


def _is_generated_project_evidence_path(source_file: str | None) -> bool:
    if source_file is None:
        return False
    return source_file.split("/", 1)[0] in GENERATED_PROJECT_EVIDENCE_PATH_PREFIXES


def ensure_onboarding_evidence_path(value: str | None) -> str | None:
    if value is None:
        return None
    if not value:
        raise ValueError("evidence path must not be empty")
    if value.startswith("/") or "\\" in value:
        raise ValueError("evidence path must be repository-relative POSIX")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("evidence path must not contain empty, '.', or '..' segments")
    if parts[0] in ONBOARDING_FORBIDDEN_PATH_PREFIXES:
        raise ValueError("evidence path uses an ignored or runtime-only prefix")
    if re.fullmatch(ONBOARDING_EVIDENCE_PATH_PATTERN, value) is None:
        raise ValueError("evidence path contains unsupported characters")
    return value


def _confidence_summary(report: AnalysisReport) -> ConfidenceSummary:
    statuses: list[ClaimStatus] = []
    statuses.extend(item.claim_status for item in report.evidence_items)
    statuses.extend(claim.status for claim in _iter_claims(report))
    statuses.extend(item.status for item in report.variants)
    statuses.extend(item.status for item in report.checkpoint_candidates)
    statuses.extend(item.status for item in report.source_strategy_candidates)
    statuses.extend(item.status for item in report.embedding_candidates)
    statuses.extend(item.status for item in report.decisions)
    statuses.extend(
        ClaimStatus.UNRESOLVED_AMBIGUITY
        for item in report.open_questions
        if item.classification
        in {
            OpenQuestionClassification.NEEDS_MORE_EVIDENCE,
            OpenQuestionClassification.NEEDS_RUNTIME_PROBE,
            OpenQuestionClassification.NEEDS_ENVIRONMENT_RESOLUTION,
            OpenQuestionClassification.NEEDS_USER_DECISION,
        }
    )
    statuses.extend(
        ClaimStatus.UNSUPPORTED_CLAIM
        for item in report.open_questions
        if item.classification == OpenQuestionClassification.UNSUPPORTED_UPSTREAM_CLAIM
    )
    return ConfidenceSummary(
        verified_fact_count=statuses.count(ClaimStatus.VERIFIED_UPSTREAM_FACT),
        locally_observed_count=statuses.count(ClaimStatus.LOCALLY_OBSERVED_BEHAVIOR),
        inference_count=statuses.count(ClaimStatus.REASONED_INFERENCE),
        unresolved_count=statuses.count(ClaimStatus.UNRESOLVED_AMBIGUITY),
        unsupported_claim_count=statuses.count(ClaimStatus.UNSUPPORTED_CLAIM),
    )
