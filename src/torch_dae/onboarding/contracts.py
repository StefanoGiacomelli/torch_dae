"""Strict onboarding artifact contracts."""

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
    REPO_RELATIVE_OR_DOTFILE_PATTERN,
    REPO_RELATIVE_PATTERN,
    SHA256_PATTERN,
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
    """Supported implementation-source decisions.

    Values distinguish an official package, a pinned official repository, a minimal vendored
    adaptation, a proven external implementation, and an unsupported or non-equivalent source.
    Selection requires explicit evidence; enum ordering does not imply preference.
    """

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
    """Stable classifications for observed environment-resolution failures.

    These values preserve the cause of a failed or blocked trial without relying on raw tool output.
    Diagnostics may accompany a classification, but should remain sanitized and evidence-backed.
    """

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
    SANDBOX_OR_EXECUTION_POLICY = "sandbox_or_execution_policy"
    NETWORK_OR_DNS = "network_or_dns"
    PACKAGE_INDEX = "package_index"
    RATE_LIMIT = "rate_limit"
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


class PublishedChecksumAlgorithm(StrEnum):
    """Algorithms accepted for checkpoint-host checksum metadata."""

    MD5 = "md5"
    SHA1 = "sha1"
    SHA256 = "sha256"
    SHA512 = "sha512"
    BLAKE2B = "blake2b"
    OTHER = "other"
    UNKNOWN = "unknown"


ONBOARDING_EVIDENCE_PATH_PATTERN = r"[A-Za-z0-9.][A-Za-z0-9._/-]*"


class EvidenceItem(StrictBaseModel):
    """Record one atomic source of onboarding evidence.

    Attributes
    ----------
    evidence_id
        Canonical identifier referenced by report claims and candidates.
    kind
        Evidence source category.
    claim_status
        Whether the item is reported, observed, inferred, unresolved, or unsupported.
    description
        Concise statement of what the source establishes.
    source_file
        Optional safe repository-relative upstream path.
    source_line_or_symbol
        Optional stable line range or program symbol.
    url
        Optional authoritative external source.
    revision
        Optional immutable source revision.
    rationale
        Required justification where the evidence status represents inference.

    Raises
    ------
    pydantic.ValidationError
        If paths, versions, provenance combinations, or inference rationale are invalid.
    """

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
    """A claim that cannot silently promote inference to verified fact.

    Empty scope tuples mean that the claim applies across the repository-level report. Nonempty
    scopes are resolved by :class:`AnalysisReport` against declared variants and checkpoints.
    """

    statement: str
    status: ClaimStatus
    evidence_ids: tuple[CanonicalId, ...] = ()
    variant_ids: tuple[CanonicalId, ...] = ()
    checkpoint_ids: tuple[CanonicalId, ...] = ()
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
        _validate_unique_scope_ids(self.variant_ids, self.checkpoint_ids)
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


class PublishedChecksum(StrictBaseModel):
    """Host-published checkpoint checksum that has not been locally verified.

    The record intentionally cannot represent local payload verification. A later SHA-256
    acquisition check remains a separate lifecycle operation even when another published digest is
    available.
    """

    algorithm: PublishedChecksumAlgorithm
    digest: str
    evidence_id: CanonicalId
    verification_state: Literal["published_not_locally_verified"] = "published_not_locally_verified"
    provenance_note: str | None = None

    @model_validator(mode="after")
    def validate_digest(self) -> PublishedChecksum:
        lengths = {
            PublishedChecksumAlgorithm.MD5: 32,
            PublishedChecksumAlgorithm.SHA1: 40,
            PublishedChecksumAlgorithm.SHA256: 64,
            PublishedChecksumAlgorithm.SHA512: 128,
            PublishedChecksumAlgorithm.BLAKE2B: 128,
        }
        expected = lengths.get(self.algorithm)
        if expected is not None and re.fullmatch(rf"[0-9a-f]{{{expected}}}", self.digest) is None:
            raise ValueError(
                f"{self.algorithm.value} digest must contain exactly {expected} lowercase hex "
                "characters"
            )
        if (
            self.algorithm
            in {
                PublishedChecksumAlgorithm.OTHER,
                PublishedChecksumAlgorithm.UNKNOWN,
            }
            and not self.provenance_note
        ):
            raise ValueError("other or unknown checksum algorithms require provenance_note")
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
    published_checksums: tuple[PublishedChecksum, ...] = ()
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
        checksum_identities = [
            (checksum.algorithm, checksum.digest) for checksum in self.published_checksums
        ]
        if len(checksum_identities) != len(set(checksum_identities)):
            raise ValueError("published checksum algorithm/digest pairs must be unique")
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
    """Candidate embedding tensor whose semantics require evidence.

    Empty variant and checkpoint scopes mean report-wide applicability.
    """

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
    variant_ids: tuple[CanonicalId, ...] = ()
    checkpoint_ids: tuple[CanonicalId, ...] = ()
    requires_user_decision: bool = False
    unresolved_reason: str | None = None

    @model_validator(mode="after")
    def candidate_requires_evidence_or_reason(self) -> EmbeddingCandidate:
        _validate_candidate_evidence(self.status, self.evidence_ids, self.unresolved_reason)
        _validate_unique_scope_ids(self.variant_ids, self.checkpoint_ids)
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
    """Represent the evidence-grounded output of static analysis mode.

    The report records repository and scientific identity, architecture and runtime claims,
    variants, checkpoint and embedding candidates, dependencies, source strategies, open questions,
    decisions, and confidence counts. All claim references must resolve to a unique evidence item.

    Notes
    -----
    An analysis report is not proof of runtime compatibility. Creating or validating it does not
    execute upstream code, install dependencies, or download checkpoints.

    Raises
    ------
    pydantic.ValidationError
        If identities are duplicated, evidence references are unresolved, confidence counts are
        inconsistent, or a claim overstates its evidence.
    """

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

        variant_ids = {item.variant_id for item in self.variants}
        checkpoint_ids = {item.checkpoint_id for item in self.checkpoint_candidates}
        scope_errors: list[str] = []
        for label, item in [("claim", claim) for claim in _iter_claims(self)] + [
            ("embedding", embedding) for embedding in self.embedding_candidates
        ]:
            missing_variants = sorted(set(item.variant_ids) - variant_ids)
            missing_checkpoints = sorted(set(item.checkpoint_ids) - checkpoint_ids)
            if missing_variants:
                scope_errors.append(f"{label} variant scope is unresolved: {missing_variants}")
            if missing_checkpoints:
                scope_errors.append(
                    f"{label} checkpoint scope is unresolved: {missing_checkpoints}"
                )
        if scope_errors:
            raise ValueError("; ".join(scope_errors))

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
            for checksum in checkpoint.published_checksums:
                compatibility_errors.extend(
                    _evidence_status_compatibility_errors(
                        "published checksum",
                        ClaimStatus.VERIFIED_UPSTREAM_FACT,
                        (checksum.evidence_id,),
                        evidence_by_id,
                        checksum.provenance_note,
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
            raise ValueError(
                "profile mode is reserved and cannot be a onboarding workflow next step"
            )
        return self


class EnvironmentCandidate(StrictBaseModel):
    """Describe one evidence-motivated environment compatibility candidate.

    Attributes
    ----------
    candidate_id
        Stable identifier within an environment-resolution report.
    reason_for_selection
        Evidence-backed explanation for trying this candidate.
    python_version
        Optional exact interpreter version proposed for the isolated environment.
    installation_strategy
        Selected source installation strategy.
    expected_compatibility_evidence
        Evidence identifiers supporting the candidate.
    trial_status
        Observed state; defaults to ``not_attempted``.
    failure_classification
        Normalized cause when a trial fails or is blocked.
    uncertainty
        Known remaining compatibility uncertainty.

    Warnings
    --------
    Candidate creation is predictive and has no execution side effects. Only a successful controlled
    trial and verification may establish compatibility.
    """

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
    """Represent the complete output of environment-resolution mode.

    Attributes
    ----------
    ordered_candidates
        Compatibility candidates in the evidence-supported trial order.
    attempted_candidates
        Candidate identifiers actually tried in isolated runtime state.
    selected_candidate_id
        Candidate selected after observed validation, if any.
    environment_artifact_paths
        Repository-relative committed inputs created for a resolved environment.
    environment_fingerprint
        SHA-256 identity of a successfully materialized environment.
    environment_verification_succeeded
        Whether the committed verification procedure passed.

    Raises
    ------
    pydantic.ValidationError
        If candidate or evidence references are unresolved, outcomes contradict their diagnostics,
        or lifecycle advancement lacks successful materialization and verification.
    """

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
    environment_artifact_paths: tuple[
        Annotated[str, Field(pattern=REPO_RELATIVE_PATTERN)], ...
    ] = ()
    environment_materialization_succeeded: bool = False
    environment_verification_succeeded: bool = False
    environment_fingerprint: Annotated[str | None, Field(pattern=r"^[0-9a-f]{64}$")] = None
    verification_report_or_diagnostic_reference: Annotated[
        str | None, Field(pattern=REPO_RELATIVE_PATTERN)
    ] = None
    next_lifecycle_status: Literal[ModelCardLifecycle.ENVIRONMENT_RESOLVED] | None = None

    @field_validator("environment_artifact_paths")
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
                    set(self.environment_artifact_paths),
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
                for item in self.environment_artifact_paths
                if item.startswith("environments/")
            }
            actual_dirs = {
                item.rsplit("/", 1)[0]
                for item in self.environment_artifact_paths
                if item.startswith("environments/")
            }
            if actual_names != required_names or len(self.environment_artifact_paths) != 5:
                raise ValueError(
                    "environment_resolved requires all five environment artifact paths"
                )
            if len(actual_dirs) != 1:
                raise ValueError("environment_resolved requires one environment artifact directory")
            artifact_card_id = next(iter(actual_dirs)).split("/", 1)[1]
            if not self.environment_materialization_succeeded:
                raise ValueError(
                    "environment_resolved requires environment materialization success"
                )
            if not self.environment_verification_succeeded:
                raise ValueError("environment_resolved requires environment verification success")
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


class OnboardingPhase(StrEnum):
    """Committed onboarding phases, including checkpoint-specific verification."""

    ANALYZE = "analyze"
    RESOLVE_ENVIRONMENT = "resolve-environment"
    INTEGRATE = "integrate"
    VERIFY = "verify"
    CARD = "card"


class WorkflowStatus(StrEnum):
    """Lifecycle of one cross-conversation onboarding workflow."""

    ACTIVE = "active"
    COMPLETED = "completed"


class HandoffStatus(StrEnum):
    """Review status of one phase handoff."""

    DRAFT = "draft"
    ACCEPTED = "accepted"
    SUPERSEDED = "superseded"


class ArtifactOriginPhase(StrEnum):
    """Origin recorded for an artifact reference."""

    ANALYZE = "analyze"
    RESOLVE_ENVIRONMENT = "resolve-environment"
    INTEGRATE = "integrate"
    VERIFY = "verify"
    CARD = "card"
    EXTERNAL = "external"


class AcceptedPhaseReference(StrictBaseModel):
    """Canonical accepted handoff path for one workflow phase."""

    phase: OnboardingPhase
    handoff_path: Annotated[str, Field(pattern=REPO_RELATIVE_PATTERN)]

    @field_validator("handoff_path")
    @classmethod
    def handoff_path_repository_relative(cls, value: str) -> str:
        return ensure_repository_relative(value) or value


class WorkflowRecord(StrictBaseModel):
    """Committed identity and accepted-phase index for an onboarding workflow."""

    schema_version: Literal["1.0.0"]
    workflow_id: CanonicalId
    model_family: str
    target_variant_ids: tuple[CanonicalId, ...]
    target_checkpoint_ids: tuple[CanonicalId, ...]
    target_card_ids: tuple[CanonicalId, ...] = ()
    created_repository_commit: Annotated[str, Field(pattern=GIT_REVISION_PATTERN)]
    current_accepted_phase: OnboardingPhase | None = None
    accepted_phase_paths: tuple[AcceptedPhaseReference, ...] = ()
    status: WorkflowStatus

    @model_validator(mode="after")
    def validate_phase_index(self) -> WorkflowRecord:
        phases = [item.phase for item in self.accepted_phase_paths]
        if len(phases) != len(set(phases)):
            raise ValueError("accepted workflow phases must be unique")
        if self.current_accepted_phase is not None and self.current_accepted_phase not in phases:
            raise ValueError("current_accepted_phase must have an accepted phase path")
        if len(self.target_variant_ids) != len(set(self.target_variant_ids)):
            raise ValueError("target variant IDs must be unique")
        if len(self.target_checkpoint_ids) != len(set(self.target_checkpoint_ids)):
            raise ValueError("target checkpoint IDs must be unique")
        if len(self.target_card_ids) != len(set(self.target_card_ids)):
            raise ValueError("target card IDs must be unique")
        expected_prefix = f"onboarding_reports/{self.workflow_id}/"
        for reference in self.accepted_phase_paths:
            expected = f"{expected_prefix}{reference.phase.value}/handoff.json"
            if reference.handoff_path != expected:
                raise ValueError(f"accepted handoff path must be {expected}")
        return self


class HandoffArtifactReference(StrictBaseModel):
    """Hash-addressed local or external artifact used by a phase handoff."""

    path: Annotated[str | None, Field(pattern=REPO_RELATIVE_OR_DOTFILE_PATTERN)] = None
    sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    media_type: Annotated[str, Field(pattern=r"^[a-z0-9.+-]+/[a-z0-9.+-]+$")]
    originating_phase: ArtifactOriginPhase
    canonical_role: CanonicalId | None = None
    external_label: CanonicalId | None = None

    @field_validator("path")
    @classmethod
    def path_repository_relative(cls, value: str | None) -> str | None:
        return ensure_repository_relative(value)

    @model_validator(mode="after")
    def validate_location(self) -> HandoffArtifactReference:
        if (self.path is None) == (self.external_label is None):
            raise ValueError("artifact reference requires exactly one of path or external_label")
        if self.path is None and self.originating_phase != ArtifactOriginPhase.EXTERNAL:
            raise ValueError("digest-only artifacts must have external origin")
        if self.path is not None and self.external_label is not None:
            raise ValueError("local artifacts must not carry an external label")
        return self


class ArtifactSupersession(StrictBaseModel):
    """One explicit transition between accepted repository artifact states."""

    path: Annotated[str, Field(pattern=REPO_RELATIVE_PATTERN)]
    prior_originating_phase: OnboardingPhase
    prior_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    new_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    reason: str
    prior_handoff_sha256: Annotated[str | None, Field(pattern=SHA256_PATTERN)] = None

    @field_validator("path")
    @classmethod
    def path_repository_relative(cls, value: str) -> str:
        return ensure_repository_relative(value) or value

    @field_validator("reason")
    @classmethod
    def reason_has_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("artifact supersession reason must contain non-whitespace text")
        return value

    @model_validator(mode="after")
    def hash_transition_changes_content(self) -> ArtifactSupersession:
        if self.prior_sha256 == self.new_sha256:
            raise ValueError("artifact supersession must change the artifact SHA-256")
        return self


class ConsumedUserDecision(StrictBaseModel):
    """One task-specific user decision consumed by a phase."""

    decision_id: CanonicalId
    decision: str
    selected_option: str


class CarriedUnresolvedItem(StrictBaseModel):
    """One unresolved item explicitly carried into a later phase."""

    item_id: CanonicalId
    description: str
    source_artifact_path: Annotated[str | None, Field(pattern=REPO_RELATIVE_PATTERN)] = None

    @field_validator("source_artifact_path")
    @classmethod
    def source_path_repository_relative(cls, value: str | None) -> str | None:
        return ensure_repository_relative(value)


class HandoffValidationSummary(StrictBaseModel):
    """Deterministic validation outcome required for handoff acceptance."""

    passed: bool
    checks: tuple[str, ...]
    errors: tuple[str, ...] = ()

    @model_validator(mode="after")
    def passed_has_no_errors(self) -> HandoffValidationSummary:
        if self.passed and self.errors:
            raise ValueError("passed validation summary must not contain errors")
        if not self.passed and not self.errors:
            raise ValueError("failed validation summary requires errors")
        return self


class PhaseHandoffManifest(StrictBaseModel):
    """Committed, hash-addressed handoff from one onboarding phase."""

    schema_version: Literal["1.0.0"]
    workflow_id: CanonicalId
    phase: OnboardingPhase
    handoff_status: HandoffStatus
    repository_commit: Annotated[str, Field(pattern=GIT_REVISION_PATTERN)]
    project_spec_sha256: Annotated[
        str,
        Field(
            pattern=SHA256_PATTERN,
            description=(
                "SHA-256 of project_spec.md when this phase candidate was prepared; after "
                "acceptance this value is immutable historical provenance."
            ),
        ),
    ]
    canonical_skill_fingerprint: Annotated[
        str,
        Field(
            pattern=SHA256_PATTERN,
            description=(
                "Canonical onboarding-skill fingerprint when this phase candidate was prepared; "
                "after acceptance this value is immutable historical provenance."
            ),
        ),
    ]
    input_artifacts: tuple[HandoffArtifactReference, ...] = ()
    output_artifacts: tuple[HandoffArtifactReference, ...]
    artifact_supersessions: tuple[ArtifactSupersession, ...] = ()
    target_variant_ids: tuple[CanonicalId, ...]
    target_checkpoint_ids: tuple[CanonicalId, ...]
    target_card_ids: tuple[CanonicalId, ...] = ()
    user_decisions_consumed: tuple[ConsumedUserDecision, ...] = ()
    unresolved_items_carried_forward: tuple[CarriedUnresolvedItem, ...] = ()
    validation_summary: HandoffValidationSummary
    allowed_next_modes: tuple[RecommendedNextMode, ...]
    lifecycle_promotion: ModelCardLifecycle | None = None
    superseded_handoff_sha256: Annotated[str | None, Field(pattern=SHA256_PATTERN)] = None

    @model_validator(mode="after")
    def validate_handoff_state(self) -> PhaseHandoffManifest:
        if self.handoff_status == HandoffStatus.ACCEPTED and not self.validation_summary.passed:
            raise ValueError("accepted handoffs require passed validation")
        if not self.output_artifacts:
            raise ValueError("phase handoffs require output artifacts")
        if any(item.canonical_role is None for item in self.output_artifacts):
            raise ValueError("output artifacts require canonical_role")
        output_paths = [item.path for item in self.output_artifacts if item.path is not None]
        if len(output_paths) != len(set(output_paths)):
            raise ValueError("output artifact paths must be unique")
        supersession_paths = [item.path for item in self.artifact_supersessions]
        if len(supersession_paths) != len(set(supersession_paths)):
            raise ValueError("artifact supersession paths must be unique within a handoff")
        if len(self.allowed_next_modes) != len(set(self.allowed_next_modes)):
            raise ValueError("allowed next modes must be unique")
        for label, values in (
            ("target variant", self.target_variant_ids),
            ("target checkpoint", self.target_checkpoint_ids),
            ("target card", self.target_card_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} IDs must be unique")
        expected_prefix = f"onboarding_reports/{self.workflow_id}/"
        phase_prefix = f"{expected_prefix}{self.phase.value}/"
        for artifact in (*self.input_artifacts, *self.output_artifacts):
            if artifact.path and artifact.path.startswith("onboarding_reports/"):
                if not artifact.path.startswith(expected_prefix):
                    raise ValueError("handoff artifacts may not mix onboarding workflows")
        allowed_external_output_roots = {
            OnboardingPhase.ANALYZE: (),
            OnboardingPhase.RESOLVE_ENVIRONMENT: ("environments/",),
            OnboardingPhase.INTEGRATE: (
                "docs/",
                "environments/",
                "model_cards/",
                "src/",
                "tests/",
            ),
            OnboardingPhase.VERIFY: ("verification_reports/",),
            OnboardingPhase.CARD: ("model_cards/",),
        }[self.phase]
        allowed_exact_output_paths = {
            OnboardingPhase.ANALYZE: (),
            OnboardingPhase.RESOLVE_ENVIRONMENT: (),
            OnboardingPhase.INTEGRATE: (
                ".gitattributes",
                "CHANGELOG.md",
                "README.md",
                "pyproject.toml",
                "schemas/environment-dependency-closure-result.schema.json",
                "schemas/environment-materialization-result.schema.json",
                "schemas/environment-verification-result.schema.json",
                "schemas/phase-handoff.schema.json",
                "scripts/check_worktree_patch.py",
                "scripts/generate_schemas.py",
                "scripts/validate_repository.py",
                "skills/audio-model-onboarding/SKILL.md",
                "skills/audio-model-onboarding/references/environment-resolution.md",
                "skills/audio-model-onboarding/references/failure-classification.md",
                "skills/audio-model-onboarding/references/integration-planning.md",
                "skills/audio-model-onboarding/templates/agent-request.md",
                "skills/audio-model-onboarding/templates/agent-response.md",
            ),
            OnboardingPhase.VERIFY: (),
            OnboardingPhase.CARD: (),
        }[self.phase]
        for artifact in self.output_artifacts:
            if artifact.path is None or not (
                artifact.path.startswith(phase_prefix)
                or artifact.path.startswith(allowed_external_output_roots)
                or artifact.path in allowed_exact_output_paths
            ):
                raise ValueError("output artifacts must be canonical for the handoff phase")
        if not any(
            artifact.path and artifact.path.startswith(phase_prefix)
            for artifact in self.output_artifacts
        ):
            raise ValueError("handoff requires at least one phase-local canonical output")
        phase_order = {
            OnboardingPhase.ANALYZE: 0,
            OnboardingPhase.RESOLVE_ENVIRONMENT: 1,
            OnboardingPhase.INTEGRATE: 2,
            OnboardingPhase.VERIFY: 3,
            OnboardingPhase.CARD: 4,
        }
        output_by_path = {
            artifact.path: artifact
            for artifact in self.output_artifacts
            if artifact.path is not None
        }
        protected_prefixes = (
            ".git/",
            ".torch-dae/",
            ".venv/",
            "checkpoints/",
            "onboarding_reports/",
            "reports/",
            "schemas/",
            "verification_reports/",
        )
        protected_names = {
            ".env",
            "credentials.json",
            "project_spec.md",
            "secrets.json",
        }
        for supersession in self.artifact_supersessions:
            if phase_order[supersession.prior_originating_phase] >= phase_order[self.phase]:
                raise ValueError("artifact supersession must move to a legal later workflow phase")
            path_parts = set(supersession.path.split("/"))
            credential_parts = {".env", "credentials", "credentials.json", "secrets.json"}
            if (
                supersession.path in protected_names
                or supersession.path.startswith(protected_prefixes)
                or not path_parts.isdisjoint(credential_parts)
            ):
                raise ValueError("protected canonical or runtime artifacts cannot be superseded")
            output = output_by_path.get(supersession.path)
            if output is None:
                raise ValueError("superseded artifact must be declared as a current phase output")
            if output.originating_phase.value != self.phase.value:
                raise ValueError(
                    "superseding output must originate in the containing handoff phase"
                )
            if output.sha256 != supersession.new_sha256:
                raise ValueError("superseding output hash must match new_sha256")
        return self


class ManagedRunManifest(StrictBaseModel):
    """Ignored runtime manifest for one managed onboarding phase execution."""

    schema_version: Literal["1.0.0"]
    run_id: CanonicalId
    workflow_id: CanonicalId
    phase: OnboardingPhase
    started_at: datetime
    repository_commit: Annotated[str, Field(pattern=GIT_REVISION_PATTERN)]
    created_paths: tuple[str, ...]
    reused_paths: tuple[str, ...] = ()
    external_paths: tuple[str, ...] = ()
    retained_paths: tuple[str, ...] = ()
    retained_reasons: dict[str, str] = Field(default_factory=dict)
    cleanup_result: dict[str, Any] | None = None


class CleanupPathRecord(StrictBaseModel):
    """One managed or external path retained by cleanup."""

    path: str
    category: str
    reason: str
    status: Literal[
        "retained-existing-file",
        "retained-existing-directory",
        "retained-symlink",
        "missing",
    ]
    sha256: Annotated[str | None, Field(pattern=SHA256_PATTERN)] = None

    @model_validator(mode="after")
    def hash_only_existing_files(self) -> CleanupPathRecord:
        if self.status == "retained-existing-file" and self.sha256 is None:
            raise ValueError("retained existing files require SHA-256")
        if self.status != "retained-existing-file" and self.sha256 is not None:
            raise ValueError("only retained existing files may carry SHA-256")
        return self


class CleanupRetentionConflict(StrictBaseModel):
    """A retained path that would be removed by a planned deletion root."""

    retained_path: str
    deletion_root: str
    reason: str
    remediation: str


class CleanupExternalProtectionConflict(StrictBaseModel):
    """An existing external output that overlaps a planned deletion root."""

    supplied_external_path: str
    resolved_path: str
    deletion_root: str
    status: Literal[
        "retained-existing-file",
        "retained-existing-directory",
        "retained-symlink",
        "missing",
    ]
    reason: str
    remediation: str


class CleanupConsumedRunManifest(StrictBaseModel):
    """A finalized run manifest embedded in a durable cleanup receipt."""

    path: str
    sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    manifest: ManagedRunManifest


class CleanupReceipt(StrictBaseModel):
    """Durable ignored-runtime receipt for one cleanup plan or execution."""

    schema_version: Literal["1.0.0"]
    workflow_id: CanonicalId
    cleanup_operation_id: CanonicalId
    time: datetime
    mode: Literal["dry-run", "execute"]
    run_manifests_consumed: tuple[CleanupConsumedRunManifest, ...]
    planned_paths: tuple[str, ...]
    removed_paths: tuple[str, ...]
    retention_conflicts: tuple[CleanupRetentionConflict, ...] = ()
    external_protection_conflicts: tuple[CleanupExternalProtectionConflict, ...] = ()
    retained_managed_paths: tuple[CleanupPathRecord, ...] = ()
    retained_external_paths: tuple[CleanupPathRecord, ...] = ()
    repository_caches_retained: tuple[CleanupPathRecord, ...] = ()
    package_caches_retained: tuple[CleanupPathRecord, ...] = ()
    materialized_environments_retained: tuple[CleanupPathRecord, ...] = ()
    checkpoint_caches_retained: tuple[CleanupPathRecord, ...] = ()
    verified_removed: bool
    errors: tuple[str, ...] = ()

    @model_validator(mode="after")
    def verified_removal_is_truthful(self) -> CleanupReceipt:
        if self.mode == "dry-run" and (self.removed_paths or self.verified_removed):
            raise ValueError("dry-run cleanup cannot remove or verify removal")
        if (
            self.errors or self.retention_conflicts or self.external_protection_conflicts
        ) and self.verified_removed:
            raise ValueError("cleanup with errors or protection conflicts cannot verify removal")
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
        if isinstance(value, CheckpointCandidate):
            references.extend(checksum.evidence_id for checksum in value.published_checksums)
    elif isinstance(value, PublishedChecksum):
        references.append(value.evidence_id)
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


def _validate_unique_scope_ids(
    variant_ids: tuple[str, ...],
    checkpoint_ids: tuple[str, ...],
) -> None:
    if len(variant_ids) != len(set(variant_ids)):
        raise ValueError("variant scope IDs must be unique")
    if len(checkpoint_ids) != len(set(checkpoint_ids)):
        raise ValueError("checkpoint scope IDs must be unique")


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
