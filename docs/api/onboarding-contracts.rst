Onboarding contracts
====================

These strict contracts preserve provenance and uncertainty. A verified upstream fact must be backed
by an authoritative upstream evidence kind; a local observation cannot silently become an upstream
fact; a reasoned inference requires both evidence references and rationale; unresolved and
unsupported statements stay explicit.

Evidence flow
-------------

.. container:: api-flow

   **static observations** → **evidence items** → **evidence-backed claims** →
   **candidates and decisions** → **analysis report** → **environment candidates** →
   **environment-resolution report**

JSON is canonical machine state. Human-readable Markdown is a deterministic view of that state.
Validation resolves every evidence identifier, rejects duplicates, and checks that status is
compatible with evidence provenance.

Enum values
-----------

.. list-table::
   :header-rows: 1
   :widths: 22 32 46

   * - Enum
     - Values
     - Meaning
   * - ``EvidenceItemKind``
     - ``source_file``, ``source_line_or_symbol``, ``package_metadata``,
       ``configuration_file``, ``official_documentation``, ``paper``
     - Static authoritative-source categories.
   * -
     - ``runtime_observation``, ``agent_inference``, ``user_decision``
     - Local execution evidence, reasoned agent evidence, and explicit user authority.
   * - ``ClaimStatus``
     - ``verified_upstream_fact``, ``locally_observed_behavior``, ``reasoned_inference``,
       ``user_provided_decision``, ``unresolved_ambiguity``, ``unsupported_claim``
     - Exact epistemic status; only the first is an upstream fact.
   * - ``OpenQuestionClassification``
     - ``needs_more_evidence``, ``needs_runtime_probe``, ``needs_environment_resolution``,
       ``needs_user_decision``, ``unsupported_upstream_claim``, ``out_of_scope``
     - Work or authority required to close an ambiguity.
   * - ``SourceStrategy``
     - ``official_package``, ``pinned_official_git_repository``,
       ``minimal_vendored_adaptation``, ``external_pytorch_implementation``,
       ``unsupported_or_non_equivalent_implementation``
     - Evidence-supported implementation-source alternatives; ordering implies no preference.
   * - ``RecommendedNextMode``
     - ``analyze``, ``resolve-environment``, ``integrate``, ``verify``, ``card``, ``profile``
     - Workflow hand-off vocabulary. Analysis reports reject reserved ``profile`` as a next step.
   * - ``CandidateTrialStatus``
     - ``not_attempted``, ``passed``, ``failed``, ``blocked``
     - Observed trial state.
   * - ``DependencyKind``
     - ``requirement``, ``conda``, ``vcs``, ``direct_url``, ``editable``, ``local_path``,
       ``locked``, ``unknown``
     - Static declaration format, not an installation decision.

``FailureClassification`` values are ``python_constraint``, ``dependency_conflict``,
``resolution_failure``, ``removed_api``, ``deprecated_api``, ``binary_or_abi_incompatibility``,
``missing_binary_wheel``, ``torch_torchaudio_mismatch``, ``numpy_compatibility``,
``checkpoint_incompatibility``, ``source_build_failure``, ``import_failure``, ``runtime_failure``,
``platform_incompatibility``, ``access_or_authentication_blocker``, and
``insufficient_evidence``. They normalize observed causes without exposing secrets or treating raw
diagnostics as proof.

Field and invariant matrix
--------------------------

Fields are required unless a ``?`` or default is shown. Canonical IDs use the repository identifier
grammar; upstream evidence paths are safe relative paths and may not point into generated project
artifact locations.

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Contract
     - Fields
     - Main validation
   * - ``EvidenceItem``
     - ID, kind, claim status, description; optional file, symbol, URL, revision, package identity,
       rationale
     - Package name normalized; version valid; inference requires rationale; user decision/kind
       agrees with status.
   * - ``EvidenceBackedClaim``
     - ``statement``, ``status``, ``evidence_ids=()``, ``rationale=None``
     - Positive statuses require evidence; inference requires rationale.
   * - ``ReportSection``
     - ``summary``, ``claims=()``
     - Claims retain their own evidence policy.
   * - ``RepositoryIdentity``
     - repository URL/owner/name/revision, license claims, package, release/maintenance claims,
       official-status claim
     - Duplicated top-level analysis identity fields must agree.
   * - ``VariantCandidate``
     - ID, name, status, evidence IDs, unresolved reason
     - Status requires compatible evidence or explicit unresolved reason.
   * - ``CheckpointCandidate``
     - ID/source type; optional filename, URL, variant, loader, hash, notes, helper/expression;
       evidence IDs, status, unresolved reason
     - Helper-based HTTPS candidates require expression status.
   * - ``SourceStrategyCandidate``
     - strategy, status, rationale, evidence IDs, decision flag, unresolved reason
     - Evidence and ambiguity must agree with status.
   * - ``EmbeddingCandidate``
     - ID, origin, exact semantic kind, shape/batch/time semantics, status, evidence, decision flag,
       unresolved reason
     - Candidate semantics remain evidence-backed; no tensor is executed.
   * - ``OpenQuestion``
     - ID, classification, description, alternatives, evidence, deferred default, failure class
     - User-decision questions require at least two alternatives.
   * - ``DecisionRecord``
     - ID, decision, selected option, status, evidence, rationale
     - Selected/derived status must have compatible evidence or rationale.
   * - ``DependencyEvidenceRecord``
     - normalized/raw declaration, constraint/exact version, source file/section, kind, transport
       flags, validity, evidence ID
     - Constraint parses; exact version parses and satisfies the constraint.
   * - ``ConfidenceSummary``
     - verified, observed, inference, unresolved, unsupported counts
     - Every count is nonnegative and must equal computed analysis content.
   * - ``AnalysisReport``
     - schema/report/time/repository/revision, evidence-backed sections, all candidates, questions,
       decisions, evidence items, confidence, next mode
     - Unique identities, resolved provenance-compatible evidence, mirrored repository fields,
       computed confidence, and non-profile next mode.
   * - ``EnvironmentCandidate``
     - candidate/rationale, optional Python/Torch/TorchAudio/NumPy constraints and exact versions,
       other dependencies, source identity, strategy, evidence, trial/failure state, uncertainty,
       risks, command plan
     - Versions satisfy constraints; official package has exact package identity; Git/vendored
       strategies have a 40-character lowercase revision; failure state is consistent.
   * - ``EnvironmentCandidateGenerationResult``
     - schema, evidence, dependency records, candidates, unresolved constraints, source context,
       gates, target platform
     - All references resolve and provenance is compatible.
   * - ``EnvironmentResolutionReport``
     - schema/report/time/analysis link, evidence, ordered/attempted/selected candidates, risks,
       gates, artifact paths, materialization/verification flags, fingerprint, report/diagnostic,
       next lifecycle
     - Selected candidate exists; successful resolution requires exact principal versions, evidence,
       all five committed artifacts, materialization and verification success, fingerprint, valid
       report reference, and no unresolved blocker.
   * - ``SkillEvaluationScenario``
     - scenario ID, ``synthetic=true``, optional expected strategy/failure/next mode, decision and
       embedding expectations, checkpoint IDs
     - Evaluation inputs are explicitly synthetic.
   * - ``ScenarioInspectionResult``
     - schema/scenario identity, inventory, packaging/dependency/import/model/output/checkpoint/source
       observations, environment candidates, warnings
     - Environment candidates are validated canonical contracts.

Minimal analysis object
-----------------------

Constructing a complete report is easiest by validating canonical JSON. This offline example checks a
small structured claim before inserting it into a report:

.. code-block:: python

   from torch_dae.onboarding import ClaimStatus, EvidenceBackedClaim

   claim = EvidenceBackedClaim(
       statement="The package name appears in pyproject.toml.",
       status=ClaimStatus.LOCALLY_OBSERVED_BEHAVIOR,
       evidence_ids=("ev-pyproject",),
   )
   assert claim.evidence_ids == ("ev-pyproject",)

The parent :class:`AnalysisReport` will reject that claim until ``ev-pyproject`` resolves to a
compatible :class:`EvidenceItem`.

.. autoclass:: torch_dae.onboarding.contracts.EvidenceItemKind

.. autoclass:: torch_dae.onboarding.contracts.ClaimStatus

.. autoclass:: torch_dae.onboarding.contracts.OpenQuestionClassification

.. autoclass:: torch_dae.onboarding.contracts.SourceStrategy

.. autoclass:: torch_dae.onboarding.contracts.RecommendedNextMode

.. autoclass:: torch_dae.onboarding.contracts.FailureClassification

.. autoclass:: torch_dae.onboarding.contracts.CandidateTrialStatus

.. autoclass:: torch_dae.onboarding.contracts.DependencyKind

.. autoclass:: torch_dae.onboarding.contracts.EvidenceItem

.. autoclass:: torch_dae.onboarding.contracts.EvidenceBackedClaim

.. autoclass:: torch_dae.onboarding.contracts.ReportSection

.. autoclass:: torch_dae.onboarding.contracts.RepositoryIdentity

.. autoclass:: torch_dae.onboarding.contracts.VariantCandidate

.. autoclass:: torch_dae.onboarding.contracts.CheckpointCandidate

.. autoclass:: torch_dae.onboarding.contracts.SourceStrategyCandidate

.. autoclass:: torch_dae.onboarding.contracts.EmbeddingCandidate

.. autoclass:: torch_dae.onboarding.contracts.OpenQuestion

.. autoclass:: torch_dae.onboarding.contracts.DecisionRecord

.. autoclass:: torch_dae.onboarding.contracts.DependencyEvidenceRecord

.. autoclass:: torch_dae.onboarding.contracts.ConfidenceSummary

.. autoclass:: torch_dae.onboarding.contracts.AnalysisReport

.. autoclass:: torch_dae.onboarding.contracts.EnvironmentCandidate

.. autoclass:: torch_dae.onboarding.contracts.EnvironmentCandidateGenerationResult

.. autoclass:: torch_dae.onboarding.contracts.EnvironmentResolutionReport

.. autoclass:: torch_dae.onboarding.contracts.SkillEvaluationScenario

.. autoclass:: torch_dae.onboarding.contracts.ScenarioInspectionResult
