# Changelog

All notable changes to this project are documented in this file.

## Unreleased

### Canonical lifecycle finalization and cross-agent review packaging

- Added `scripts/onboarding_handoff.py finalize`, the single canonical end-of-task path. It
  composes existing `validate`, `bundle`, and `cleanup` primitives with the bounded set of required
  repository gates (`validate_repository.py`, `validate_skill_artifacts.py`, `check_worktree_patch.py
  --json`, `git diff --check`), reused via subprocess rather than duplicated. Ordering: evidence
  validation, required gates, cleanup preflight (dry-run; blocks on unmanaged workspace content),
  the deterministic review bundle (embedding the gates, lifecycle state, evidence-invariance summary,
  and cleanup preflight as archive metadata), then optional real cleanup execution, then one
  `finalize-result.json` exposing every absolute artifact path, the workflow/lifecycle state,
  evidence-invariance status, and cleanup status/receipt. Any required-gate failure, unaccepted
  phase, evidence drift, or unmanaged workspace content fails the call outright.
- Added `scripts/onboarding_handoff.py run-manifest create`, the single canonical *allocator* for a
  managed workspace run: it atomically creates `run_root` and its manifest together and returns the
  absolute path, rather than registering a workspace created independently beforehand.
- Hardened `cleanup`/`finalize` to distinguish a genuinely legacy workflow (no workspace content at
  all, returning a structured, successful `not_applicable` result) from unmanaged workspace content
  (a run directory present without its own manifest), which now blocks cleanup and finalize instead
  of being silently guessed at or deleted.
- Corrected evidence-invariance reporting to distinguish unchanged phase-local evidence from validly
  superseded shared artifacts, instead of describing both as simply "unchanged".
- Updated the canonical `audio-model-onboarding` skill, its request/response templates, and related
  docs with an explicit agent completion contract: every successfully completed lifecycle mode must
  allocate workspaces only through `run-manifest create`, run `finalize`, and report lifecycle state,
  validation-gate status, evidence invariance, cleanup status, limitations, and absolute artifact
  paths, supporting independent cross-agent/cross-LLM review.

### Stable local-package identity and concurrent wheel caching

- Made local package identity content-addressed over wheel build inputs so Git commits, HEAD-only
  changes, and evidence/report changes do not alter package identity, environment fingerprints, or
  wheel-cache keys; Git state is retained only as separate informational provenance.
- Added bounded cross-platform inter-process locking, safe stale-lock recovery, process-specific
  build paths, deterministic timestamps, and atomic publication for shared local wheel caches.
- Added real process-level regression coverage for same-identity contention, failed builders,
  waiter recovery, stale/live locks, identity-scoped parallelism, final-state validation, and
  read-only audit generation.

### Environment wheel dependency closure

- Added a deterministic card-independent preflight that validates active local-wheel runtime
  requirements against direct and transitive packages reachable from each accepted lock before
  environment creation; missing and incompatible requirements remain separate structured evidence.
- Corrected all three PANNs environment projects with the six derived package-runtime pins and
  canonically regenerated their locks without changing Python or numerical runtime pins.
- Made local wheel builds offline, linked successful preflight evidence into materialization
  results, and added marker, extras, normalization, reachability, ordering, reuse, and isolation
  regressions.

### Authoritative checkpoint acquisition

- Added strict schema `2.0.0` checkpoint authority contracts with Zenodo metadata-only resolution,
  exact file/size identity, algorithm-tagged MD5 and SHA-256 evidence, and bounded response
  provenance while preserving legacy schema `1.0.0` sources.
- Enforced exact size, every published checksum, independently observed SHA-256, and maximum-byte
  ceilings before content-addressed cache installation; offline reuse now revalidates bytes,
  authority checksums, specification fingerprint, and cached metadata provenance.
- Added card-independent checkpoint `resolve`, `ensure-spec`, and `info-spec` CLI operations, a
  dedicated safe provider-filename grammar including metric-bearing `=`, and authority-aware runtime
  target/report validation.
- Documented the Python 3.11 all-groups setup gate, category-based cleanup retention, and read-only
  final Git inventory for intentionally non-writable repository metadata.

### Historical control-plane provenance

- Treated accepted handoff skill and specification hashes as immutable historical provenance, so
  later generic control-plane hardening no longer retroactively invalidates accepted workflows.
- Kept exact current-hash enforcement for pending promotion candidates and atomic mismatch failure,
  while permitting later phases to consume prerequisites created under older control planes.
- Added deterministic validation, discovery, repository-report, and bundle metadata that separates
  per-phase historical hashes from current hashes and reports skill and specification drift.

### Card-independent environment lifecycle

- Made accepted environment definitions the authority for direct resolution, materialization, and
  infrastructure verification; retained card-based entry points as delegating conveniences.
- Added strict materialization, environment-verification, and runtime-target contracts and schemas,
  deterministic evidence-complete fingerprints, and matching final-card evidence requirements.
- Separated environment evidence from checkpoint-specific runtime reports and added explicit
  `resolve`, `materialize`, and environment-ID `verify` CLI operations.
- Added verify and card onboarding phases to the accepted lifecycle contract without promoting or
  changing any existing model workflow handoff.

### Verification evidence completeness

- Required successful environment results to contain nonempty, uniquely named import and smoke
  observations with passed status only; failed results remain partial diagnostic evidence.
- Added schema `2.0.0` runtime-target required and optional check contracts, target-aware report
  coverage enforcement, and exact repository/model-card target binding while preserving conservative
  legacy readability.
- Corrected remaining environment-ID authority documentation and retained card-oriented environment
  commands as compatibility conveniences.

### Staged-equivalent whitespace validation

- Added exact `.gitattributes` whitespace exemptions for the byte-preserved PANNs vendored
  substrate and deterministic source-reduction patch without weakening validation elsewhere.
- Added a temporary-index working-tree validator so untracked non-ignored outputs receive the same
  whitespace gate as the future staged commit while the real Git index remains unchanged.
- Added Git-fixture regressions for tracked, untracked, ignored, deleted, renamed, exempt, and
  non-exempt paths plus temporary-index cleanup on success and failure.

### Onboarding phase-handoff hardening

- Added strict cross-phase artifact supersession for shared repository outputs, including ordered
  lineage validation, protected canonical paths, atomic pending-handoff validation, historical
  discovery status, and bundle metadata resolving the latest accepted file.
- Closed promotion to an exact declared-artifact allowlist, with byte-preserving rejection of
  unreferenced JSON, Markdown, nested runtime content, and symlinks.
- Made cleanup conflict-aware and all-or-nothing for retained paths and existing external outputs;
  pre-deletion protection covers supplied and symlink-resolved path topology, and durable atomic
  receipts include finalized source manifests and explicit conflict and retention inventories.
- Added explicit bundle cleanliness and normalization result fields plus deterministic SHA-256
  sidecars, while documenting the intentional self-exclusions of the bundled artifact manifest.
- Added strict workflow and phase-handoff contracts, generated schemas, committed
  `onboarding_reports/`, local prerequisite discovery, atomic promotion, explicit supersession, and
  accepted three-tuple analyze/resolve-environment migration fixtures.
- Added deterministic normalized review bundles with working-tree evidence, artifact hashes,
  declared/actual inventory checks, and external result metadata.
- Added managed workflow workspaces and manifest-scoped cleanup that preserves reusable caches,
  materialized environments, checkpoints, and external audit outputs by default.
- Clarified direct dependency closure, shared source/environment evidence, constructor-trial scope,
  external execution failures, and successful draft environment resolution without lifecycle
  promotion.
- Preserved `verification_reports/` exclusively for checkpoint-specific runtime observations.

### Analyze-skill hardening

- Made checkpoint discovery binary-safe with deterministic skipped-file reporting and retained valid
  textual candidates when unrelated binary or malformed files are present.
- Added class-qualified output candidates with lexical method ownership and source spans.
- Added backward-compatible variant/checkpoint scopes, structured host-published checksum metadata,
  strict reference validation, regenerated schemas, and more complete Markdown reports.
- Added bounded local PDF text-layer extraction through the lightweight root `pypdf` dependency,
  without OCR, external processes, network retrieval, or model-runtime dependencies.
- Documented best-effort source revision evidence, metadata-only authoritative checkpoint-host
  analysis, and locked root-environment execution semantics.

### Documentation and release metadata

- Added Zenodo concept and version DOI metadata, the `0.1.0` release date, and a concept DOI badge.
- Linked the README documentation references to the stable Read the Docs release.
- Added release and version badges, the embedding-pipeline graphic, and an illustrative
  model-wrapper usage example.
- Updated the release workflow to `actions/download-artifact@v8`.

## 0.1.0 - 2026-07-28

### Repository foundation

- Added the lightweight control-plane package, strict typed contracts, generated schemas, registry,
  CLI structure, canonical skill links, synthetic fixtures, and repository safety rules.
- Established checkpoint-specific model-card identity, canonical waveform inputs, explicit
  capabilities, and ignored runtime state.

### Environment and checkpoint management

- Implemented reproducible environment specifications, materialization, verification, caching,
  source strategies, offline behavior, and the environment CLI.
- Implemented checkpoint acquisition, hashing, cache integrity, package-bundle ownership checks,
  local/remote sources, typed failure handling, redacted diagnostics, and checkpoint CLI behavior.
- Added backend-built local wheels, sanitized model-environment subprocesses, Git source recovery,
  wheel metadata verification, cross-document identity validation, and failure cleanup.

### Audio-model onboarding

- Added the canonical evidence-grounded onboarding skill with static inspection, environment
  candidate generation, analysis/report rendering, decision gates, source strategies, integration
  planning, runtime verification planning, and model-card authoring.
- Grounded synthetic scenario evaluations in production-inspector observations, including dependency,
  checkpoint-helper, package identity, Git revision, source strategy, and embedding evidence.
- Added explicit production integration prerequisites while keeping profiling reserved.

### Validation and quality

- Added strict dual Pydantic/JSON Schema validation, repository and skill validators, synthetic
  behavioral checks, public-safety scans, CI, Codecov configuration, and local line/branch coverage
  thresholds.
- Added validation for new model cards, static wrapper entry points, committed environments,
  verification reports, root dependency isolation, and forbidden binary assets.

### Documentation and public metadata

- Added public package metadata, Apache-2.0 licensing, citation metadata, contribution guidance,
  typed-package marker, public README, and canonical agent request/response templates.
- Documented environment management, checkpoint management, onboarding evidence, artifacts,
  workflow boundaries, and development commands.
- Added warning-clean Sphinx documentation with MyST Markdown, Furo, a curated API reference, and
  NumPy-style public API docstrings.
- Added Read the Docs configuration and contributor guidance for documentation maintenance.
- Added PyPI Trusted Publishing on published GitHub Releases and manual TestPyPI publication, both
  using OIDC without stored package-index credentials.
- Added build-once release artifact validation, seven-day workflow artifacts, clean-wheel checks,
  and wheel/source distributions attached to GitHub Releases.
- Added software and IEEE citations, ORCID, research funding acknowledgement, and Apache NOTICE
  metadata.
