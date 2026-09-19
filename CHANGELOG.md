# Changelog

All notable changes to this project are documented in this file.

## Unreleased

No unreleased changes are currently recorded.

## 0.2.0 - 2026-09-19

### Accepted PANNs integrations

- Added three checkpoint-specific PANNs / AudioSet integrations: Cnn14 at 16 kHz, ResNet38 at
  32 kHz, and Wavegram-Logmel-Cnn14 at 32 kHz.
- Added typed PyTorch wrappers with explicit `[B, 1, T]` waveform contracts, `[B, 527]` logits and
  sigmoid probabilities, and the upstream post-`fc1` `[B, 2048]` embedding interface.
- Added accepted isolated environments, authoritative checkpoint specifications, runtime-verification
  reports, and `runtime_verified` Model Cards for all three integrations.
- Preserved upstream checkpoint payloads outside the repository; accepted weights are acquired and
  verified from authoritative providers instead of redistributed.

### Profiling v1 and Technical Cards

- Added the Profiling v1 subsystem with deterministic synthetic inputs, 10 warmups, 50 measured
  iterations, canonical batch scaling, empirical minimum-input discovery, host and accelerator
  memory evidence, optional architecture metrics, and explicit energy-provenance semantics.
- Added CPU, MPS, and CUDA device selectors, CPU `single_thread` / `native_default` regimes, raw
  lossless NPZ timing assets, and candidate Technical Card generation.
- Added CodeCarbon-backed energy collection and explicit Apple PowerMetrics privilege consent, while
  preserving incomplete sensor coverage through `coverage_complete` and `unaccounted_components`.
- Promoted nine reviewed canonical Technical Cards for the accepted PANNs models: CPU
  single-thread, CPU native-default, and Apple MPS contexts for each model.
- Extended repository-wide validation so canonical Technical Card JSON/NPZ pairs, cross-references,
  hashes, reconstructed timing summaries, privacy constraints, and orphan assets are checked in CI.

### Model onboarding and reproducible runtimes

- Hardened the `audio-model-onboarding` lifecycle across analyze, environment resolution,
  integration, verification, Model Card generation, finalization, review bundling, and cleanup.
- Added strict cross-phase handoffs, shared-artifact supersession, deterministic review bundles,
  evidence-invariance reporting, managed workspaces, and conflict-aware cleanup receipts.
- Made accepted environment definitions the authority for materialization and verification, with
  card-oriented commands retained as conveniences. Static repository validation is host-independent
  while runtime materialization continues to enforce declared platform constraints.
- Added deterministic local-wheel dependency-closure checks, stable content-addressed package
  identity, concurrent wheel-cache locking, and atomic publication.
- Added authoritative checkpoint acquisition with provider metadata, exact size/checksum validation,
  bounded downloads, content-addressed reuse, offline revalidation, and card-independent CLI entry
  points.
- Strengthened runtime-target and verification-report semantics, including environment/checkpoint
  identity, required observations, target-aware report coverage, and conservative legacy reading.

### Skills and developer workflows

- Added and documented the dedicated `audio-model-profiling` skill with resolve, plan, profile,
  validate, and finalize/review workflows independent from model onboarding.
- Added copy-paste-ready prompt libraries and canonical request templates for model onboarding,
  profiling, continuation, and independent review.
- Added `torch-dae --version` and expanded CLI help for model, checkpoint, profiling, Model Card, and
  Technical Card workflows.

### Documentation and public usability

- Rebuilt the README and hosted documentation around installation, supported models, input/output
  shapes, checkpoint acquisition, model execution, embeddings, profiling, Technical Cards, and
  agent-assisted integration.
- Added a living PANNs runtime guide and API reference without modifying immutable onboarding
  evidence captured by the accepted integration workflow.
- Added practical tutorials for PANNs inference, model onboarding, and profiling, together with a
  developer-oriented Technical Card field guide.
- Clarified the distinction between the lightweight PyPI package and the full repository workspace,
  including the isolation of heavyweight model-runtime dependencies.

### Validation, CI, and release engineering

- Hardened Linux CI for platform-specific environment artifacts while preserving runtime platform
  enforcement.
- Added repository, documentation, skill, schema, package, distribution, and Technical Card
  regressions across Python 3.11 and 3.12.
- Retained Codecov OIDC uploads, warning-clean Sphinx builds, wheel/source-distribution checks, and
  clean-wheel installation in the production release workflow.
- Continued production publishing through GitHub Releases, the protected `pypi` environment, and
  PyPI Trusted Publishing without package-index credentials.

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
