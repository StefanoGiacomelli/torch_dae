# Changelog

All notable changes to this project are documented in this file.

## Unreleased

No changes yet.

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
