# Onboarding Artifacts

The onboarding workflow provides strict machine-readable contracts for technical analysis reports,
environment-candidate generation results, and environment-resolution reports. Schemas are generated
through `scripts/generate_schemas.py`; do not hand-edit generated schemas.

Skill templates live under `skills/audio-model-onboarding/templates/`:

- `technical-analysis-report.json`
- `technical-analysis-report.md`
- `environment-resolution-report.json`
- `integration-plan.md`
- `verification-plan.md`
- `decision-request.md`
- `model-card-draft.json`
- `agent-request.md`
- `agent-response.md`

Deterministic utilities live under `skills/audio-model-onboarding/scripts/` and operate on local
synthetic repositories without executing upstream code or contacting public services.

`scripts/generate_environment_candidates.py` emits an `EnvironmentCandidateGenerationResult` with
schema version, evidence items, normalized dependency records, ordered candidates, unresolved
constraints, source-strategy context, decision gates, and optional target platform. Target platform
is a result-level field; individual candidates carry only evidence-backed compatibility details.
Official-package candidates include exact `source_package_name` and `source_package_version`, and
promotion requires exact matching package identity from verified upstream `package_metadata` or
locally observed `environments/<card-id>/pyproject.toml`, `uv.lock`, or `environment.json` evidence.
`sources.json`, `verify_environment.py`, unrelated files, runtime observations, and inference cannot
establish package identity.
Any remaining `source_strategy_decision_gates` entry blocks `environment_resolved`.

Golden synthetic reports under `tests/skills/golden/` are validated against production-inspector
observations rather than scenario oracle fields. The validator rejects reports whose cited files,
model symbols, embedding tensor origins, checkpoint URLs, dependency declarations, source
strategies, or revisions do not match inspected fixture evidence.
Checkpoint hashes are compared only with hashes statically associated with the exact observed source
file, helper symbol, URL, and filename candidate; repository-global hashes do not satisfy a report.

Committed production artifacts remain governed by existing repository contracts: model cards under
`model_cards/`, environments under `environments/`, and verification reports under
`verification_reports/`. Runtime state, checkpoints, reports, materialized environments, and
coverage JSON remain under ignored `.torch-dae/`.
Environment-resolution reports may reference committed verification reports as
`verification_reports/<card-id>/<report>.json` or environment diagnostics relative to `.torch-dae` as
`reports/environments/<card-id>/<fingerprint>/<report>.json`; checkpoint and source report paths are
not valid environment-promotion references.
