# Onboarding Artifacts

The onboarding workflow provides strict machine-readable contracts for technical analysis reports,
environment-candidate generation results, environment-resolution reports, stable workflow records,
and phase-handoff manifests. Schemas are generated through `scripts/generate_schemas.py`; do not
hand-edit generated schemas.

Accepted pre-runtime phase artifacts are committed as:

```text
onboarding_reports/<workflow-id>/
├── workflow.json
├── analyze/handoff.json
├── resolve-environment/handoff.json
├── integrate/handoff.json
└── card/handoff.json
```

Only accepted phase directories need exist. Handoffs use repository-relative paths and SHA-256
digests, carry consumed user decisions and unresolved items, and record allowed next modes. External
attachments may remain digest-only evidence and never become required future local paths. Explicit
supersession records the prior accepted handoff digest.
Canonical phase reports use JSON or Markdown; deterministic source-reduction evidence may also use
the narrowly allowed `.diff` suffix when declared and hash-addressed by the phase handoff.

Shared repository outputs may evolve across legal later phases only through an
`artifact_supersessions` entry in the later handoff. Each entry records the repository-relative
path, latest accepted originating phase and SHA-256, new SHA-256, non-empty reason, and optionally
the prior handoff SHA-256. The same path must be a declared output of the containing later phase.
Validation preserves every historical declaration while checking the current filesystem against
the latest accepted hash. Missing, stale, duplicate, backward, ambiguous, cyclic, or forked
transitions fail validation.

Artifact supersession cannot target accepted handoffs, canonical phase reports, workflow history,
`project_spec.md`, generated schemas, verification reports, checkpoints, credentials, or ignored
runtime state. Discovery of an earlier accepted phase reports affected repository outputs as
superseded instead of calling the immutable historical handoff corrupt. Bundle metadata includes
the validated transition list and the latest external-artifact resolution; the archive contains
all historical handoffs and only the current repository bytes for a superseded shared path.

Use the root control-plane command to discover, validate, promote, bundle, and clean:

```bash
uv run python scripts/onboarding_handoff.py discover \
  --workflow-id <workflow-id> \
  --required-phase <phase> \
  --json
```

Promotion uses an exact allowlist: `workflow.json`, the selected phase's `handoff.json`, and every
declared phase-local output. All declared files and hashes are validated before mutation. Any other
regular file, including an unreferenced JSON or Markdown file in a nested directory, and every
symlink are rejected. Outputs under canonical external repository roots such as `environments/`,
`src/`, `tests/`, `docs/`, and `model_cards/` are hash-validated at their repository paths and are
never copied from the managed promotion source. Promotion remains atomic and refuses to overwrite an
accepted handoff without explicit supersession.
Candidate validation resolves the complete accepted history plus the pending handoff before the
atomic rename, so stale prior hashes, undeclared changes, absent later outputs, and new-hash
mismatches leave the accepted workflow byte-identical.

Bundles normalize TAR and gzip metadata, independently compare declared and actual inventories, and
write a `<archive>.sha256` sidecar. The external result JSON explicitly reports archive cleanliness,
metadata normalization, inventory agreement, repository cleanliness, and handoff validation.
`metadata/artifact-manifest.json` hashes every staged archive file that exists when the manifest is
created. It intentionally excludes itself and the subsequently generated `bundle-result.json`,
`declared-archive-inventory.json`, and `actual-archive-inventory.json`; those four exclusions are
recorded in the bundled result metadata, while the declared and actual inventories still cover every
archive member.

Cleanup consumes only managed run manifests. Retained diagnostics must be moved to and explicitly
recorded under `.torch-dae/reports/onboarding/<workflow-id>/` before workspace deletion. A retained
path at or below a planned deletion root is a blocking `retention_conflict`, and conflict handling
performs no deletion. Existing external audit outputs are checked before mutation against every
planned deletion root: equality, containment in either direction, and both the supplied and
symlink-resolved paths are protected. A conflict instructs the caller to move the audit output
outside managed deletion roots, update `external_paths`, and rerun cleanup. Each dry run or execution
atomically persists a durable cleanup receipt under
`.torch-dae/reports/onboarding/<workflow-id>/cleanup/` before eligible execution deletion begins.
The final receipt embeds finalized source manifests, planned and removed paths, retained managed and
external paths, external protection conflicts, cache retention classes, verification, errors, and
applicable SHA-256 values. External audit paths preserve the manifest spelling, are never selected by
cleanup flags, and report existing files, directories, symlinks, and missing outputs distinctly.

Analysis claims and embedding candidates may carry optional `variant_ids` and `checkpoint_ids`.
Empty tuples mean report-wide applicability; nonempty IDs must resolve to candidates declared in the
same report. Checkpoint candidates may retain the backward-compatible `hash_evidence` string and may
also carry structured `published_checksums`. Each structured checksum records its exact algorithm,
validated digest, evidence reference, and fixed `published_not_locally_verified` state. Published
MD5 metadata never substitutes for later local SHA-256 verification.

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
locally observed `environments/<environment-id>/pyproject.toml`, `uv.lock`, or `environment.json`
evidence.
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
`model_cards/`, environments under `environments/`, accepted pre-runtime handoffs under
`onboarding_reports/`, and checkpoint-specific runtime observations created by `verify` under
`verification_reports/`. Diagnostic reports, managed workspaces, checkpoints, materialized
environments, and coverage JSON remain under ignored `.torch-dae/`. Review bundles are generated
outside the repository and are not duplicated under `onboarding_reports/`.
Environment-resolution reports may reference committed verification reports as
`verification_reports/<card-id>/<report>.json` or environment diagnostics relative to `.torch-dae` as
`reports/environments/<environment-id>/<fingerprint>/<report>.json`; checkpoint and source report
paths are not valid environment-promotion references.
