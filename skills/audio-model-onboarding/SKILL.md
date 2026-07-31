---
name: audio-model-onboarding
description: Analyze, resolve, integrate, verify, card, and profile PyTorch audio model integrations according to project_spec.md.
---

# audio-model-onboarding

This is the canonical, agent-neutral onboarding skill for `torch-dae`. Treat `project_spec.md` as
normative. Preserve it unless the user explicitly asks to edit the spec.

Codex and Claude entry points must resolve to this same directory. Do not create agent-specific
scientific workflows. Keep model runtime dependencies out of the root environment. Execute only the
workflow mode explicitly requested by the user, and never create a Git commit.

Before using any mode, read the relevant references:

- `references/workflow-overview.md`
- `references/evidence-policy.md`
- `references/repository-analysis.md`
- `references/environment-resolution.md`
- `references/source-strategy.md`
- `references/checkpoint-discovery.md`
- `references/architecture-and-embeddings.md`
- `references/integration-planning.md`
- `references/runtime-verification.md`
- `references/model-card-authoring.md`
- `references/lifecycle-and-decision-gates.md`
- `references/failure-classification.md`
- `references/synthetic-evaluation.md`

Canonical request and response formats are available at:

- `templates/agent-request.md`
- `templates/agent-response.md`

Use deterministic utilities under `scripts/` for evidence collection. They may collect, normalize,
and validate static evidence; they do not replace scientific or architectural reasoning.
For a local paper or technical PDF, `scripts/extract_pdf_text.py` may extract its machine-readable
text layer under explicit limits. Extraction performs no OCR and does not make the text verified
scientific evidence without source review.

## Phase Handoff Protocol

Every mode accepts a stable `WORKFLOW_ID`. Before requesting a prerequisite attachment, run:

```bash
uv run python scripts/onboarding_handoff.py discover \
  --workflow-id <workflow-id> \
  --required-phase <phase> \
  --json
```

Validate the discovered manifest, every local artifact hash, `project_spec.md` hash, and canonical
skill fingerprint. Consume recorded user decisions and unresolved items. Do not ask the user to
reattach a canonical artifact that is already present. When a duplicate attachment is supplied,
compare its SHA-256 with the canonical artifact and stop on mismatch. Never search Desktop,
Downloads, home, or product-specific attachment directories. Without `WORKFLOW_ID`, auto-select
only when exactly one compatible active workflow exists; otherwise open a decision gate.

Perform phase work in `.torch-dae/workspaces/<workflow-id>/<phase>/<run-id>/` or a context-managed
system temporary directory. Record a managed run manifest, validate outputs in the workspace, and
promote only accepted canonical artifacts. For `analyze`, `resolve-environment`, `integrate`, and
`card`, promote the accepted handoff under `onboarding_reports/<workflow-id>/` before declaring the
phase complete. `verify` instead creates checkpoint-specific runtime observations only under
`verification_reports/`; it must not repurpose that root for a phase handoff.

Generate the deterministic external review bundle after promotion, then run scoped cleanup. Default
cleanup removes only recorded ephemeral workspaces and completed/failed trial environments for the
selected workflow. Report reusable repository/package caches, materialized model environments,
checkpoint caches, and external audit outputs that remain. Never delete an unrecorded path.

## Evidence Vocabulary

Every material claim must be classified as one of:

- `verified_upstream_fact`
- `locally_observed_behavior`
- `reasoned_inference`
- `user_provided_decision`
- `unresolved_ambiguity`
- `unsupported_claim`

Evidence items must identify their source kind:

- `source_file`
- `source_line_or_symbol`
- `package_metadata`
- `configuration_file`
- `official_documentation`
- `paper`
- `runtime_observation`
- `agent_inference`
- `user_decision`

Inferences require a rationale. A verified or observed claim must not cite agent inference as proof.
Absent evidence becomes an open question, not a fabricated value.

## `analyze` Mode

Purpose: investigate an unfamiliar upstream project and produce a structured technical analysis
report plus a Markdown rendering.

Required inputs: official repository URL, local repository checkout, or official package identifier.
Accept an optional stable `WORKFLOW_ID`; create or update its workflow record for accepted output.

Optional inputs: paper/documentation references, candidate checkpoint URL, requested variant,
requested checkpoint, intended task, target use, intended embedding, target platform.

Prerequisites: repository baseline gates must pass; no production model card or wrapper may exist
for the target; root dependencies must remain model-agnostic.

Ordered procedure:

1. Establish repository identity, revision, license, package, release, maintenance, and official
   status evidence.
2. Run static inventory utilities: repository, packaging, dependencies, checkpoint candidates, model
   candidates, and output candidates.
3. Read upstream source, docs, papers, and package metadata needed to interpret the static evidence.
   Use bounded local PDF extraction only when a supplied document has a machine-readable text layer.
4. Produce the machine-readable report using `templates/technical-analysis-report.json`.
5. Render the human report with `templates/technical-analysis-report.md`.
6. Present unresolved scientific choices before making user-dependent decisions.

Evidence requirements: every architecture, preprocessing, output, embedding, checkpoint,
dependency, and source-strategy claim must cite evidence or be marked unresolved. Variant-sensitive
claims and embeddings must use declared variant/checkpoint scopes. Host-published checksums must
preserve their explicit algorithm and published-not-locally-verified status.

Generated outputs: technical analysis report JSON, Markdown report, open-question list, candidate
source strategy list, checkpoint candidates, environment evidence summary.

User-decision gates: ambiguous model variant, card scope, source implementation, official
checkpoint, preprocessing, outputs, embedding selection, technical access/authentication blocker,
license metadata issue, distribution/publication decision, or non-equivalent wrapper behavior.

Failure conditions: unsupported upstream, missing authoritative source, insufficient evidence,
path-unsafe local checkout, attempted repository-code execution, public-network dependence in tests,
or fabricated scientific facts.

Prohibited behavior: generate a model card from unsupported fields, download checkpoints, import
upstream code in the root environment, execute `setup.py`, or select an ambiguous embedding silently.

Completion criteria: strict report validation passes, Markdown report is consistent with the JSON,
all unresolved items are explicit, the user has seen decision gates, the accepted handoff is
promoted, its review bundle is generated, and scoped cleanup is verified.

Next allowed lifecycle transition: `analyzed` after report review and card authoring; otherwise
continue in `analyze` or `resolve-environment`.

## `resolve-environment` Mode

Purpose: determine an evidence-supported, reproducible compatibility configuration.

Required inputs: `WORKFLOW_ID` with an accepted analyze handoff and selected source/checkpoint
strategy. Discover the accepted analysis locally before requesting any missing attachment.

Optional inputs: target OS/architecture, manually supplied constraints, selected candidate ID,
controlled trial results.

Prerequisites: analyze-mode evidence is sufficient; user decisions affecting compatibility are
resolved; environment contracts and CLI are available.

Ordered procedure:

1. Discover and validate the accepted analyze handoff and consume its decisions and unresolved items.
2. Collect Python, PyTorch, TorchAudio, NumPy, build backend, source revision, CI, Docker, and import
   evidence.
3. Generate ordered candidates using `scripts/generate_environment_candidates.py`.
4. Trial only an explicitly selected evidence-motivated candidate in an isolated model-specific
   environment.
5. Classify every failure with `references/failure-classification.md`.
6. On success, prepare `environments/<card-id>/environment.json`, `pyproject.toml`, `uv.lock`,
   `sources.json`, and `verify_environment.py` using the committed environment contracts and
   commands.

Evidence requirements: candidate rationale, expected compatibility evidence, trial status, failure
classification, diagnostics, unresolved risk, platform evidence, and exact commands.

Generated outputs: environment-resolution report JSON, environment artifact draft paths, selected
candidate record, and failed-candidate diagnostics.

User-decision gates: multiple source strategies, multiple meaningful compatibility tracks,
unsupported platform, technical access/authentication blocker, license metadata issue, or
insufficient evidence.

Failure conditions: all evidence-supported candidates exhausted, external blocker, unsupported
implementation, source build failure, missing wheels, incompatible checkpoint, or unsafe execution.

Prohibited behavior: arbitrary Cartesian version search, hidden environment materialization, adding
model dependencies to the root environment, relying on transitive installation for a directly
imported dependency, or duplicating the environment subsystem.

Completion criteria: either (a) draft resolution is complete—an evidence-supported candidate,
isolated import/constructor trial, coherent environment drafts, synchronized locks, no fingerprint,
and no lifecycle promotion—or (b) canonical materialization and verification satisfy the strict
`environment_resolved` lifecycle contract. In both cases artifact references agree, the accepted
handoff is promoted, its review bundle is generated, and scoped cleanup is verified.

Next allowed lifecycle transition: `environment_resolved` only for completion state (b). Draft
resolution may recommend `integrate` without lifecycle promotion.

## `integrate` Mode

Purpose: implement the project-side wrapper while preserving upstream semantics.

Required inputs: `WORKFLOW_ID`, accepted analyze and resolve-environment handoffs, resolved source
and checkpoint strategies, selected variant, and selected embedding default when required.

Optional inputs: user target use, wrapper package path, adaptation notes.

Prerequisites: the user explicitly requested `MODE: integrate`; analysis is complete and reviewed;
the source strategy, target model variant, target checkpoint, and model-specific environment
strategy are resolved; the primary embedding is resolved or explicitly deferred by user decision;
and the user explicitly authorized production integration.

Ordered procedure:

1. Discover and validate accepted analyze and resolve-environment handoffs.
2. Define wrapper package path, model construction, checkpoint loading, preprocessing ownership,
   sample-rate/channel/length behavior, output mapping, embedding interface, device behavior,
   deterministic behavior, and tests.
3. Verify source-strategy rules in `references/source-strategy.md`.
4. Preserve upstream inference semantics and document any deviation.
5. Add the wrapper, model-specific package code, committed environment and checkpoint
   specifications, integration documentation, and tests needed for the selected model.

Evidence requirements: source provenance, upstream forward semantics, preprocessing evidence,
checkpoint compatibility evidence, selected embedding evidence, and user decisions.

Generated outputs: integration plan, wrapper and package code, committed specifications,
documentation, and tests for the selected model.

User-decision gates: non-equivalent wrapper behavior, multiple plausible embeddings, ambiguous
preprocessing, unclear output semantics, or vendored adaptation scope.

Failure conditions: source strategy unsupported, equivalence cannot be justified, required evidence
missing, integration prerequisites unresolved, or production integration not explicitly authorized.

Prohibited behavior: adding model dependencies to the root project, committing checkpoint binaries,
silently beginning verification or another workflow mode, creating a Git commit, semantic
reimplementation without provenance, or presenting logits/task decisions as embeddings.

Completion criteria: integration plan is reviewable, evidence-backed, declares verification
requirements, carries unresolved items forward, is promoted with an accepted handoff, is bundled,
and has completed scoped cleanup.

Next allowed lifecycle transition: none. `integrate` is a workflow mode, not a lifecycle state.
Existing committed lifecycle states remain authoritative.

## `verify` Mode

Purpose: verify runtime behavior after environment resolution and wrapper implementation.

Required inputs: `WORKFLOW_ID`, accepted prerequisite handoffs, model card draft, wrapper
implementation, resolved environment, checkpoint specification, expected outputs, and embeddings.

Optional inputs: accelerator targets, extra input cases, user target assertions.

Prerequisites: environment verified, wrapper exists, the selected model and checkpoint are explicit,
and card-declared outputs and embeddings are evidence-backed.

Ordered procedure:

1. Discover and validate accepted prerequisite handoffs and consume recorded decisions.
2. Test random initialization when supported, checkpoint initialization, invalid checkpoint behavior,
   model variant agreement, and loading diagnostics.
3. Test canonical `[B,C,T]` waveform inputs, sample-rate behavior, channels, valid lengths, short and
   long inputs, zero input, batches, and deterministic synthetic waveforms.
4. Observe every declared output and embedding for key, rank, shape, dtype, device, temporal
   semantics, NaN/Inf, and repeated-call behavior.
5. Generate a structured checkpoint-specific verification report.

Evidence requirements: environment fingerprint, checkpoint hash, source revision, package identity,
test inputs, observed outputs, embedding observations, warnings, failures, unsupported capabilities.

Generated outputs: verification plan and verification report JSON.

User-decision gates: unsupported capabilities, output/card mismatch, embedding/card mismatch,
unsupported device behavior, or non-reproducible runtime observations.

Failure conditions: required runtime tests fail, card/report disagree, checkpoint cannot load,
outputs/embeddings are missing, or environment is not verified.

Prohibited behavior: acquiring any checkpoint other than the explicitly selected checkpoint,
acquiring checkpoints outside the checkpoint subsystem, running the model outside its isolated
environment, lifecycle promotion from schema validity alone, profiling, or fabricated runtime
observations.

Completion criteria: every card-declared output and embedding is observed, the runtime report
validates under `verification_reports/`, a review bundle is generated, and scoped cleanup is
verified. No pre-runtime artifact is stored in `verification_reports/`.

Next allowed lifecycle transition: `runtime_verified` only after `checkpoint_verified` and runtime
verification prerequisites are satisfied.

## `card` Mode

Purpose: generate or update a checkpoint-specific model card from verified evidence.

Required inputs: `WORKFLOW_ID`, accepted prerequisite handoffs, user decisions, source/checkpoint
strategy, environment artifacts, wrapper evidence where applicable, and verification report where
applicable.

Optional inputs: unresolved issue decisions, intended default embedding, accepted limitations.

Prerequisites: actual committed lifecycle model is checked; every populated field has traceable
evidence; unresolved information remains explicit.

Ordered procedure:

1. Discover and validate all accepted prerequisite handoffs and consume recorded decisions.
2. Map report fields to the committed `ModelCard` schema.
3. Use null, unresolved status, or issues where evidence is absent.
4. Enforce checkpoint-specific scope: one family, one variant, one checkpoint.
5. Validate through Pydantic and generated JSON Schema.
6. Check lifecycle gates and card/report agreement.

Evidence requirements: upstream source, paper/docs, environment evidence, checkpoint evidence,
runtime observation, or explicit user decision for every populated field.

Generated outputs: model-card JSON draft, evidence references, checkpoint specification, embedding
specification, environment references, known issues.

User-decision gates: checkpoint scope, default embedding, unresolved technical
access/authentication blocker, license metadata issue, distribution/publication decision, source
strategy, lifecycle promotion, or accepted known issue.

Failure conditions: evidence reference missing, lifecycle skip, unsupported fact promoted, schema
failure, card not checkpoint-specific, or report/card disagreement.

Prohibited behavior: optional filler values, family-level cards for incompatible checkpoints, hidden
lifecycle promotion, or legal conclusions from license metadata.

Completion criteria: Pydantic and JSON Schema validation both pass, lifecycle gates are truthful,
the accepted card handoff is promoted, its review bundle is generated, and scoped cleanup is
verified.

Next allowed lifecycle transition: the next legal committed lifecycle state only; do not invent
states beyond `project_spec.md`.

## `profile` Mode

Purpose: reserved future mode that may inspect eligibility for profiling.

Required inputs: `WORKFLOW_ID`, runtime-verified card, and verification report.

Optional inputs: future profiling target platform and protocol.

Prerequisites: none because no profiling workflow has been implemented.

Ordered procedure:

1. Refuse profiling execution.
2. List missing prerequisites when asked.
3. Explain expected future inputs without producing profiling evidence.

Evidence requirements: runtime verification status may be inspected; no profiling measurement is
created.

Generated outputs: eligibility notes only.

User-decision gates: future profiling protocol selection remains unresolved.

Failure conditions: any attempt to measure latency, memory, energy, MACs, FLOPs, or benchmark
performance.

Prohibited behavior: latency measurement, memory profiling, energy measurement, MAC/FLOP
calculation, computational characterization, benchmark reports, or lifecycle promotion to
`profiled`.

Completion criteria: profiling remains reserved and no fabricated profiling evidence exists.

Next allowed lifecycle transition: none.

Profiling remains unavailable until a model is runtime_verified and a profiling workflow is
explicitly implemented and invoked.
