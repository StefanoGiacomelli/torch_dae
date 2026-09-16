# Model Onboarding Skill

The canonical skill is
[`skills/audio-model-onboarding/`](https://github.com/StefanoGiacomelli/torch_dae/blob/main/skills/audio-model-onboarding/SKILL.md).
Codex and Claude resolve to that directory through project-local relative symlinks, so the workflow
and scientific policy remain agent-neutral.

Run model-agnostic skill tooling through the locked root environment:

```bash
uv sync --all-groups --frozen
uv run python skills/audio-model-onboarding/scripts/inspect_repository.py <checkout> --json
```

The ignored root `.venv` is expected control-plane state, not a materialized model environment.
`pypdf` is installed there only for bounded document analysis; PyTorch and all model-runtime
dependencies remain isolated under the model-environment subsystem. Report root environment
creation when it occurs, and keep `.venv`, bytecode, and tool caches out of audit archives.

For a supplied local paper, use
`uv run python skills/audio-model-onboarding/scripts/extract_pdf_text.py <paper.pdf> --json`.
This extracts only a machine-readable text layer under file, page, and character limits. It does not
retrieve documents, provide OCR, or turn extracted prose into verified evidence.

Supported modes are `analyze`, `resolve-environment`, `integrate`, `verify`, `card`, and `profile`.
`integrate` is a workflow mode, not a lifecycle state. The committed lifecycle states remain
`draft`, `analyzed`, `environment_resolved`, `checkpoint_verified`, `runtime_verified`, and
`profiled`.

## Workflow boundaries

- `analyze` statically inspects an upstream project and produces evidence-grounded reports. It does
  not execute upstream code, download checkpoints, or silently choose a source, variant,
  checkpoint, or embedding.
- `resolve-environment` derives compatibility candidates from evidence and may run controlled trials
  only in isolated model-specific environments. Promotion requires successful recreation and
  verification with exact evidence and failure records.
- `integrate` may add a real wrapper, model-specific package code, committed environment and
  checkpoint specifications, documentation, and tests. It requires an explicit `MODE: integrate`
  request, reviewed analysis, resolved source strategy/variant/checkpoint/environment strategy, a
  resolved or explicitly deferred primary embedding, and explicit production authorization.
- `verify` may acquire only the explicitly selected checkpoint and execute only the selected model
  through the existing environment and checkpoint infrastructure.
- `card` creates or updates one checkpoint-specific card only from validated evidence and completed
  workflow artifacts.
- `profile` is reserved. Profiling remains unavailable until a model is `runtime_verified` and a
  profiling workflow is explicitly implemented and invoked.

Every mode executes only its own scope. No mode adds model dependencies to the root project, commits
checkpoint binaries, silently begins another mode, or creates a Git commit.

## Cross-conversation handoffs

Every request accepts `WORKFLOW_ID`. The skill first discovers the required accepted phase through
`scripts/onboarding_handoff.py`, validates local artifact hashes and historical control-plane hash
syntax, reports skill/specification drift from the current repository, then consumes recorded
decisions and unresolved items. Accepted hashes remain immutable historical provenance; a later
control-plane revision does not retroactively invalidate the handoff. It asks for an
attachment only when the canonical prerequisite is genuinely absent. Duplicate attachments must
match the canonical digest.

Pending phase candidates use different semantics: their canonical-skill fingerprint and
`project_spec.md` SHA-256 must match the current repository exactly before atomic promotion. A new
phase can consume accepted prerequisites with older hashes, but records current hashes for itself.
Historical hashes identify the declared control plane and do not, without retained files or audit
evidence, reconstruct its bytes or establish a Git revision mapping.

Phase execution uses a managed `.torch-dae/workspaces/<workflow-id>/<phase>/<run-id>/`, allocated
only through `scripts/onboarding_handoff.py run-manifest create` before any managed workspace work
begins; a run directory left unregistered is unmanaged content that blocks cleanup rather than being
guessed at. Accepted pre-runtime output is atomically promoted under `onboarding_reports/`, followed
by the canonical `scripts/onboarding_handoff.py finalize` command, which re-validates accepted
evidence, runs the required repository gates, runs cleanup, and generates the deterministic external
review bundle in one call. Default cleanup removes recorded ephemeral workspaces and trial
environments only; reusable caches, materialized environments, checkpoints, and external audit
bundles remain. External audit outputs are never cleanup targets. Retained diagnostics must be moved
under `.torch-dae/reports/onboarding/<workflow-id>/`; cleanup writes a durable receipt there before
deleting an eligible workspace.
`retained_paths` is reserved for those copied bounded diagnostics; categorized environments and
caches are retained by category instead. Before the Python 3.11 validation suite, run
`uv sync --all-groups --frozen --python 3.11`. Final Git inventory may use read-only status/cached
diffs or staged-equivalent validation and does not require `git write-tree` when Git metadata is
intentionally non-writable.

Resolve-environment may finish successfully as a draft without lifecycle promotion when isolated
import/construction passed but production source, wrapper, or card prerequisites are intentionally
absent. Constructor trials do not establish checkpoint compatibility, forward/output/embedding
correctness, inference equivalence, or runtime verification.

## Evidence and evaluation

The end-to-end workflow is static inspection, structured observations, evidence-grounded analysis,
decision gates, compatibility resolution, integration, runtime verification, and model-card
authoring. See the [evidence policy](onboarding-evidence-policy.md) and
[artifact guide](onboarding-artifacts.md).

Synthetic evaluation compares each scenario, production-inspector observations, and its golden
analysis report. Reports must cite concrete files, symbols, dependency declarations, revisions,
checkpoint candidates, source strategies, and embedding tensor candidates. Static inspection shares
a bounded `InspectionBudget` and never executes repository code.

Use the
[canonical agent request](https://github.com/StefanoGiacomelli/torch_dae/blob/main/skills/audio-model-onboarding/templates/agent-request.md)
to start a workflow and the
[canonical response](https://github.com/StefanoGiacomelli/torch_dae/blob/main/skills/audio-model-onboarding/templates/agent-response.md)
to report results consistently.
