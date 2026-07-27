# Model Onboarding Skill

The canonical skill is
[`skills/audio-model-onboarding/`](https://github.com/StefanoGiacomelli/torch_dae/blob/main/skills/audio-model-onboarding/SKILL.md).
Codex and Claude resolve to that directory through project-local relative symlinks, so the workflow
and scientific policy remain agent-neutral.

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
