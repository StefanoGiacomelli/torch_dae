# Model Onboarding Skill

The canonical Phase 02 skill is `skills/audio-model-onboarding/`. Codex and Claude resolve to that
same directory through project-local symlinks, so the workflow is agent-neutral.

Supported modes are `analyze`, `resolve-environment`, `integrate`, `verify`, `card`, and `profile`.
`integrate` is a workflow mode, not a lifecycle state. The only lifecycle states are `draft`,
`analyzed`, `environment_resolved`, `checkpoint_verified`, `runtime_verified`, and `profiled`.
`profile` is reserved and must not produce measurements in Phase 02.

The end-to-end workflow is: local checkout or official upstream identifier -> static inspection ->
`ScenarioInspectionResult` observations -> evidence-grounded technical analysis report -> user
decision gates -> strict environment candidate generation -> integration plan -> verification plan
-> model-card draft. Phase 01 APIs remain responsible for environment materialization and checkpoint
acquisition.

The synthetic evaluation harness compares each scenario, its production-inspector observations, and
its golden analysis report. Golden reports must cite concrete fixture files, symbols, dependency
declarations, revisions, and checkpoint URLs; generic evidence placeholders are not sufficient.
Hidden checkpoint helpers must preserve helper symbol, expression status, unresolved components,
URL, filename, hash, and source-file provenance. Local real-Git evaluation covers full 40-character
revision observation, pinned-Git source candidates, and package metadata that remains distinct from
officiality evidence.

Static inspection shares a per-operation `InspectionBudget` across repository, packaging,
dependency, import, model, output, checkpoint, source-strategy, and environment-candidate inspectors.
The budget enforces file and byte limits and caches file text for the current operation only.

External PyTorch implementations are considered only when the upstream fixture is non-PyTorch and an
explicit second PyTorch repository is inspected. TensorFlow or JAX imports alone produce an
unsupported or unresolved result. Minimal vendoring requires broken package evidence, unsuitable
package/Git strategy evidence, copied-file provenance, adaptation notes, and a pinned upstream
revision.

Phase 02 does not integrate a production model or download real checkpoints. PANNs and all other
real model work begin only in Phase 03.
