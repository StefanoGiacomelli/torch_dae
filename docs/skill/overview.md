# Skill overview

The canonical skill is an agent-neutral, mode-scoped workflow. It turns authoritative upstream
evidence and controlled observations into reviewable analysis, environment, integration,
verification, and card artifacts.

```{toctree}
:maxdepth: 1

analyze
resolve-environment
integrate
verify
card
```

The skill never treats missing evidence as fact. Repository analysis is static by default;
execution, network acquisition, environment creation, and integration occur only in modes that
authorize them.

Every mode accepts `WORKFLOW_ID`. Accepted prerequisites are discovered and hash-validated from
`onboarding_reports/` before attachments are requested. Phase runs use managed workspaces allocated
through `run-manifest create`, promote accepted canonical outputs, and finish with the canonical
`finalize` command, which validates evidence invariance, runs the required repository gates,
performs cleanup preflight, optionally executes explicitly requested cleanup, and generates the
deterministic external review bundle.

When a shared repository output legitimately changes in a later phase, the later handoff records a
strict artifact-supersession edge. Historical handoffs and their hashes remain evidence; workflow
validation and bundles resolve the current file only through the unique ordered accepted chain.

## Profiling boundary

The onboarding skill terminates new Model Card lifecycle at `runtime_verified`. Its `profile` mode
remains reserved compatibility only. Profiling v1 is an independent, repeatable Technical Card
workflow and is not implemented yet.
