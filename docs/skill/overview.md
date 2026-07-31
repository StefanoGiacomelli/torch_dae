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
`onboarding_reports/` before attachments are requested. Phase runs use recorded managed workspaces,
promote accepted canonical outputs, generate a deterministic external review bundle, and finish with
scoped cleanup.
