# AI skills overview

`torch-dae` ships agent-neutral skills for two different jobs:

- **audio-model onboarding** turns authoritative upstream evidence into an accepted,
  checkpoint-specific `runtime_verified` Model Card;
- **audio-model profiling** consumes an already accepted Model Card and produces candidate Profiling
  v1 Technical Card evidence.

These workflows are deliberately separate. Profiling does not advance Model Card lifecycle state,
and onboarding does not create canonical Technical Cards.

```text
new upstream model
      |
      v
analyze -> resolve-environment -> integrate -> verify -> card
                                                   |
                                                   v
                                      runtime_verified Model Card
                                                   |
                                                   v
                               resolve -> plan -> profile -> validate
                                                   |
                                                   v
                                  candidate Technical Card evidence
```

## How to use the skills

Use one phase per agent request. A phase request authorizes only the work required by that phase.
Review the result before starting the next phase.

For onboarding, keep one stable `WORKFLOW_ID` across all five lifecycle phases. The skill discovers
accepted prerequisite handoffs from `onboarding_reports/`, so you should not repeatedly attach
artifacts that are already canonical in the repository.

For profiling, identify one accepted `runtime_verified` Model Card and choose the requested device
scope and energy policy. Profiling uses its own campaign identity and never reuses the onboarding
workflow as a new lifecycle phase.

Start from {doc}`prompt-library` when you want a copy-paste-ready request. The canonical low-level
onboarding request template remains available at
`skills/audio-model-onboarding/templates/agent-request.md`; profiling has an equivalent template at
`skills/audio-model-profiling/templates/agent-request.md`.

## Onboarding lifecycle

```{toctree}
:maxdepth: 1

analyze
resolve-environment
integrate
verify
card
```

### `analyze`

Static inspection only. Establish upstream identity, architecture, preprocessing, output semantics,
checkpoint candidates, embedding candidates, dependencies, source strategies, and unresolved
questions. This phase does not download checkpoints or create model environments.

### `resolve-environment`

Turn dependency evidence into a reproducible model-specific environment. Controlled trials are
allowed only inside the phase's isolated runtime state. The root `torch-dae` environment remains
model-agnostic.

### `integrate`

Add the public wrapper and integration artifacts after model identity, source strategy, checkpoint,
input/output contract, and embedding choices are resolved. The wrapper must preserve the canonical
public audio contract and keep model-specific imports lazy.

### `verify`

Create a strict runtime target, resolve and verify the environment, acquire the checkpoint through
the checkpoint manager, and observe the public wrapper under controlled execution. Successful
runtime verification requires complete passed coverage of all target-required checks.

### `card`

Author the final checkpoint-specific Model Card from accepted evidence. A successful card closes
new onboarding at `runtime_verified`.

## Profiling workflow

Profiling is documented separately because it operates on accepted models rather than creating
one. See:

- {doc}`../profiling/overview` for the operational entry point;
- {doc}`../tutorials/profiling` for an end-to-end campaign;
- {doc}`../profiling/protocol` for the fixed Profiling v1 methodology;
- {doc}`../user-guide/technical-cards` for interpreting the resulting evidence.

The onboarding skill still contains a legacy/reserved `profile` compatibility boundary, but new
profiling work must use the canonical `audio-model-profiling` skill.

## What to review after every phase

A professional agent response should make the following explicit:

1. **Scope** — which phase ran and what was intentionally not attempted.
2. **Evidence** — which authoritative files, upstream sources, runtime observations, and user
   decisions support the result.
3. **Artifacts** — canonical or candidate files created or changed.
4. **Validation** — commands/gates executed and whether they passed.
5. **Open decisions** — unresolved scientific or technical choices that require human input.
6. **Next allowed action** — the next phase, if the current one was accepted.

Treat missing evidence as unresolved, not as permission to guess. A polished answer is not a
substitute for validated repository artifacts.

## Advanced lifecycle mechanics

The onboarding implementation also records immutable handoffs, deterministic review bundles,
cleanup receipts, artifact supersession, and control-plane provenance. These mechanisms protect
reproducibility and independent review but are normally handled by the skill itself rather than by
the person requesting a model integration.

For the underlying contracts, see {doc}`../api/onboarding-contracts`,
{doc}`../api/onboarding-inspection`, and {doc}`../api/report-rendering`.
