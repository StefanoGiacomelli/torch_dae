# Working with an integration or profiling agent

The `torch-dae` skills are designed for supervised technical work. The agent performs one bounded
phase, produces reviewable artifacts, reports validation, and stops. The human reviewer decides
whether the evidence is sufficient to continue.

## The basic interaction pattern

```text
you: request one phase
          |
          v
agent: inspect -> act within scope -> validate -> report
          |
          v
you: review evidence and unresolved decisions
          |
          +---- revise/re-run current phase
          |
          `---- approve next phase
```

This boundary matters. An `analyze` request is not permission to install dependencies or download a
checkpoint, and a profiling `plan` is not permission to start a potentially long hardware campaign.

## What to put in the request

For a new model integration, provide as much of the following as you know:

- official upstream repository or package;
- paper, DOI, model card, or authoritative technical reference;
- intended model/architecture variant;
- intended checkpoint;
- preferred embedding, if you already know it;
- target platform or deployment constraints;
- a stable `WORKFLOW_ID` that will be reused across phases.

Unknown values should stay explicit. `AUTO_DISCOVER` asks the selected phase to discover candidates;
`UNRESOLVED` states that a decision is intentionally open.

For profiling, provide:

- the accepted Model Card ID;
- requested device selector(s);
- `--energy auto` or `--energy off`;
- a repository-local candidate output directory;
- explicit permission only if privileged energy counters are acceptable.

Use {doc}`../skill/prompt-library` for complete copy-paste-ready examples.

## How to review the response

A good response should let you answer six questions without reading the agent's hidden reasoning:

1. **What exactly was requested and completed?**
2. **Which evidence supports each material decision?**
3. **Which files were created or changed?**
4. **Which validation gates passed or failed?**
5. **What remains unresolved or unsupported?**
6. **What is the next legal phase?**

For onboarding, accepted evidence is persisted through canonical workflow handoffs. For profiling,
candidate Technical Cards and raw NPZ assets remain outside `technical_cards/` until independent
review and later promotion.

## Evidence language

Onboarding deliberately distinguishes:

- verified upstream facts;
- locally observed behavior;
- reasoned inference;
- explicit user decisions;
- unresolved ambiguity;
- unsupported claims.

This vocabulary prevents a plausible interpretation from silently becoming a repository fact.
Runtime observations likewise prove only what was actually exercised by the declared target and
environment.

## Common requests that should be split

Avoid requests such as:

```text
Analyze this repository, integrate the best checkpoint, verify it, create the Model Card,
profile every device, and commit everything.
```

That request erases the decision gates around variant selection, checkpoint authority, source
strategy, embedding choice, dependency resolution, runtime verification, profiling scope, and
promotion.

Prefer:

```text
analyze
-> review
resolve-environment
-> review
integrate
-> review
verify
-> review
card
-> review
profiling resolve/plan
-> review
profile/validate
```

## When to stop the workflow

Stop rather than forcing progress when:

- the authoritative checkpoint is ambiguous;
- the requested model variant cannot be distinguished from another variant;
- source licensing or source authority is unresolved;
- the embedding location is inferred but not supported strongly enough for the intended contract;
- environment compatibility cannot be reproduced;
- required runtime checks fail;
- a profiling device fails its smoke test;
- energy evidence is partial and the intended analysis requires complete component coverage.

An explicit limitation is a valid result. Inventing evidence is not.

## Files versus prose

Treat prose as a navigation layer over repository evidence. For onboarding, the accepted handoff,
reports, runtime targets, verification reports, and Model Card are authoritative. For profiling, the
Technical Card JSON and raw `.npz` contain the evidence; `campaign-result.json` contains orchestration
metadata for the local campaign.

See {doc}`audio-model-onboarding` and {doc}`profiling` for complete examples.
