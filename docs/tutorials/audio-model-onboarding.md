# Integrate a new audio model with the onboarding skill

This tutorial describes the complete human-supervised lifecycle for adding one checkpoint-specific
model integration to `torch-dae`.

The onboarding skill is not a one-shot code generator. It is a five-phase protocol with review
boundaries between phases:

```text
analyze
  -> resolve-environment
      -> integrate
          -> verify
              -> card
                  -> runtime_verified
```

Profiling happens later through the separate `audio-model-profiling` skill.

## Prerequisites

Work from the full repository workspace, not only the PyPI installation. See
{doc}`../getting-started/installation`.

Before starting, identify at least the official upstream project. A paper, model page, checkpoint
URL, intended variant, target task, and preferred embedding are helpful but may be left for
controlled discovery when the selected phase permits it.

Choose one stable workflow identifier, for example:

```text
<your-stable-workflow-id>
```

Reuse that exact `WORKFLOW_ID` for every onboarding phase.

## Phase 1 — Analyze

Start with the complete request in {doc}`../skill/prompt-library`. The analyze phase performs static
inspection of upstream code, metadata, documentation, and supplied scientific references. It should
identify:

- authoritative implementation candidates;
- architecture/variant candidates;
- waveform and preprocessing requirements;
- public output semantics;
- embedding candidates;
- direct/runtime dependency evidence;
- checkpoint candidates and authority evidence;
- possible source-integration strategies;
- unresolved questions requiring human choice.

It must not create a model environment, download a checkpoint, or execute upstream code.

Review the analysis before continuing. In particular, do not accept an embedding or checkpoint just
because it appears plausible: the selected artifact becomes part of the public model contract.

## Phase 2 — Resolve the environment

Request `resolve-environment` with the same `WORKFLOW_ID`. The skill discovers the accepted analyze
handoff from the repository and turns dependency evidence into a reproducible model-specific runtime.

The root environment remains lightweight. Model dependencies belong to the model environment under
`environments/<environment-id>/` and its managed runtime state.

A successful phase should distinguish:

- declared dependency inputs;
- exact locked resolution;
- platform constraints;
- source materialization strategy;
- compatibility trial evidence;
- final verification status.

Environment success does not prove that the checkpoint can be loaded correctly; that belongs to
runtime verification.

## Phase 3 — Integrate the public wrapper

Request `integrate` only after the model identity, source strategy, checkpoint scope, environment,
preprocessing contract, public outputs, and embedding decisions are sufficiently resolved.

The wrapper must expose the `torch-dae` public audio contract while preserving model-specific
behavior. Model-specific imports remain lazy so importing the root package does not require every
supported model's dependencies.

The phase normally adds or updates:

- wrapper implementation;
- model registry/integration metadata;
- focused tests;
- model-specific documentation required to use the wrapper;
- integration evidence and handoff artifacts.

The integration phase validates code and repository state but does not yet claim checkpoint-specific
runtime behavior.

## Phase 4 — Verify the checkpointed runtime

Request `verify` with the same workflow ID.

The skill first creates strict runtime targets describing the checks that must be observed. It then
resolves and verifies the model environment, acquires the checkpoint through the canonical manager,
loads the model, and executes the declared public behavior.

Depending on the accepted contract, verification may cover:

- state loading;
- waveform forward inference;
- output keys/shapes/dtypes;
- probability behavior;
- one or more embeddings;
- device behavior;
- gradient behavior;
- declared failure/unsupported behavior.

A successful import is not enough. Every required target check must be present exactly once and
pass before runtime verification can close successfully.

## Phase 5 — Author the Model Card

Request `card` only after accepted runtime evidence exists.

The resulting Model Card represents exactly one model-family / architecture-variant / checkpoint
tuple. It consolidates the accepted identity, sources, scientific references, environment,
checkpoint, waveform contract, output contract, embeddings, capabilities, verification evidence,
and limitations.

New onboarding terminates at:

```text
card_status = runtime_verified
```

Profiling does not change this status and does not add performance measurements into the Model Card.

## What happens after each phase

The skill manages detailed lifecycle mechanics itself: prerequisite discovery, immutable handoffs,
managed workspaces, validation gates, deterministic review bundles, cleanup, and legal artifact
supersession when a later accepted phase changes a shared output.

As the reviewer, focus on:

- whether the evidence supports the scientific/technical claims;
- whether unresolved choices are explicit;
- whether the phase stayed within scope;
- whether validation passed;
- whether the files reported by the skill match what you expected.

The lower-level mechanics are documented in {doc}`../api/onboarding-contracts` and the individual
mode pages under {doc}`../skill/overview`.

## Next step: profiling

Once the Model Card is accepted as `runtime_verified`, you may stop: the model is already supported.

If empirical performance evidence is useful, continue with the independent tutorial
{doc}`profiling`. One Model Card can accumulate multiple Technical Cards from different profiling
sessions without changing its bytes.
