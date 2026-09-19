# Prompt library

This page provides complete starting prompts for the canonical `torch-dae` skills. Replace values
inside `<...>` and send one phase at a time. Do not combine lifecycle phases simply to reduce the
number of messages: the review boundary between phases is intentional.

## Before you start

For onboarding, choose one stable `WORKFLOW_ID` and reuse it throughout the integration. If a
workflow already exists, provide that ID and ask the skill to discover accepted prerequisites from
the repository instead of reattaching them.

Use `AUTO_DISCOVER` only when the selected phase is allowed to discover the value. Use `UNRESOLVED`
when you explicitly want ambiguity to remain open for review.

## Start a new integration — `analyze`

```text
Use the canonical audio-model-onboarding skill in this repository.

MODE: analyze
WORKFLOW_ID: <stable-workflow-id>

MODEL_NAME: <model-or-family-name>
UPSTREAM_REPOSITORY: <official-repository-url>
PAPER_OR_TECHNICAL_REFERENCE: <url-doi-or-none>

TARGET_VARIANT: <variant-or-AUTO_DISCOVER>
TARGET_CHECKPOINT: <checkpoint-or-AUTO_DISCOVER>
PREFERRED_EMBEDDING: <embedding-or-UNRESOLVED>

ADDITIONAL_CONSTRAINTS:
<platform/task/deployment constraints or NONE>

Inspect the upstream project statically and produce the canonical evidence-grounded analysis.
Identify authoritative implementation candidates, checkpoint candidates, the waveform/preprocessing
contract, public outputs, embedding candidates, dependency constraints, source-strategy options, and
all unresolved decisions.

Do not download a checkpoint, create an environment, execute upstream code, integrate a wrapper, or
continue into another lifecycle phase. Validate and finalize analyze-mode artifacts, report the
review bundle, and stop at the analyze boundary for human review.
```

## Continue — `resolve-environment`

```text
Use the canonical audio-model-onboarding skill in this repository.

MODE: resolve-environment
WORKFLOW_ID: <existing-workflow-id>

Discover and validate the accepted analyze handoff from the repository. Use its accepted user
decisions and unresolved items as the starting point.

Resolve a reproducible model-specific environment for the selected model/source/checkpoint strategy.
Keep the root torch-dae environment model-agnostic. Perform only the controlled compatibility work
authorized by resolve-environment mode, classify failures explicitly, validate the resulting
environment artifacts, finalize this phase, and stop for human review.

Do not integrate model code or acquire/verify the checkpoint in this request.
```

## Continue — `integrate`

```text
Use the canonical audio-model-onboarding skill in this repository.

MODE: integrate
WORKFLOW_ID: <existing-workflow-id>

Discover and validate the accepted analyze and resolve-environment handoffs. Integrate only the
already selected model/variant/checkpoint scope.

Implement the public torch-dae wrapper and required integration artifacts using the accepted source
strategy and contracts. Preserve the canonical waveform interface, lazy model-specific imports, and
model-specific runtime isolation. Add focused tests and documentation required by the integration.

Validate the complete staged-equivalent worktree, promote accepted integrate-mode outputs, run the
canonical finalization path, report all artifacts and validation results, and stop at the integrate
boundary.

Do not perform checkpoint runtime verification in this request.
```

## Continue — `verify`

```text
Use the canonical audio-model-onboarding skill in this repository.

MODE: verify
WORKFLOW_ID: <existing-workflow-id>

Discover and validate the accepted integrate prerequisites. Build strict runtime-verification
targets before checkpoint acquisition, resolve and verify the accepted model environment, acquire
the selected checkpoint through the canonical checkpoint manager, and execute the declared public
runtime checks.

Require complete passed coverage for every target-required check. Record unsupported optional
behavior and limitations explicitly rather than silently omitting them. Promote only successful
canonical verification evidence, run the canonical finalization path, report the review bundle and
remaining limitations, and stop at the verify boundary.

Do not author the final Model Card in this request.
```

## Finish onboarding — `card`

```text
Use the canonical audio-model-onboarding skill in this repository.

MODE: card
WORKFLOW_ID: <existing-workflow-id>

Discover and validate all accepted prerequisite handoffs and checkpoint-specific runtime evidence.
Author exactly one final Model Card for the accepted model/variant/checkpoint tuple. Preserve the
observed input/output contract, embedding interface, environment identity, checkpoint identity,
runtime-verification coverage, device support, and known limitations without strengthening claims
beyond the accepted evidence.

Validate the Model Card and repository state, promote the accepted card handoff, run the canonical
finalization path, report the final lifecycle state and review bundle, and stop. New onboarding must
terminate at runtime_verified; do not embed Profiling v1 results into the Model Card.
```

## Resume an interrupted onboarding workflow

```text
Use the canonical audio-model-onboarding skill.

WORKFLOW_ID: <existing-workflow-id>

Inspect the canonical workflow state and accepted handoffs already present in the repository. Report:
- current accepted phase and lifecycle state;
- validated canonical artifacts already available;
- unresolved decisions that still block progress;
- the next legal lifecycle phase;
- any historical/current control-plane drift that is informational rather than evidence corruption.

Do not execute the next phase yet. Stop after the recovery/status report so I can approve the next
request explicitly.
```

## Independent review of an onboarding result

```text
Review the following completed torch-dae onboarding phase independently from the conversation that
produced it.

WORKFLOW_ID: <workflow-id>
PHASE: <analyze|resolve-environment|integrate|verify|card>
REVIEW_BUNDLE: <path-to-review-bundle>
SHA256_SIDECAR: <path-to-sidecar>

Use the repository contracts and the bundle contents as the review basis. Verify archive integrity,
phase scope, declared evidence, validation results, unresolved items, artifact inventory, and any
supersession/control-plane drift metadata. Distinguish blocking defects from acceptable limitations.

Do not modify repository files. Return a concise audit with PASS/FAIL per review area and the exact
evidence supporting each finding.
```

# Profiling prompts

Profiling requires an accepted `runtime_verified` Model Card. It uses the
`audio-model-profiling` skill, not onboarding `profile` compatibility mode.

## Resolve profiling eligibility

```text
Use the canonical audio-model-profiling skill in this repository.

MODE: resolve
MODEL_CARD_ID: <accepted-model-card-id>

Resolve the canonical Model Card and confirm that it is eligible for Profiling v1. Report the
wrapper entry point, checkpoint identity, verified environment, native sample rate, documented input
constraints, and any limitations relevant to profiling.

Do not run a profiling campaign yet.
```

## Plan a campaign

```text
Use the canonical audio-model-profiling skill in this repository.

MODE: plan
MODEL_CARD_ID: <accepted-model-card-id>
DEVICES: <auto|cpu|mps|cuda|cuda:index>[, ...]
ENERGY: <auto|off>

Describe exactly what Profiling v1 will execute: resolved devices, canonical duration, minimum-input
search, batch matrix, CPU thread regimes, warmups, measured iterations, energy policy, and expected
candidate artifacts. Do not execute the campaign. Stop after the plan so I can approve or redirect
scope.
```

## Profile CPU only

```text
Use the canonical audio-model-profiling skill in this repository.

MODE: profile
MODEL_CARD_ID: <accepted-model-card-id>
DEVICES: cpu
ENERGY: <auto|off>
OUTPUT_DIR: profiling_candidates/<campaign-name>
ALLOW_PRIVILEGED_ENERGY: false

Execute only the approved CPU Profiling v1 campaign. Keep all output in the candidate workspace,
never in technical_cards/. Report both CPU thread regimes, successful/unsupported benchmark
conditions, minimum-input search result, generated Technical Card/NPZ paths, and campaign-result
path. Do not promote candidate evidence.
```

## Profile all automatically discovered devices without energy

```text
Use the canonical audio-model-profiling skill in this repository.

MODE: profile
MODEL_CARD_ID: <accepted-model-card-id>
DEVICES: auto
ENERGY: off
OUTPUT_DIR: profiling_candidates/<campaign-name>
ALLOW_PRIVILEGED_ENERGY: false

Execute the approved Profiling v1 campaign. Smoke-test each discovered backend, profile successful
devices, report failed device diagnostics without substituting CPU for an explicitly requested
backend, validate every generated candidate Technical Card, and leave all evidence in the candidate
workspace. Do not promote it.
```

## Profile with energy and explicit privileged-counter consent

Use this only when you have deliberately decided to permit local privileged counters such as Apple
PowerMetrics.

```text
Use the canonical audio-model-profiling skill in this repository.

MODE: profile
MODEL_CARD_ID: <accepted-model-card-id>
DEVICES: <auto-or-explicit-device-list>
ENERGY: auto
OUTPUT_DIR: profiling_candidates/<campaign-name>
ALLOW_PRIVILEGED_ENERGY: true

I explicitly authorize the profiling command to request local privileged hardware counters when the
configured energy backend requires them. Do not alter sudoers, persist credentials, or perform
geolocation. If a component cannot be measured, preserve the resulting incomplete coverage and
unaccounted_components honestly rather than treating a partial total as complete energy.

Execute the approved campaign, validate generated candidates, report whether privilege was actually
used, and do not promote candidate evidence.
```

## Validate an existing candidate campaign

```text
Use the canonical audio-model-profiling skill in this repository.

MODE: validate
CANDIDATE_DIRECTORY: <repository-relative-candidate-directory>

Locate every candidate Technical Card JSON in this directory and validate each card together with
its referenced raw NPZ asset. Report Technical Card ID, validation result, Model Card/checkpoint
association, raw-asset integrity, and any errors. Optional unavailable energy/FLOPs/accelerator
memory must not be treated as failure when the card represents that absence correctly.

Do not modify or promote any candidate artifact.
```

## Final review/report before promotion

```text
Use the canonical audio-model-profiling skill in this repository.

MODE: finalize
CANDIDATE_DIRECTORY: <repository-relative-candidate-directory>

Review the completed profiling campaign as candidate evidence. Report the campaign-result path,
each validated Technical Card JSON/NPZ pair, device/thread-regime coverage, unsupported conditions,
minimum-input findings, energy measurement kind and coverage, runtime classification, and all
limitations relevant to comparison.

Do not copy files into technical_cards/ and do not modify the referenced Model Card. Stop with a
clear promotion-readiness assessment for human review.
```

## Low-level templates

For automation-oriented requests, use the repository templates directly:

- `skills/audio-model-onboarding/templates/agent-request.md`
- `skills/audio-model-profiling/templates/agent-request.md`

The tutorials {doc}`../tutorials/audio-model-onboarding` and {doc}`../tutorials/profiling` show how
these requests fit into complete workflows.
