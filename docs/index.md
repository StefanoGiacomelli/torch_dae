# torch-dae

**Reproducible audio-model integration, runtime verification, embedding access, and profiling.**

`torch-dae` turns an explicit audio-model implementation and checkpoint into a validated,
repository-backed integration. It keeps model identity, runtime environment, checkpoint acquisition,
waveform/output contracts, embeddings, verification evidence, and optional profiling evidence
separate but connected.

The repository currently ships three accepted PANNs / AudioSet integrations. All three have
`runtime_verified` Model Cards and canonical Profiling v1 Technical Cards for CPU and Apple MPS
execution contexts.

## Start here

If you are new to the project, read these pages in order:

1. {doc}`getting-started/installation` — choose between the lightweight package install and the full
   repository workspace.
2. {doc}`getting-started/mental-model` — understand the control plane, model runtimes, Model Cards,
   and Technical Cards.
3. {doc}`getting-started/quickstart` — inspect the accepted registry and choose a workflow.

## Choose a workflow

**I want to use a supported model**
: Start with {doc}`models/index`, then follow {doc}`tutorials/panns-inference` for the currently
  supported PANNs models.

**I want to extract embeddings**
: Read {doc}`user-guide/model-execution` and {doc}`user-guide/embeddings`.

**I want to understand a Model Card or Technical Card**
: Read {doc}`user-guide/model-cards` and {doc}`profiling/technical-cards`.

**I want to integrate a new model**
: Start with {doc}`skill/overview`, use the copy-paste requests in {doc}`skill/prompt-library`, and
  follow the end-to-end {doc}`tutorials/audio-model-onboarding`.

**I want to profile an accepted model**
: Start with {doc}`profiling/overview` and {doc}`tutorials/profiling`; the profiling workflow is
  independent from model onboarding.

**I want to contribute to or release the framework**
: Use {doc}`development/architecture`, {doc}`development/contributing`, and
  {doc}`development/releasing`.

```{toctree}
:maxdepth: 2
:caption: Getting started

getting-started/installation
getting-started/mental-model
getting-started/quickstart
```

```{toctree}
:maxdepth: 2
:caption: Tutorials

tutorials/panns-inference
tutorials/audio-model-onboarding
tutorials/profiling
tutorials/agent-interaction
```

```{toctree}
:maxdepth: 2
:caption: Models and user guide

models/index
models/panns-runtime
user-guide/model-execution
user-guide/embeddings
user-guide/model-registry
user-guide/environments
user-guide/checkpoints
user-guide/model-cards
user-guide/technical-cards
```

```{toctree}
:maxdepth: 2
:caption: Profiling

profiling/overview
profiling/protocol
profiling/technical-cards
profiling/contributing
```

```{toctree}
:maxdepth: 2
:caption: AI skills

skill/overview
skill/prompt-library
```

```{toctree}
:maxdepth: 2
:caption: API reference

api/index
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference/cli
reference/schemas
reference/lifecycle
```

```{toctree}
:maxdepth: 2
:caption: Development

development/architecture
development/contributing
development/testing
development/documentation
development/releasing
```

```{toctree}
:hidden:

checkpoint-management
environment-management
runtime-execution
model-onboarding-skill
onboarding-artifacts
onboarding-evidence-policy
```

## Project principles

`torch-dae` uses a few deliberate constraints that explain much of the architecture:

- **One Model Card identifies one model-family / variant / checkpoint tuple.** A card is not a vague
  family-level description.
- **The root package is a control plane, not a universal model environment.** Heavy model-runtime
  dependencies remain isolated.
- **Checkpoint acquisition is explicit.** Model construction never silently downloads weights.
- **Model support and model profiling are separate.** A `runtime_verified` Model Card may have zero
  or many Technical Cards.
- **Unsupported behavior fails explicitly.** The wrapper does not silently resample, downmix, pad,
  crop, normalize, or reinterpret tensors unless that behavior is part of the accepted contract.

## Citation

Cite the software using the repository's `CITATION.cff`. When discussing the framework design,
standardization rationale, or deployment methodology, also cite the IEEE ISCC paper:
DOI `10.1109/ISCC65549.2025.11326439`.

## Funding

Research project: *Methods of Computational Auditory Scene Analysis and Synthesis supporting
extended and immersive reality services*.

Research activities were mainly funded under Ministerial Decree (DM) 118/2023, Mission 4,
Component 1, Investment 4.1 of the National Recovery and Resilience Plan (PNRR) — “PNRR Research” —
CUP: E11I23000100001.
