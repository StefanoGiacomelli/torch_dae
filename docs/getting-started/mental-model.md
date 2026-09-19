# Mental model

`torch-dae` is easiest to understand as a small set of connected layers rather than as a single
model-loading function.

## 1. The root control plane

The root `torch_dae` package owns model-agnostic operations:

- strict typed contracts;
- Model Card discovery and validation;
- environment resolution and materialization;
- checkpoint acquisition and cache validation;
- runtime-verification orchestration;
- profiling orchestration;
- onboarding and profiling workflow artifacts.

It intentionally avoids installing every model's runtime dependencies into one Python environment.

## 2. The model runtime

Each accepted model points to an isolated environment definition under `environments/`. That runtime
contains the exact dependencies needed to import and execute the model wrapper.

For the current PANNs integrations, this separation prevents the root environment from depending on
PyTorch and TorchLibrosa while still allowing model execution in a controlled environment.

```text
repository / root environment
        │
        ├── inspect Model Card
        ├── materialize model environment
        ├── acquire checkpoint
        └── launch controlled work
                    │
                    ▼
          isolated model environment
                    │
                    └── public wrapper → model forward / embedding
```

## 3. The checkpoint

A checkpoint is a concrete pretrained byte asset. The wrapper never downloads one implicitly.

`CheckpointManager` or `torch-dae checkpoint ensure <card-id>` resolves the accepted checkpoint,
acquires it when required, validates the resulting bytes, and returns a local content-addressed path.
The model wrapper then receives that local path explicitly through `from_pretrained()` or
`load_checkpoint()`.

## 4. The Model Card

A Model Card answers:

> **What exact model/checkpoint is supported, and what is its accepted runtime contract?**

One Model Card represents exactly one model-family, architecture variant, and checkpoint tuple. It
records, among other fields:

- model and wrapper identity;
- authoritative checkpoint identity;
- recommended environment;
- waveform shape and native sample rate;
- output tensors and semantics;
- available embeddings;
- supported capabilities;
- locally verified devices;
- runtime-verification references;
- known limitations and unresolved issues.

The current accepted PANNs cards are all in the terminal `runtime_verified` lifecycle state.

## 5. The Technical Card

A Technical Card answers a different question:

> **How did an already accepted model behave in one empirical profiling context?**

It records device/backend and execution context together with performance/resource observations such
as latency, throughput, real-time factor, memory, energy coverage, minimum-input search, architecture
metrics, and a reference to raw measurements.

A Model Card is therefore a model identity/runtime contract. A Technical Card is empirical evidence
about one profiling session. Profiling does not modify the Model Card.

```text
                    ┌───────────────────────────────┐
                    │          Model Card           │
                    │ model/checkpoint/runtime API  │
                    └───────────────┬───────────────┘
                                    │ 1 : N
                    ┌───────────────┴───────────────┐
                    │       Technical Card(s)       │
                    │ device/session measurements   │
                    └───────────────────────────────┘
```

## 6. The wrapper

The public wrapper is the PyTorch-facing execution surface. Conceptually it provides:

```text
from_random(...)
from_pretrained(checkpoint=...)
load_checkpoint(...)
preprocess(...)
forward(...)
predict_probability(...)
available_embeddings()
compute_embedding(...)
```

The exact supported subset and tensor semantics are declared by the Model Card and implementation.
For PANNs, `forward()` returns raw logits as the primary tensor while also exposing native sigmoid
probabilities and the verified clipwise embedding.

## 7. The onboarding skill

Adding a model is a controlled lifecycle, not a one-shot model import:

```text
analyze → resolve-environment → integrate → verify → card
```

Each phase consumes accepted evidence from the previous phase, produces a reviewable output, and
stops at a mode boundary. This prevents unresolved scientific or runtime choices from being silently
promoted into the public registry.

## 8. The profiling skill

Profiling is independent from onboarding. It starts only from an accepted `runtime_verified` Model
Card and may be repeated on different hardware or execution contexts:

```text
accepted Model Card → plan/resolve → profile → validate → review/promote Technical Card
```

This is why one model can accumulate multiple Technical Cards without changing its accepted identity.

## Practical rule of thumb

When you are unsure where an operation belongs, ask what it is changing:

- **Model identity or runtime contract?** → Model Card / onboarding.
- **Local runtime files?** → environment or checkpoint manager.
- **A tensor produced by the model?** → wrapper API.
- **Performance on a machine/session?** → profiling / Technical Card.
- **Repository history or evidence promotion?** → developer workflow, not normal model inference.
