# Architecture

`torch-dae` separates a lightweight root control plane from isolated model runtimes.

- `torch_dae.cards` validates checkpoint-specific metadata and lifecycle constraints.
- `torch_dae.core` defines generic errors, registry behavior, checkpoint contracts, embeddings,
  preprocessing, and wrapper-output rules.
- `torch_dae.environment` materializes locked, fingerprinted environments under ignored runtime
  state.
- `torch_dae.onboarding` provides deterministic static inspection, evidence-backed reports, strict
  cross-phase handoffs, deterministic review bundles, exact promotion allowlists, and
  receipt-backed managed-workspace cleanup.
- the canonical skill coordinates mode-specific agent work.

Public wrapper modules must import without model-specific dependencies. Heavy dependencies are
loaded lazily during controlled construction, verification, or inference.

- `torch_dae.profiling` defines Profiling v1's typed contracts, identity, synthetic input,
  timing/statistics, minimum-input search, device parsing, privacy validation, hardware/execution-
  context fingerprints, the energy backend abstraction, host/accelerator memory evidence, raw
  `.npz` handling, Technical Card validation, and repository storage listing. Importing this
  package never requires PyTorch, CodeCarbon, or `psutil`; those are imported lazily inside
  individual functions so the same package is safe both in the root control plane and inside a
  model-runtime worker.
- `torch_dae.profiling_worker` runs only inside a model's own materialized environment (invoked as
  `python -I -m torch_dae.profiling_worker <request.json> <result.json>`, mirroring
  `torch_dae.runtime_worker`). It is the only module that imports the model wrapper and PyTorch for
  profiling.
- `torch_dae.profiling_executor` runs in the root process. It materializes/verifies the target
  environment, launches `profiling_worker` subprocesses, wraps them with `psutil`-based host-RSS
  sampling and CodeCarbon energy measurement, and assembles/validates candidate Technical Cards.

The curated surface is recorded in {doc}`../api/index` and
`docs/api/public-api.toml`. Stable subpackage namespaces expose generic contracts; canonical
implementation-module paths remain documented where they make provenance clearer. The registry is
the only root-level service export. Import tests ensure these namespaces do not load prohibited
model-runtime packages.

## Profiling evidence boundary

Profiling is an optional subsystem outside Model Card lifecycle. The implemented profiling layer
consumes accepted, `runtime_verified` Model Cards, executes the public wrapper in the model's own
isolated runtime, and emits immutable candidate Technical Cards plus compact `.npz` raw
observations. Profiler dependencies (CodeCarbon, `psutil`) are the root `profiling` optional
dependency group and are never installed into an accepted model-runtime environment; identity is
kept distinct across three concepts recorded on every Technical Card: the accepted model-runtime
environment fingerprint, the torch-dae package/content identity, and the profiler implementation
version. `torch-dae model profile` never writes into `technical_cards/`; candidate evidence is
written under an explicit `--output-dir`, and promotion into the official tree is a separate, later
step this implementation does not perform.
