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
loaded lazily during controlled construction, verification, or inference. Profiling remains outside
the current implemented workflow.

The curated surface is recorded in {doc}`../api/index` and
`docs/api/public-api.toml`. Stable subpackage namespaces expose generic contracts; canonical
implementation-module paths remain documented where they make provenance clearer. The registry is
the only root-level service export. Import tests ensure these namespaces do not load prohibited
model-runtime packages.

## Profiling evidence boundary

Profiling is an optional subsystem outside Model Card lifecycle. The planned profiling layer will
consume accepted Model Cards, execute the public wrapper in the appropriate isolated runtime, and
emit immutable Technical Cards plus compact `.npz` raw observations. Additional profiler
dependencies such as CodeCarbon must not silently mutate accepted onboarding environment
definitions.
