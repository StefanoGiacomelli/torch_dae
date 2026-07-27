# Architecture

`torch-dae` separates a lightweight root control plane from isolated model runtimes.

- `torch_dae.cards` validates checkpoint-specific metadata and lifecycle constraints.
- `torch_dae.core` defines generic errors, registry behavior, checkpoint contracts, embeddings,
  preprocessing, and wrapper-output rules.
- `torch_dae.environment` materializes locked, fingerprinted environments under ignored runtime
  state.
- `torch_dae.onboarding` provides deterministic static inspection and evidence-backed reports.
- the canonical skill coordinates mode-specific agent work.

Public wrapper modules must import without model-specific dependencies. Heavy dependencies are
loaded lazily during controlled construction, verification, or inference. Profiling remains outside
the current implemented workflow.

The curated surface is recorded in {doc}`../api/index` and
`docs/api/public-api.toml`. Stable subpackage namespaces expose generic contracts; canonical
implementation-module paths remain documented where they make provenance clearer. The registry is
the only root-level service export. Import tests ensure these namespaces do not load prohibited
model-runtime packages.
