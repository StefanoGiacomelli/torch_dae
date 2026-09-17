# Profiling overview

Profiling in `torch-dae` is optional empirical evidence. It starts from an accepted
`runtime_verified` Model Card but is not part of model onboarding and is not required for model
support, release inclusion, or contribution acceptance.

One immutable Model Card may have zero or more immutable Technical Cards for different devices,
platforms, and execution contexts. Profiling never rewrites the Model Card.

Profiling v1 uses deterministic seeded white noise only and measures public-wrapper behavior,
architecture, cold/steady-state latency, throughput, host RAM, accelerator memory where available,
and CodeCarbon-backed energy measurement/estimation when supported.

The executable profiler is not implemented yet. This documentation defines the contract the
implementation must satisfy.

See:

- {doc}`protocol` for measurement methodology;
- {doc}`technical-cards` for evidence and storage semantics;
- {doc}`contributing` for the planned contribution workflow.
