# Profiling overview

Profiling in `torch-dae` is optional empirical evidence. It starts from an accepted
`runtime_verified` Model Card but is not part of model onboarding and is not required for model
support, release inclusion, or contribution acceptance.

One immutable Model Card may have zero or more immutable Technical Cards for different devices,
platforms, and execution contexts. Profiling never rewrites the Model Card.

Profiling v1 uses deterministic seeded white noise only and measures public-wrapper behavior,
architecture, cold/steady-state latency, throughput, host RAM, accelerator memory where available,
and CodeCarbon-backed energy measurement/estimation when supported.

The executable profiler is implemented (`torch-dae model profile`; see
`src/torch_dae/profiling/` and `src/torch_dae/profiling_executor.py`). Real profiling output is
always candidate evidence under a repository-local candidate/workspace path of your choosing
(`--output-dir`) until it is independently reviewed and promoted into
`technical_cards/<model-id>/`; this implementation never performs that promotion itself.

See:

- {doc}`protocol` for measurement methodology;
- {doc}`technical-cards` for evidence and storage semantics;
- {doc}`contributing` for the planned contribution workflow.
