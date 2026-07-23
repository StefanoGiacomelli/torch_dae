# torch-dae

`torch-dae` is a PyTorch audio-model onboarding framework. Phase 00 establishes the repository scaffold, typed control-plane contracts, strict JSON schemas, CLI skeleton, and shared Codex/Claude skill location.

No real model is integrated in Phase 00. The root environment stays lightweight and model-agnostic; model-specific dependencies, materialized environments, checkpoints, reports, and profiling outputs belong under ignored `.torch-dae/` runtime state.

Core invariants:

- one model card describes exactly one model-family, variant, and checkpoint tuple;
- public waveform APIs use `[B,C,T]` plus `sample_rate`, with optional `[B]` valid lengths;
- environment creation is explicit and reproducible from committed specifications;
- each environment spec references one canonical committed `sources.json` manifest;
- checkpoints are cached under `.torch-dae/checkpoints/` and are never committed;
- profiling is reserved for later runtime-verified integrations.

Implemented Phase 00 commands:

```bash
torch-dae card list
torch-dae card show <card-id>
torch-dae card validate <card-id-or-path>
torch-dae env info <card-id>
```

Other command groups are present and fail truthfully until Phase 01 or later.

The canonical onboarding skill lives at `skills/audio-model-onboarding/`. Codex and Claude project skill entries are relative symlinks to that single directory.
