# Agent Instructions

`project_spec.md` is the authoritative project specification and must remain unchanged unless the user explicitly asks to edit the spec.

Keep the root control-plane environment model-agnostic. Do not add PyTorch, TorchAudio, Transformers, TensorFlow, JAX, librosa, checkpoints, or model-specific dependencies to the root environment. Real model dependencies belong only in isolated model environments under ignored runtime state.

`.torch-dae/` is ignored runtime state for repositories, builds, materialized environments, checkpoints, reports, and profiling outputs. Do not stage files from that directory.

One model card represents exactly one model-family, variant, and checkpoint tuple. Public waveform inputs use `[B,C,T]` plus `sample_rate`, with optional `[B]` valid lengths. Licenses are informational and non-blocking.

No legacy backbone data is a project input. Profiling is deferred. Unresolved information must remain explicit. Prefer primary upstream evidence. Do not start real model integration during Phase 00.
