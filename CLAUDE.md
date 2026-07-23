# Claude Code Instructions

Follow `project_spec.md` as the sole normative source for this repository. Preserve it unless specifically instructed otherwise.

The root package is a lightweight control plane only. Keep model runtime dependencies out of the root environment; PyTorch, TorchAudio, Transformers, TensorFlow, JAX, librosa, checkpoints, and model-specific packages belong in future isolated model environments.

Treat `.torch-dae/` as ignored runtime state and never stage its contents. Each model card is checkpoint-specific: one model family, one variant, one checkpoint.

Use `[B,C,T]` waveform inputs with `sample_rate` for public APIs, and represent optional valid lengths as `[B]`. Record licenses without making automatic legal blocking decisions.

Do not use legacy backbone files. Defer profiling. Preserve unresolved facts explicitly, prefer primary upstream evidence, and do not begin real model onboarding in Phase 00.
