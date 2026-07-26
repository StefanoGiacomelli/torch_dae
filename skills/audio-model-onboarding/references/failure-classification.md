# Failure Classification

Resolve-environment failures must be classified with a specific cause, including:
`python_constraint`, `dependency_conflict`, `resolution_failure`, `removed_api`, `deprecated_api`,
`binary_or_abi_incompatibility`, `missing_binary_wheel`, `torch_torchaudio_mismatch`,
`numpy_compatibility`, `checkpoint_incompatibility`, `source_build_failure`, `import_failure`,
`runtime_failure`, `platform_incompatibility`, `access_or_authentication_blocker`, and
`insufficient_evidence`.

The next candidate must be motivated by evidence or failure diagnostics.

Licenses are informational and non-blocking. Missing, ambiguous, or restrictive license text must be
recorded as evidence or an open question, but it must not automatically classify a model as
unsupported.
