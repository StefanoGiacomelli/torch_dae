# Verify a runtime target

Verify a model/checkpoint/environment tuple before writing a model card:

```bash
uv run --python 3.11 --all-groups --frozen torch-dae model verify \
  --target onboarding_reports/<workflow>/verify/runtime-targets/<target>.json --json
```

Add `--offline` to require retained environments, packages, checkpoint bytes and authoritative
metadata. The checkpoint manager validates cached bytes and provenance before reuse. Source changes
may require a new content-addressed model environment even when checkpoint bytes remain reusable.

The schema-2 target binds the accepted integrate handoff, environment/source hashes, public entry
point, checkpoint loader and integrity policy, bounded inputs and ordered required/optional checks.
Future card identity is an evidence association; no card is loaded. The primitive validates complete workflow history and these target associations. Accepted generic
control-plane file hashes retain their historical meaning through the existing explicit artifact
classification; model wrapper, source and environment artifacts remain subject to current-byte
and accepted-supersession validation.

`execute_runtime_verification(target, repository_root, offline=False)` in
`torch_dae.runtime_executor` returns a `RuntimeVerificationExecution` containing the validated report,
report path, environment result, checkpoint and retained evidence directory. Infrastructure errors
raise; runtime check failures return a failed report. The CLI exits 0 for passed reports, 1 for failed
runtime checks and 2 for invalid requests or infrastructure failures.

Runtime reports and commands initially live under `.torch-dae/reports/runtime/<target>/<run>/`.
Onboarding verify promotes target files and environment results into its phase tree, copies
checkpoint-specific reports to `verification_reports/`, and promotes all requested tuples together
only when required checks pass. Failed evidence remains available for diagnosis.

## Provider interface

For `package:PublicModel`, the managed child imports `package.verification.Provider`. The constructor
receives `(model_class, target, checkpoint_path)` and exposes `check(name)`, `tensor_observations` and
`embedding_results`. The model class always comes from the declared public entry point. Providers
implement declared loader contracts using the integrated public wrapper, enforce strict loading, and
keep forward checks dependent on successful loading. They must honor target input/resource limits.

The worker calls each required then optional check exactly once in target order. Exceptions produce
failed checks. `UnsupportedCapabilityError` produces an unsupported optional check with details and
a recorded limitation; it fails a required check. Unknown required checks fail explicitly. Environment,
checkpoint integrity and optional offline-cache checks are owned by the generic executor. Tensor
observations must cover every declared output and default embedding for success. No model dependency
is imported by the root control plane. The child uses the verified environment interpreter with
isolated Python imports, a timeout and native MPS fallback disabled.

The PANNs provider observes checkpoint-loaded CPU behavior through the existing wrapper and vendored
forward path. It does not reproduce published AudioSet mAP or establish the historical training-source
revision. Unsupported resampling, padded variable lengths and native MPS operations remain explicit.
