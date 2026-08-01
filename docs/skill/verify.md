# Verify mode

`verify` consumes the accepted integrate handoff and first creates strict runtime-verification
targets that bind the model/variant, checkpoint, environment, source, public entry point, future
card identity, expected contract, device scope, acquisition policy, limits, and ordered required and
optional check IDs. It resolves,
materializes, and verifies each environment by environment ID before any checkpoint operation.

It then performs controlled checkpoint acquisition, checksum validation, state loading, forward
inference, probability checks where supported, declared embedding checks, device behavior, and
gradient behavior. Results are written as explicit runtime observations.

A successful import alone is not runtime verification. Every target-required check and declared
public output must be observed exactly once and pass. Optional checks remain bounded diagnostics and
may be unsupported only with explicit details and a recorded limitation.

No model card is required. Targets and normalized environment-verification evidence are promoted in
`onboarding_reports/<workflow-id>/verify/`; promoted environment evidence must retain passed status
and its matching fingerprint. Only checkpoint-specific runtime observations are written under
`verification_reports/`, and new target-aware reports declare explicit overall status plus the
target's exact required and optional check contract. Passed reports cannot be empty. Failed
environment or runtime evidence remains diagnostic and cannot close or promote lifecycle. The
accepted verify handoff, review bundle, and scoped cleanup close the run.
