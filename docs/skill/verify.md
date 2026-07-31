# Verify mode

`verify` performs controlled checkpoint acquisition, checksum validation, state loading, forward
inference, probability checks where supported, declared embedding checks, device behavior, and
gradient behavior. Results are written as explicit runtime observations.

A successful import alone is not runtime verification. Every declared public output and lifecycle
precondition must be checked.

With `WORKFLOW_ID`, accepted prerequisite handoffs are discovered locally. `verify` writes only
checkpoint-specific runtime observations under `verification_reports/`; pre-runtime analysis,
environment, integration, handoff, and audit artifacts remain outside that root. The review bundle
and scoped cleanup still close the run.
