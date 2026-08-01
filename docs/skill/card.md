# Card mode

`card` produces one final checkpoint-specific model card from accepted analysis, environment,
integration, environment-verification, runtime-target, and checkpoint-runtime evidence. The card is
an evidence consumer; it is never required to materialize an environment or create runtime
evidence.
It does not combine incompatible variants or checkpoints. Officially reported facts, local
observations, inferences, unresolved items, and non-applicable fields remain distinguishable.

The card validator enforces successful overall evidence status, canonical paths, hashes,
fingerprints, and model/checkpoint/environment/source/integration associations across its promoted
environment result, runtime target, and verification report. It also requires exact target/report
check-contract agreement and complete passed required-check coverage. A failed, unsupported,
undeclared, duplicate, or missing required check remains diagnostic or invalid and cannot promote
the card lifecycle.

With `WORKFLOW_ID`, accepted prerequisite handoffs and their recorded decisions are discovered
locally. The accepted card handoff is promoted under `onboarding_reports/<workflow-id>/card/`,
bundled, and followed by scoped cleanup.
