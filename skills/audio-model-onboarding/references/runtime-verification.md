# Runtime Verification

Verification starts from a strict `RuntimeVerificationTarget`, never a provisional model card. The
target binds the accepted integrate handoff, integrated model identity, checkpoint, environment,
source manifest, future card identity, public entry point, expected input/output/probability and
embedding contract, permitted devices, acquisition policy, execution limits, unresolved items, and
ordered required and optional check IDs. Completeness-aware targets use schema `2.0.0`; required IDs
are nonempty, canonical, and unique, optional IDs are canonical and unique, and the sets are
disjoint. Schema `1.0.0` targets remain readable only as legacy requests and do not satisfy new
completeness-aware promotion.

Resolve and materialize the target environment directly by environment ID. Run infrastructure-only
environment verification and record its fingerprint before checkpoint acquisition. Promote the
normalized environment result under `onboarding_reports/<workflow-id>/verify/`; the promoted result
must retain `verification_status=passed`, `lifecycle_state=environment_verified`, and the matching
fingerprint. Do not place it in `verification_reports/`. Managed command logs and materialization
records are not substitutes for this promoted canonical result on a final card. Successful
environment evidence also has nonempty import and smoke observations, every observation passed, and
unique names across both collections.

Checkpoint-specific verification then covers construction, checkpoint loading, invalid checkpoint behavior, model variant
agreement, inputs, outputs, embeddings, device movement, dtypes, NaN/Inf checks, and repeated-call
behavior.

The structured target-aware report must include explicit overall `verification_status`,
target/workflow/integration identity, environment specification and source-manifest hashes,
environment fingerprint, checkpoint identity and hash, source revision, package identity, test
inputs, observed outputs, embedding observations, warnings, failures, unsupported capabilities, and
runtime evidence.

Schema validity and report existence are insufficient for `runtime_verified`. The report must have
`verification_status=passed`, contain nonempty canonical unique checks, cover every target-required
check exactly once with passed status, contain no undeclared check, and match its target's exact
required/optional check contracts plus model,
variant, checkpoint, environment, source, integration handoff, future card, and public entry point.
Required checks cannot be unsupported. Optional checks remain non-failing only when target-declared;
an unsupported optional check requires nonempty details, a matching capability entry, and a recorded
limitation. Failed evidence remains diagnostic and never promotes lifecycle. Legacy `1.0.0` reports
remain readable conservatively; any failed check makes a legacy report unsuccessful, and legacy
reports cannot promote a new runtime-verified card. Promote an accepted verify handoff only after
those checks pass.

A successful result proves both successful global status and complete successful coverage of the
evidence required by its authority contract.
