# Model Card Authoring

Cards are final checkpoint-specific evidence consumers: one model family, one variant, one
checkpoint. They are not prerequisites for environment materialization or runtime verification.
Every populated field
must trace to upstream source, paper or official docs, environment evidence, checkpoint evidence,
runtime observation, or explicit user decision.

Use nulls, unresolved states, TODO markers where allowed, and explicit issues when evidence is
absent. A verified environment claim requires a matching promoted environment-verification result
with `verification_status=passed`, `lifecycle_state=environment_verified`, no failure
classification, and the exact fingerprint. A runtime-verified claim requires a matching runtime
target and checkpoint-specific report with successful overall status, exact required/optional check
contract agreement, complete passed required-check coverage, and no duplicate or undeclared check.
Validate their hashes and model/checkpoint/environment/integration associations through Pydantic,
generated JSON Schema, and repository validation. Failed evidence remains diagnostic and never
promotes the card lifecycle.
