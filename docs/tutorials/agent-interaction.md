# Agent interaction

A precise request identifies the workflow mode, upstream repository, technical reference, target
variant, target checkpoint, and preferred embedding. Use `AUTO_DISCOVER` only when discovery is
within the selected mode; use `UNRESOLVED` when evidence is incomplete.

Review an agent response for:

- evidence paths and authoritative URLs;
- explicit observed, inferred, and unresolved claims;
- environment isolation and exact dependency inputs;
- checkpoint identity and hash evidence;
- wrapper input/output contracts;
- validation commands and remaining decisions.

The canonical response template requires `Summary`, `Work completed`, `Problems and resolutions`,
`Open questions`, `Files`, and `Validation`. A request to analyze is not authorization to integrate,
download a checkpoint, or create a runtime.

In contract terms, **evidence** is an atomic {class}`torch_dae.onboarding.EvidenceItem`; a **source
strategy** is a {class}`torch_dae.onboarding.SourceStrategy`; and ambiguity is carried by
{class}`torch_dae.onboarding.OpenQuestion`. Review the validation flow in
{doc}`../api/onboarding-contracts` and deterministic presentation in
{doc}`../api/report-rendering`.
