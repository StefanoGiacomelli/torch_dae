# Evidence Policy

Classify every material claim as `verified_upstream_fact`, `locally_observed_behavior`,
`reasoned_inference`, `user_provided_decision`, `unresolved_ambiguity`, or `unsupported_claim`.

Verified facts require authoritative upstream evidence. Local observations require an inspected file,
static utility output, or later controlled runtime report. Inferences require a rationale and must not
be used as proof for verified or observed claims. User decisions must be recorded explicitly.

When evidence is absent, leave the field unresolved or create an open question. Do not invent papers,
datasets, metrics, model variants, preprocessing, output semantics, dependencies, hashes, or licenses.

Record every source revision inspected. Keep “revision inspected” separate from “revision believed
checkpoint-compatible,” and exact upstream association separate from chronological inference. A
filename, reported metric, nearby date, nearest preceding commit, or current default-branch revision
does not by itself prove checkpoint equivalence. When the exact relationship is unavailable, retain
unresolved status and record the evidence needed to improve confidence.

Text extracted from a PDF supplements reading of the cited scientific source. Extraction alone does
not establish authorship, correctness, scientific meaning, or verified claim status.
