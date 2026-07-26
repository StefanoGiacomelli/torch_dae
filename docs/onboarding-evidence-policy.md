# Onboarding Evidence Policy

Every material onboarding claim is classified as one of:

- `verified_upstream_fact`
- `locally_observed_behavior`
- `reasoned_inference`
- `user_provided_decision`
- `unresolved_ambiguity`
- `unsupported_claim`

Verified facts need authoritative upstream evidence from official documentation, papers, inspected
upstream source or configuration, source symbols, or package metadata. Runtime observations,
agent inference, and user decisions cannot prove `verified_upstream_fact`, regardless of the status
written on the evidence item. Local observations need inspected files, deterministic static utility
output, or later controlled runtime reports. Inferences require rationale and must not be cited as
verified evidence. User decisions are explicit evidence items.

The same semantic compatibility rules apply inside environment candidate-generation results:
dependency records must cite local or runtime observations, source-strategy candidates cannot cite
agent inference as verified proof, reasoned environment candidates require a nonempty rationale and
factual or local evidence, and user decisions must cite `user_decision` evidence.

When evidence is missing, the correct output is an unresolved item, decision request, or unsupported
claim. The workflow must not invent papers, datasets, metrics, variants, checkpoints, hashes,
preprocessing, output semantics, embeddings, dependencies, or licensing conclusions.

Licenses are informational and non-blocking. Ambiguous license evidence remains an explicit open
question; it must not automatically classify a model or source strategy as unsupported.

Static inspection treats repositories as untrusted input. Inspectors use bounded reads, reject binary
or disguised text where text is required, do not execute upstream code, and refuse symlinked fixed
input files such as package metadata or Python source. Dependency evidence is normalized with
provenance, but candidate environments remain unresolved unless every exact version and source
strategy is supported by compatible evidence. Invalid dependency records remain diagnostic only and
do not influence exact-version selection, constraint merging, conflicts, or principal dependencies.

Onboarding evidence paths use a dedicated repository-relative contract. Paths such as
`.github/workflows/ci.yml` are preserved exactly, while absolute paths, traversal, `.git/`,
`.torch-dae/`, `.venv/`, `venv/`, and external symlink targets are rejected.
