# Curated generic API

This reference is generated from the intentionally reviewed symbols in
[`public-api.toml`](public-api.toml). The manifest is the single machine-readable source for API
documentation and consistency tests.

The three audiences are:

- **users**, who discover cards, inspect metadata, obtain outputs, and manage local runtime state;
- **integrators**, who implement the waveform, checkpoint, embedding, card, and environment
  contracts;
- **developers**, who maintain evidence processing, static inspection, and deterministic reports.

These pages document stable generic contracts and explicitly selected supporting types. Internal
helpers remain implementation details. Model-specific wrappers are opt-in and must be documented
manually. No recursive module scan, model-directory scan, generated wrapper page, or model catalog
is produced.

```{toctree}
:maxdepth: 1

registry
model-execution
outputs-embeddings
capabilities
checkpoints
model-cards
environment-specifications
environment-lifecycle
runtime-verification
onboarding-contracts
onboarding-inspection
report-rendering
```

## Recurring terms

card
: Validated JSON contract for exactly one model-family, variant, and checkpoint tuple.

checkpoint
: A pretrained byte asset plus its source, loader, format, and integrity evidence.

embedding
: A named intermediate or pooled tensor with declared semantics and layout.

environment
: An isolated, fingerprinted Python runtime built from committed inputs.

evidence
: A traceable source supporting a fact, observation, inference, decision, or unresolved claim.

source strategy
: The evidence-backed way implementation code enters an isolated environment.

verification
: Local checks recorded as runtime evidence; it is narrower than upstream scientific validity.

wrapper
: An opt-in model implementation satisfying the generic execution protocol.

lifecycle
: The validated progression of a card as evidence and artifacts become available.
