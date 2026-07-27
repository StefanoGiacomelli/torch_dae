# Model cards

One model card represents exactly one model-family, architecture variant, and checkpoint. It records
identity, sources, scientific references, tasks, datasets, metrics, environment inputs, waveform and
output contracts, embeddings, capabilities, device support, evidence, issues, and lifecycle state.

Cards progress through `draft`, `analyzed`, `environment_resolved`, `checkpoint_verified`,
`runtime_verified`, and `profiled`. A lifecycle state never substitutes for explicit unresolved
issues or evidence provenance.

A **card** is a strict {class}`torch_dae.cards.ModelCard`, while a **wrapper** is the optional class
named by `identity.wrapper_entry_point`. Read a card without importing its wrapper:

```python
from pathlib import Path
from torch_dae import ModelCardRegistry

card = ModelCardRegistry(Path.cwd()).get_card("example-card")
print(card.identity.variant, card.checkpoint.checkpoint_id)
```

Malformed paths, unresolved evidence IDs, conflicting capabilities/outputs, inconsistent default
embeddings, and missing lifecycle artifacts raise validation errors. License fields are
informational.

Use {doc}`../reference/schemas` for validation commands and {doc}`../reference/lifecycle` for the
state transitions. See {doc}`../api/model-cards` for every field and invariant and
{meth}`torch_dae.ModelCardRegistry.get_card` for the main read path.
