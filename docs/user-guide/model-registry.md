# Model registry

The registry discovers checkpoint-specific JSON cards committed under `model_cards/`. Card IDs are
unique, stable, and map to exactly one model-family, variant, and checkpoint tuple.

Use {class}`torch_dae.ModelCardRegistry` when Python code needs validated card objects or exact
paths. Discovery happens on each call:

```python
from pathlib import Path
from torch_dae import ModelCardRegistry

registry = ModelCardRegistry(Path.cwd())
for card in registry.list_cards():
    print(card.card_id)
```

```bash
uv run torch-dae card list
uv run torch-dae card show <card-id>
uv run torch-dae card validate <card-id-or-path>
```

Duplicate IDs, malformed cards, and invalid wrapper entry points are rejected. Registry discovery
does not import model wrappers or model-specific runtime packages. Unknown IDs raise `KeyError`;
{meth}`~torch_dae.ModelCardRegistry.get_model_class` is the explicit boundary that can raise import,
attribute, or class-type errors.

See {meth}`~torch_dae.ModelCardRegistry.list_cards`,
{meth}`~torch_dae.ModelCardRegistry.get_card`,
{meth}`~torch_dae.ModelCardRegistry.get_card_path`, and
{meth}`~torch_dae.ModelCardRegistry.get_model_class` in {doc}`../api/registry`.
