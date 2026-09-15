# Quickstart

List the currently distributed model cards and validate a synthetic fixture:

```bash
uv run torch-dae card list
uv run torch-dae card validate tests/fixtures/valid/model-card.analyzed.json
uv run torch-dae env --help
uv run torch-dae checkpoint --help
```

The public registry remains empty until checkpoint-specific cards are authored. Card-oriented
environment and checkpoint conveniences require such a card, while the canonical onboarding
workflow can operate card-independently from accepted environment definitions, checkpoint
specifications, and runtime-verification targets. See
{doc}`../tutorials/audio-model-onboarding`.

`model inspect` remains an unavailable-feature placeholder. Runtime verification is available
card-independently through `torch-dae model verify --target <runtime-target.json> [--offline]`.
