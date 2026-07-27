# Quickstart

List the currently distributed model cards and validate a synthetic fixture:

```bash
uv run torch-dae card list
uv run torch-dae card validate tests/fixtures/valid/model-card.analyzed.json
uv run torch-dae env --help
uv run torch-dae checkpoint --help
```

The public registry is empty in this release. Environment and checkpoint operations require a
committed checkpoint-specific card and its environment specification. To create those artifacts,
use the canonical workflow described in {doc}`../tutorials/audio-model-onboarding`.

The `model inspect` and `model verify` commands are explicit unavailable-feature placeholders; they
do not execute onboarding or inference.
