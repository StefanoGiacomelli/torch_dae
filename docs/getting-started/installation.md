# Installation

## Requirements

Use Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/). The root environment intentionally
contains no audio-model runtime frameworks.

## Package index

The distribution name is intentionally distinct from the software title, import package, and
console command:

```bash
pip install torch-deepaudioembedding
python -c "import torch_dae"
torch-dae --help
```

## Source checkout

```bash
git clone https://github.com/StefanoGiacomelli/torch_dae.git
cd torch_dae
uv sync --all-groups
```

Verify the package and CLI:

```bash
uv run python -c "import torch_dae; import torch_dae.onboarding"
uv run torch-dae --help
```

For documentation-only installation from a source checkout, install `.[docs]`. Model-specific
dependencies belong only to isolated environments described in {doc}`../user-guide/environments`.
