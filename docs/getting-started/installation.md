# Installation

`torch-dae` has two intentionally different installation modes. Choose the one that matches what
you want to do.

## Requirements

The root package supports Python 3.11 and 3.12. Source-development commands use
[uv](https://docs.astral.sh/uv/).

The root environment is deliberately model-agnostic: it does not install PyTorch, TorchLibrosa, or
other heavyweight model-runtime frameworks as mandatory package dependencies. Accepted models carry
their own isolated environment definitions.

## Option 1 — install the published package

Use the package-index distribution when you need the `torch_dae` Python package and `torch-dae`
control-plane CLI without cloning the repository:

```bash
pip install torch-deepaudioembedding
python -c "import torch_dae"
torch-dae --help
```

The names are intentionally different:

| Purpose | Name |
|---|---|
| PyPI distribution | `torch-deepaudioembedding` |
| Python import | `torch_dae` |
| CLI command | `torch-dae` |

### What the wheel contains

The wheel contains the Python package, including the generic contracts, managers, CLI, and model
wrapper source code.

### What the wheel does not contain

The repository-backed workflow depends on committed artifacts that are not packaged into the wheel:

- `model_cards/`
- `environments/`
- `verification_reports/`
- `technical_cards/`
- `onboarding_reports/`
- `skills/`
- `project_spec.md`

Model checkpoint payloads are never redistributed by the repository or wheel.

For registry discovery, managed model environments, checkpoint acquisition by accepted Model Card,
runtime verification, profiling, or the agent skills, use a full source checkout.

## Option 2 — full repository workspace

Clone the repository and create the root control-plane environment:

```bash
git clone https://github.com/StefanoGiacomelli/torch_dae.git
cd torch_dae
uv sync --python 3.11 --all-groups --extra profiling --frozen
```

Verify the control plane:

```bash
uv run python -c "import torch_dae; import torch_dae.onboarding"
uv run torch-dae --help
uv run torch-dae card list
```

The `profiling` extra installs root profiling support such as NumPy, `psutil`, and CodeCarbon. It
does **not** install the runtime dependencies of every audio model.

## How model runtimes are installed

A supported Model Card points to a committed environment definition. The environment manager builds
that runtime under ignored `.torch-dae/` state.

For example:

```bash
uv run torch-dae env ensure panns-cnn14-16k-map-0438
```

The current accepted PANNs environment definitions are constrained to macOS on arm64. Static card
inspection remains available on other platforms, but managed PANNs materialization is rejected when
the host platform is outside the accepted environment constraints.

Checkpoint acquisition is separate:

```bash
uv run torch-dae checkpoint ensure panns-cnn14-16k-map-0438
```

This separation is intentional:

```text
root control plane
    ├── cards / registry / validation / acquisition logic
    └── lightweight development tooling

model-specific environment
    ├── exact Python/runtime dependencies
    ├── torch / model DSP dependencies
    └── wrapper execution

checkpoint cache
    └── explicitly acquired pretrained bytes
```

## Offline reuse

After an environment and checkpoint have been materialized successfully, compatible commands can be
run with their `--offline` option to require local reuse instead of network access. An offline cache
miss fails explicitly rather than silently falling back to the network.

## Documentation-only environment

For documentation work from a source checkout, the project also defines a `docs` dependency group.
The repository's canonical documentation validation command is:

```bash
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
```

## Next step

Continue with {doc}`mental-model` if you are new to the architecture, or go directly to
{doc}`quickstart` to inspect the accepted model registry.
