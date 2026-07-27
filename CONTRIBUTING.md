# Contributing to torch-dae

Thank you for helping improve `torch-dae`. Contributions should preserve evidence provenance,
checkpoint-specific identity, reproducibility, and the model-agnostic root control plane.

## Development setup

Install [uv](https://docs.astral.sh/uv/) and synchronize all development groups:

```bash
uv sync --all-groups
```

Do not add PyTorch, TorchAudio, Transformers, TensorFlow, JAX, librosa, checkpoints, or other
model-specific runtime dependencies to the root project. Declare real model dependencies only in a
committed specification under `environments/<card-id>/`, and materialize them through the isolated
environment subsystem.

## Quality checks

Run formatting, linting, and strict typing:

```bash
uv run ruff format --check
uv run ruff check
uv run mypy src scripts
```

Run tests and the local coverage gates:

```bash
mkdir -p .torch-dae
uv run pytest -q \
  --cov=torch_dae \
  --cov-branch \
  --cov-report=term-missing \
  --cov-report=json:.torch-dae/coverage.json
uv run python scripts/check_coverage.py \
  .torch-dae/coverage.json \
  --min-line 85 \
  --min-branch 70
```

Generate and validate schemas, then run both repository validators:

```bash
uv run python scripts/generate_schemas.py
uv run python scripts/generate_schemas.py --check
uv run python scripts/validate_repository.py
uv run python skills/audio-model-onboarding/scripts/validate_skill_artifacts.py . --json
```

Build the documentation with warnings treated as errors:

```bash
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
```

Build and validate release distributions without publishing them:

```bash
uv run python -m build
uv run python -m twine check dist/*
```

## Contribution expectations

- Keep one model card scoped to one model family, variant, and checkpoint.
- Preserve the `[B,C,T]` waveform contract and explicit evidence/decision records.
- Keep unresolved information explicit and cite primary upstream evidence where available.
- Use NumPy-style docstrings for curated public APIs.
- Add public API documentation only through the explicit lists under `docs/api/`; do not generate
  recursive model catalogs or expose private helpers.
- Keep public wrapper modules importable without model-specific dependencies. Import heavy runtime
  dependencies lazily during controlled construction, verification, or inference.
- Never commit model or checkpoint binaries; checkpoint assets belong in ignored runtime state.
- Never commit `.torch-dae/`, caches, coverage output, `dist/`, wheels, source distributions, or
  other build artifacts.
- Never add manual PyPI or TestPyPI tokens to repository files or GitHub workflow configuration.
- Add focused tests for behavior and schema changes.
- Regenerate schemas whenever their Pydantic contracts change.
- Keep documentation and canonical skill templates synchronized with public behavior.

## Pull requests

Keep pull requests focused and explain the evidence, compatibility decisions, validation performed,
and any remaining limitations. Confirm that the full quality, test, coverage, schema, repository,
skill, and build checks pass. Do not create commits that include generated runtime state or
model-specific dependencies in the root environment.

## Documentation and releases

The [documentation homepage](docs/index.md) links the user, skill, API, and contributor guides.
Follow the [release guide](docs/development/releasing.md) for validation, GitHub environments,
Trusted Publishing, service setup, and the release workflow. Production publication is initiated
only by publishing a GitHub Release whose tag exactly matches the project version. TestPyPI
publication is manual. Both workflows use OIDC Trusted Publishing and reuse a single validated build.
