# Contributing to torch-dae

Thank you for helping improve `torch-dae`. Contributions should preserve evidence provenance,
checkpoint-specific identity, reproducibility, and the model-agnostic root control plane.

## Development setup

Install [uv](https://docs.astral.sh/uv/) and synchronize the canonical development environment:

```bash
uv sync --python 3.11 --all-groups --extra profiling --frozen
```

The `profiling` extra is root control-plane tooling (`numpy`, `psutil`, and CodeCarbon), not a
model-runtime dependency.

Do not add PyTorch, TorchAudio, Transformers, TensorFlow, JAX, librosa, checkpoints, or other
model-specific runtime dependencies to the root project. Declare real model dependencies only in a
committed specification under `environments/<environment-id>/`, and materialize them through the isolated
environment subsystem.

## Quality checks

Run formatting, linting, and strict typing:

```bash
uv run --python 3.11 --all-groups --extra profiling --frozen ruff format --check
uv run --python 3.11 --all-groups --extra profiling --frozen ruff check
uv run --python 3.11 --all-groups --extra profiling --frozen mypy src scripts
```

Run tests and the local coverage gates:

```bash
mkdir -p .torch-dae
uv run --python 3.11 --all-groups --extra profiling --frozen pytest -q \
  --cov=torch_dae \
  --cov-branch \
  --cov-report=term-missing \
  --cov-report=json:.torch-dae/coverage.json
uv run --python 3.11 --all-groups --extra profiling --frozen python scripts/check_coverage.py \
  .torch-dae/coverage.json \
  --min-line 85 \
  --min-branch 70
```

Generate and validate schemas, then run both repository validators:

```bash
uv run --python 3.11 --all-groups --extra profiling --frozen python scripts/generate_schemas.py --check
uv run --python 3.11 --all-groups --extra profiling --frozen python scripts/generate_profiling_schema.py --check
uv run --python 3.11 --all-groups --extra profiling --frozen python scripts/validate_repository.py
uv run --python 3.11 --all-groups --extra profiling --frozen python skills/audio-model-onboarding/scripts/validate_skill_artifacts.py . --json
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
- Commit accepted pre-runtime phase artifacts only under `onboarding_reports/<workflow-id>/`; keep
  `verification_reports/` exclusive to checkpoint-specific observations created by `verify`.
- Use recorded `.torch-dae/workspaces/<workflow-id>/<phase>/<run-id>/` state for phase execution,
  promote validated canonical artifacts, generate the deterministic external review bundle, and run
  scoped cleanup.
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

## Profiling evidence

Accepted Model Cards are immutable. Performance evidence (`torch-dae model profile`) is
contributed only as paired `technical_cards/<model-id>/<technical-card-id>.json` and `.npz` assets,
produced with `torch-dae model profile` and checked with
`torch-dae technical-card validate`. A profiling run never writes to `technical_cards/` directly;
candidate evidence lands under an explicit `--output-dir` and is promoted into the official tree
only after independent review.

Do not hand-author Technical Cards, and never modify the referenced Model Card in a profiling
contribution.
