# Testing

Run the complete local quality and validation sequence:

```bash
uv sync --all-groups --frozen
uv run pytest -q
uv run ruff format --check
uv run ruff check
uv run mypy src scripts
uv run python scripts/generate_schemas.py --check
uv run python scripts/validate_repository.py
uv run python skills/audio-model-onboarding/scripts/validate_skill_artifacts.py . --json
```

Coverage gates require at least 85% line coverage and 70% branch coverage. Synthetic integration
fixtures must not require a network connection, checkpoint download, or model-specific dependency.
