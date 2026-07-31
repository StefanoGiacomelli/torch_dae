# Testing

Run the complete local quality and validation sequence:

```bash
uv sync --all-groups --frozen
uv run pytest -q
uv run ruff format --check
uv run ruff check
uv run mypy src scripts
uv run python scripts/generate_schemas.py --check
uv run python scripts/check_worktree_patch.py --json
uv run python scripts/validate_repository.py
uv run python skills/audio-model-onboarding/scripts/validate_skill_artifacts.py . --json
```

`git diff --check` checks tracked unstaged changes, but it does not inspect new untracked files.
`check_worktree_patch.py` builds the complete `git add -A` equivalent in a temporary index and runs
`git diff --cached --check` there. It includes non-ignored additions, modifications, deletions, and
detected renames, respects `.gitattributes`, removes the temporary index, and leaves the real index
unchanged.

Coverage gates require at least 85% line coverage and 70% branch coverage. Synthetic integration
fixtures must not require a network connection, checkpoint download, or model-specific dependency.
