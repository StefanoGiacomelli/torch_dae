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

For Python 3.11, begin with `uv sync --all-groups --frozen --python 3.11` before the unchanged full
suite. Environment dependency-closure regressions exercise wheel markers, extras, normalized names,
version incompatibility, reachable and orphan transitive lock entries, no-network behavior,
pre-materialization ordering, deterministic results, and root dependency isolation.
Package-identity regressions use temporary Git repositories to prove dirty-to-clean and HEAD-only
commit stability while retaining sensitivity to package metadata, README, source, data, and vendored
inputs. Wheel-cache concurrency regressions use spawned processes and real backend builds to prove
single authoritative publication, waiter reuse, failed-builder recovery, bounded stale/live lock
handling, identity-scoped concurrency, and temporary cleanup.

Managed cleanup retains environments and caches by category. `retained_paths` is reserved for
bounded diagnostics already copied below the workflow onboarding-report root. Final Git inventory
uses read-only status/cached-diff checks or staged-equivalent validation and does not require a
real-index `write-tree` operation.

Handoff regressions distinguish immutable accepted history from pending promotion candidates.
Accepted control-plane hashes may drift from current files and are reported without failure;
malformed hashes, invalid lineage, and stale candidate hashes still fail. Atomic-promotion tests must
prove that a control-plane mismatch leaves accepted workflow bytes unchanged, and bundle tests must
assert separate historical/current identities and drift flags.

The final validation matrix begins only after the current handoff and all repository evidence have
been generated. From that point, source, tests, schemas, documentation, environment definitions,
reports, workflows, handoffs, and superseded archives are immutable. Any mutation invalidates the
entire final matrix. Record the final staged-equivalent inventory and current handoff SHA-256 in the
gate logs, compare the inventory again afterward, and generate audit archives read-only outside the
repository.
