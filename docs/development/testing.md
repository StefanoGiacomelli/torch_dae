# Testing

Run the complete local quality and validation sequence:

```bash
uv sync --python 3.11 --all-groups --extra profiling --frozen
uv run --python 3.11 --all-groups --extra profiling --frozen pytest -q
uv run --python 3.11 --all-groups --extra profiling --frozen ruff format --check
uv run --python 3.11 --all-groups --extra profiling --frozen ruff check
uv run --python 3.11 --all-groups --extra profiling --frozen mypy src scripts
uv run --python 3.11 --all-groups --extra profiling --frozen python scripts/generate_schemas.py --check
uv run --python 3.11 --all-groups --extra profiling --frozen python scripts/generate_profiling_schema.py --check
uv run --python 3.11 --all-groups --extra profiling --frozen python scripts/check_worktree_patch.py --json
uv run --python 3.11 --all-groups --extra profiling --frozen python scripts/validate_repository.py
uv run --python 3.11 --all-groups --extra profiling --frozen python skills/audio-model-onboarding/scripts/validate_skill_artifacts.py . --json
```

The `profiling` extra (`numpy`, `psutil`, `codecarbon`) is a root optional-dependency group for the
Profiling v1 control-plane tooling itself -- not a model-runtime dependency -- so it is safe to
install alongside the base control plane; see `pyproject.toml`'s `[project.optional-dependencies]`.
Omitting `--extra profiling` will fail to collect `tests/profiling/`, `tests/cli/test_profiling_cli.py`,
and `tests/test_profiling_checkpoint_reuse.py`, and will change `mypy`'s treatment of the profiling
modules' `numpy` imports.

`git diff --check` checks tracked unstaged changes, but it does not inspect new untracked files.
`check_worktree_patch.py` builds the complete `git add -A` equivalent in a temporary index and runs
`git diff --cached --check` there. It includes non-ignored additions, modifications, deletions, and
detected renames, respects `.gitattributes`, removes the temporary index, and leaves the real index
unchanged.

Coverage gates require at least 85% line coverage and 70% branch coverage. Synthetic integration
fixtures must not require a network connection, checkpoint download, or model-specific dependency.
The PANNs wrapper and its verification provider run only in managed model environments; both are
excluded from the root control-plane coverage denominator. Their actual runtime evidence comes from
the target executor. Generic worker dispatch, report validation and orchestration remain covered
by the root suite.

Every Python 3.11 gate uses `uv run --python 3.11 --all-groups --frozen ...`;
selecting an interpreter only at sync time does not constrain later `uv run` commands.
For the Python 3.12 leg, use the same commands with `--python 3.12`.
Twine 7 validates Core Metadata 2.5 emitted by the current build backend without changing
package metadata semantics. See the [Twine changelog](https://twine.readthedocs.io/en/latest/changelog.html). Environment dependency-closure regressions exercise wheel markers, extras, normalized names,
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
