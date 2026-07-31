# Integrate mode

`integrate` adds a generic public wrapper only after model identity, checkpoint selection,
environment resolution, source strategy, preprocessing, output semantics, and embedding choices are
resolved. The wrapper accepts canonical `[B,C,T]` waveforms, a sample rate in hertz, and optional
`[B]` valid lengths.

Model-specific imports remain lazy and occur only inside controlled construction, verification, or
inference. The root package must stay importable without those dependencies.

With `WORKFLOW_ID`, the mode discovers and validates accepted analyze and resolve-environment
handoffs rather than requesting them again. Accepted integration artifacts are promoted under the
workflow, bundled, and followed by scoped cleanup.

Before promotion, `uv run python scripts/check_worktree_patch.py --json` must pass. It validates the
complete staged-equivalent working tree through a temporary index, including new untracked outputs,
and leaves the real index untouched. `git diff --check` alone is not a complete gate when the phase
creates files.

If integration changes a shared output declared by an earlier accepted phase, such as an
environment source manifest or verification script, its handoff must declare the exact prior and
new SHA-256 values in `artifact_supersessions`. Promotion validates the pending transition against
the full accepted chain and the current repository file without rewriting the historical handoff.
