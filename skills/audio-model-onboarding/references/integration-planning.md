# Integration Planning

The integration plan must specify wrapper path, source strategy, checkpoint strategy, construction,
random and checkpoint initialization, preprocessing ownership, sample-rate behavior, channel
behavior, waveform scaling, padding/truncation, valid lengths, forward signature, output
normalization, task capabilities, embeddings, device movement, evaluation mode, deterministic
behavior, tests, and errors.

Production integration is permitted only in an explicitly requested `integrate` mode after analysis
review, source/variant/checkpoint/embedding decisions, environment strategy resolution, and explicit
user authorization. Integration must remain scoped to the selected model and must not start
verification, add root model dependencies, commit checkpoints, or create a Git commit.

Before phase completion, run `uv run python scripts/check_worktree_patch.py --json`. The command
constructs the `HEAD` plus complete working-tree patch in a temporary index, so tracked changes,
untracked non-ignored outputs, deletions, renames, and `.gitattributes` rules receive the same
whitespace check as a future staged commit. `git diff --check` is still useful for tracked unstaged
changes, but it is insufficient as the only gate when integration creates untracked files. Never
stage the real index for this validation.

When integration legitimately updates an earlier phase's shared repository output, its handoff must
declare the exact prior and new hashes through `artifact_supersessions` and declare the path as a
current integrate output. Do not rewrite the earlier handoff or use supersession for protected
canonical, specification, schema, credential, checkpoint, or runtime artifacts.
