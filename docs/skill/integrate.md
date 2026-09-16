# Integrate mode

`integrate` adds a generic public wrapper only after model identity, checkpoint selection,
environment resolution, source strategy, preprocessing, output semantics, and embedding choices are
resolved. The wrapper accepts canonical `[B,C,T]` waveforms, a sample rate in hertz, and optional
`[B]` valid lengths.

Model-specific imports remain lazy and occur only inside controlled construction, verification, or
inference. The root package must stay importable without those dependencies.

With `WORKFLOW_ID`, the mode discovers and validates accepted analyze and resolve-environment
handoffs rather than requesting them again. Accepted integration artifacts are promoted under the
workflow, followed by the canonical `finalize` command, which validates evidence invariance, runs
the required repository gates, runs cleanup, and generates the deterministic review bundle.

Before any integrate-mode environment materialization or reuse, the card-independent dependency
preflight must prove that the accepted lock can satisfy the current local wheel's active runtime
requirements under the locked-sync then `--no-deps` installation policy. The result is deterministic,
network-free, marker-aware, and separate from later environment verification evidence.
The local package identity is content-addressed over wheel build inputs. Git HEAD and cleanliness
are separate informational provenance and must not affect package identity, fingerprinting, cache
reuse, or offline eligibility. Shared deterministic wheel caches are protected by the generic
inter-process manager; callers may materialize environments concurrently and must not rely on
manual serialization.

Before promotion, `uv run python scripts/check_worktree_patch.py --json` must pass. It validates the
complete staged-equivalent working tree through a temporary index, including new untracked outputs,
and leaves the real index untouched. `git diff --check` alone is not a complete gate when the phase
creates files.

If integration changes a shared output declared by an earlier accepted phase, such as an
environment source manifest or verification script, its handoff must declare the exact prior and
new SHA-256 values in `artifact_supersessions`. Promotion validates the pending transition against
the full accepted chain and the current repository file without rewriting the historical handoff.

After the final handoff and repository evidence are generated, run the complete final validation
matrix. Repository mutation is forbidden from that point; any change requires rerunning the whole
matrix. Automated audit archives are generated afterward as a read-only operation and must contain
only files covered by the recorded final staged-equivalent inventory.
