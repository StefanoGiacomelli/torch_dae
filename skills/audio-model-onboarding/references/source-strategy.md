# Source Strategy

Priority order:

1. Official package: official, exact version lockable, required API exposed, behavior equivalent, and
   no uncontrolled runtime download.
2. Pinned official Git repository: official repo, immutable revision, reproducible build, required API
   importable without modification.
3. Minimal vendored adaptation: package/Git insufficient, copied files minimal, provenance explicit,
   revision pinned, changes documented, semantics preserved, tests cover copied files.
4. External PyTorch implementation: only when official upstream is not PyTorch and equivalence can be
   justified; request a user decision when uncertain.

Unsupported or non-equivalent implementations must terminate truthfully.
