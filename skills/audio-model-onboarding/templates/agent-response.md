## Summary
[Two or three concise sentences.]

## Work completed
[Concise description of the completed workflow.]

## Consumed handoff
[Canonical accepted prerequisite handoff path, or None for the first analyze phase.]

## Produced handoff
[Promoted canonical handoff path. For verify, also list target, environment-result, and
checkpoint-specific report paths separately.]

## Evidence separation
[Environment materialization/result/fingerprint and checkpoint-specific target/report associations,
including required/optional check-contract agreement and complete passed required-check coverage.]

## Control-plane provenance
[Historical skill/specification hashes consumed, current skill/specification hashes recorded,
separate drift flags, and any phase transition. State that hashes alone do not reconstruct bytes.]

## Artifact supersessions
[Validated count and affected repository-relative paths, or None.]

## Finalize result
[Absolute paths for the review archive, its SHA-256 sidecar, the bundle result JSON, and the
`finalize-result.json` produced by
`uv run python scripts/onboarding_handoff.py finalize --workflow-id <id> --phase <phase> --json`.
Required-gate pass/fail summary and overall status.]

## Evidence invariance
[Phase-local accepted evidence, current external evidence, declared historical supersessions, and
unexpected-mutation status, as reported by `finalize`.]

## Workspace cleanup
[Cleanup status (`complete`, `dry-run`, `blocked`, or `not_applicable`), removed recorded ephemeral
paths, and retained managed runtime paths. Absolute cleanup receipt path and SHA-256 when one exists.]

## Problems and resolutions
- [Problem]&#58; [Resolution or safe workaround]

## Open questions
- [Question requiring user input]

When nothing remains:

## Open questions
None.

## Files
- [Human-readable file description](repository-relative/path)

## Validation
- [Validation command or check]&#58; [Result]
- Required finalization gates (repository validation, skill validation, staged-equivalent worktree
  validation, `git diff --check`)&#58; [Result]

Only files actually generated, modified, or added for the requested model may be listed.
