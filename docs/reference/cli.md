# CLI reference

```text
torch-dae card list
torch-dae card show <card-id>
torch-dae card validate <card-id-or-path>

torch-dae env create <card-id>
torch-dae env ensure <card-id>
torch-dae env resolve <environment-id>
torch-dae env preflight <environment-id>
torch-dae env materialize <environment-id>
torch-dae env verify <environment-id>
torch-dae env remove <card-id>
torch-dae env info <card-id>
torch-dae env run <card-id> -- <command>

torch-dae checkpoint ensure <card-id>
torch-dae checkpoint info <card-id>
torch-dae checkpoint remove <card-id>
torch-dae checkpoint resolve --spec <checkpoint-spec.json> [--offline] [--json]
torch-dae checkpoint ensure-spec --spec <checkpoint-spec.json> [--offline] [--maximum-bytes N] [--json]
torch-dae checkpoint info-spec --spec <checkpoint-spec.json> [--json]
```

The `--spec` commands are card-independent. `resolve` is metadata-only, `ensure-spec` uses the same
canonical manager primitive as card-based `ensure`, and `info-spec` inspects authority/cache state
without network access.

Run `uv run torch-dae <group> --help` for option details. Model inspection and verification CLI
entries are unavailable-feature placeholders in this release.

`env preflight` is card-independent and network-free. It validates active local-wheel
`Requires-Dist` requirements against packages reachable from the accepted environment lock and
returns structured missing or incompatible requirement evidence without creating an environment.

Onboarding handoff management is a root control-plane script:

```text
uv run python scripts/onboarding_handoff.py discover ...
uv run python scripts/onboarding_handoff.py validate ...
uv run python scripts/onboarding_handoff.py promote ...
uv run python scripts/onboarding_handoff.py bundle ...
uv run python scripts/onboarding_handoff.py cleanup ...
```

`discover` searches committed `onboarding_reports/` only. `promote` accepts exactly
`workflow.json`, the selected `handoff.json`, and its declared phase-local outputs from managed
workspace content; external repository outputs are hash-validated in place. Promotion is atomic.
Valid later-phase artifact supersessions are resolved before promotion; failures leave the accepted
workflow unchanged. `discover` reports superseded external outputs when an earlier accepted phase
is requested after the later phase has been accepted.
`discover` and `validate` also report accepted historical skill/specification hashes beside current
values. Separate drift flags are informational for accepted handoffs. `promote` still rejects a
pending candidate unless both hashes match the current repository exactly.

`bundle` creates a normalized external audit archive, an explicit result JSON, and a matching
`<archive>.sha256` sidecar. The result reports archive cleanliness, TAR and gzip metadata
normalization, and declared/actual inventory agreement.
It also reports the validated supersession count and paths, and includes explicit supersession and
latest-external-artifact metadata. Historical control-plane identities for every included phase,
current identity, and separate drift flags are present in both result and archive metadata.

`cleanup` deletes only recorded, selected managed paths. Every invocation atomically writes a
durable receipt under `.torch-dae/reports/onboarding/<workflow-id>/cleanup/` and returns its path and
SHA-256. Retained diagnostics must first be moved under that workflow diagnostics root. A retained
child below a deletion root blocks the whole execution and is reported in `retention_conflicts`.
Existing external audit paths are protected before mutation when their supplied path or resolved
target overlaps a deletion root in either direction; the receipt reports these conflicts in
`external_protection_conflicts`. Non-conflicting external paths are never selected for deletion and
distinguish retained files, directories, symlinks, and missing outputs. Cleanup JSON with errors is
emitted before the command exits unsuccessfully.

`retained_paths` is only for bounded diagnostics already copied under the workflow onboarding-report
root. Environments and caches are retained by category and are not redundantly declared as retained
diagnostics. Under intentionally non-writable Git metadata, final inventory uses read-only status and
cached-diff checks or the staged-equivalent validator rather than `git write-tree`.
