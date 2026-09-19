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

torch-dae model verify --target <runtime-target.json> [--offline] [--json]
torch-dae model profile --model <card-id> [--device auto|cpu|mps|cuda|cuda:<index> ...] \
  [--protocol audio-inference-v1] [--energy auto|off] [--allow-privileged-energy] \
  [--output-dir ./candidate_technical_cards] [--json]

torch-dae technical-card validate <technical-card.json> [--json]
torch-dae technical-card inspect <technical-card.json>
torch-dae technical-card list [--json]
```

The `--spec` commands are card-independent. `resolve` is metadata-only, `ensure-spec` uses the same
canonical manager primitive as card-based `ensure`, and `info-spec` inspects authority/cache state
without network access.

Run `uv run torch-dae <group> --help` for option details. `model inspect` remains an
unavailable-feature placeholder in this release; `model verify` and `model profile` are
implemented.

`model profile` implements Profiling v1 (`docs/profiling/protocol.md`). It requires the target
Model Card to be `runtime_verified`, materializes/verifies its accepted environment, smoke-tests
every requested device, and writes candidate Technical Card JSON/`.npz` evidence under
`--output-dir`. That path must resolve to a candidate/workspace directory inside the repository;
the repository root, official `technical_cards/` tree, descendants of that tree, and external
paths are rejected before profiling side effects. Profiling never mutates the Model Card;
candidate promotion is a separate, later step. `--energy auto` requires the root
`profiling` optional dependency group (`uv sync --extra profiling`, adding CodeCarbon and
`psutil`); without it, energy evidence is reported as `unavailable` rather than failing the run.
On platforms where CodeCarbon requires privileged local hardware counters, such as Apple
PowerMetrics, the privileged path is permitted only with explicit `--allow-privileged-energy`;
otherwise profiling continues with non-privileged energy evidence.

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
uv run python scripts/onboarding_handoff.py finalize ...
uv run python scripts/onboarding_handoff.py run-manifest create ...
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

A workflow whose `.torch-dae/workspaces/<workflow-id>/` scope has no phase/run directories at all is
the genuinely legacy case (for example a historical workflow completed before this workflow adopted
run manifests) and is not an error: `cleanup` returns a structured
`{"status": "not_applicable", "reason": "no_managed_run_manifests", "removed": []}` result and exits
`0`. A run directory that *is* present but carries no `run-manifest.json` of its own is unmanaged
content, not the legacy case: `cleanup` reports it in `errors` and blocks the whole operation (no
deletion happens) rather than guessing ownership or silently treating it as `not_applicable`.

`run-manifest create` is the single canonical *allocator* for a managed workspace run, not an
after-the-fact registration step. It atomically creates
`.torch-dae/workspaces/<workflow-id>/<phase>/<run-id>/` and its `run-manifest.json` together and
prints the absolute `run_root`/`manifest_path`. Call it before performing any managed workspace
work and perform that work inside the returned `run_root` — never create an independent workspace
directory first and register it afterward. It always records the run's own workspace directory as a
created path; pass `--created-path`, `--reused-path`, and `--external-path` to declare additional
paths as they are created.

`finalize` is the one canonical end-of-task path for a completed lifecycle phase. It composes
`validate`, the bounded required repository gates, `cleanup`, and `bundle` — it does not duplicate
any validator's logic. In order, it: (1) revalidates every accepted phase's declared artifacts
against the current repository, which proves prior accepted evidence is unchanged and fails outright
(not merely a warning) on any undeclared drift — see `evidence_invariance` in the result, which
distinguishes unchanged phase-local evidence from validly *superseded* shared artifacts rather than
claiming both are simply unchanged; (2) runs the required repository gates — `validate_repository.py`,
`validate_skill_artifacts.py . --json`, `check_worktree_patch.py --json`, and `git diff --check` —
captured in `validation_gates`, any failure aborting the call; (3) runs a cleanup preflight (always
dry-run first), which aborts the call on unmanaged workspace content or any other blocking condition
before any archive exists; (4) generates the deterministic review bundle through `--phase`,
defaulting its output under `../torch-dae-review-bundles/<workflow-id>/<phase>/` (override with
`--review-root`), embedding the required-gate results, lifecycle state, evidence-invariance summary,
and cleanup preflight as archive metadata (`metadata/finalization-validation.json`,
`metadata/lifecycle-state.json`, `metadata/evidence-invariance.json`, `metadata/cleanup-plan.json`)
so the archive alone supports independent review; (5) only after that immutable archive exists,
optionally performs real cleanup execution (`--cleanup-execute`); and (6) writes one
`<workflow-id>-through-<phase>.finalize-result.json` exposing every absolute artifact path —
accepted handoff, canonical phase artifacts, review archive, its `.sha256` sidecar, the bundle result
JSON, the finalize result itself, and the cleanup receipt path/hash when one exists — plus workflow
status, current/requested phase, and cleanup status.

## Card-independent runtime verification

`torch-dae model verify --target <runtime-target.json> --json [--offline]` executes
a schema-2 target in its verified model environment before a model card exists.
See [runtime execution](../runtime-execution.md) for provider and evidence details.
