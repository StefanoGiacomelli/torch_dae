---
name: audio-model-profiling
description: Plan, execute, validate, and finalize Profiling v1 Technical Card evidence for an accepted, runtime_verified torch-dae Model Card.
---

# audio-model-profiling

This is the canonical, agent-neutral profiling skill for `torch-dae`. Treat `project_spec.md`
Section 27 ("Technical Cards") and `docs/profiling/*` as normative for this skill. Preserve them
unless the user explicitly asks to edit the specification.

This skill is **not** the onboarding skill. It never creates, resolves, integrates, or
runtime-verifies a Model Card, and it never mutates an accepted Model Card. It consumes an
already-`runtime_verified` Model Card and produces independent, immutable Technical Card
evidence. `skills/audio-model-onboarding/SKILL.md`'s reserved `profile` mode intentionally does
not execute profiling; this skill is the actual implementation.

Codex and Claude entry points resolve to this same directory (`.agents/skills/audio-model-profiling`
and `.claude/skills/audio-model-profiling` are symlinks to it). Never create a Git commit from
this skill.

## Modes

### `resolve`

Resolve the target Model Card and confirm profiling eligibility.

1. Locate `model_cards/<family>/<card-id>.json` and load it.
2. Require `card_status == "runtime_verified"` (or a later terminal state). A card in an earlier
   lifecycle state, or the legacy `profiled` compatibility value, is not eligible; report this and
   stop rather than proceeding.
3. Confirm `usage.recommended_environment.verified` is true and its referenced environment ID
   exists under `environments/<environment-id>/`.
4. Report the resolved `wrapper_entry_point`, `checkpoint.observed_sha256`,
   `input.sample_rate_hz`, and any documented duration constraints from `limitations`.

### `plan`

Given a resolved Model Card, describe the campaign that `torch-dae model profile` will execute
without running it:

- requested devices (default `auto`, expanded via device discovery — see
  `src/torch_dae/profiling/devices.py`);
- the profiling protocol (`audio-inference-v1`, `docs/profiling/protocol.md`);
- canonical duration: an explicit fixed/canonical duration declared by the model/runtime contract
  if one exists, else the protocol default of `10.0s` (`project_spec.md` Section 10) — for a
  variable-length wrapper like the current PANNs integrations, this is always the `10.0s` default;
- the benchmark matrix: canonical-duration batches `1, 2, 4, 8`, plus the empirically discovered
  minimum-input duration at batch 1 when it differs from the canonical duration;
- CPU thread regimes `single_thread` and `native_default`, each in its own isolated subprocess;
- `--energy auto|off`.

Report this plan in chat before executing anything, so the user can redirect scope (e.g. a single
device, `--energy off`) before a real run starts.

### `profile`

Execute the accepted plan:

```bash
torch-dae model profile --model <card-id> --device auto --protocol audio-inference-v1 \
  --energy auto --output-dir <candidate-workspace> --json
```

`--output-dir` **must** resolve to a candidate/workspace path inside the repository and must never
be the repository root or its official `technical_cards/` tree (including descendants). Use a
dedicated candidate workspace (for example
`profiling_candidates/<campaign-id>/` or a clearly-named `candidate_technical_cards/` directory)
so generated evidence is reviewable before any later, separate promotion decision. Promotion into
`technical_cards/<model-id>/` is explicitly out of scope for this skill.

If CodeCarbon can only use hardware-backed measurement through a privileged path (for example
Apple `powermetrics` under `sudo`), do not invoke it silently. Ask the user for explicit consent
first, describing exactly what will run. When consent is granted, add
`--allow-privileged-energy`; otherwise omit that flag and record the resulting non-privileged
`software_estimated` or `unavailable` classification honestly rather than blocking the campaign.

Report, per attempted device: whether it passed its smoke test, which conditions succeeded,
which were explicitly `unsupported` (with reason), and the empirical minimum-input search
outcome, including a `non_monotonic` result if one occurs — never fabricate a minimum.

### `validate`

```bash
torch-dae technical-card validate <path/to/tc-....json> --json
```

Validate every candidate Technical Card produced by `profile` before reporting completion.
Optional/unavailable evidence (energy, FLOPs/MACs, accelerator memory on CPU-only runs) must
never be treated as a validation failure; only structural, identity, hash, and cross-reference
defects are.

### `finalize`

There is no separate profiling-specific finalize command in this first implementation; the
campaign result JSON written under the run's workspace (`campaign-result.json`) together with each
validated candidate Technical Card and its `.npz` raw asset **is** the reviewable evidence set for
this mode. Report their absolute paths, and remind the user that promotion into `technical_cards/`
is a distinct, later, human-reviewed step this skill does not perform.

## Non-negotiable boundaries

- Never modify accepted Model Card bytes, and never write the legacy `profiled` status.
- Never write candidate evidence into `technical_cards/<model-id>/`.
- Never introduce real-audio profiling assets; Profiling v1 uses only deterministic seeded white
  noise (`src/torch_dae/profiling/synthetic_input.py`).
- Never use adaptive repetition counts; canonical v1 is exactly 10 warmups / 50 measured samples
  (`src/torch_dae/profiling/timing.py`).
- Never silently escalate privileges or geolocate the contributor. CodeCarbon energy measurement
  in this implementation always uses an offline, non-geolocating tracker
  (`src/torch_dae/profiling/energy.py`).
- Never install CodeCarbon/`psutil` ad hoc; they are the root project's `profiling` optional
  dependency group (`uv sync --extra profiling`), kept separate from any accepted model-runtime
  environment.

## Reference implementation

- `src/torch_dae/profiling/` — typed contracts, identity, synthetic input, timing, minimum-input
  search, device parsing, privacy validation, energy backend, host/accelerator memory, raw-asset
  handling, validation, storage.
- `src/torch_dae/profiling_worker.py` — runs inside the model's own materialized environment.
- `src/torch_dae/profiling_executor.py` — root-side orchestration and Technical Card assembly.
- `src/torch_dae/cli/models.py` (`model profile`) and `src/torch_dae/cli/technical_cards.py`
  (`technical-card validate|inspect|list`).
