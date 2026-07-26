# torch-dae

`torch-dae` is a PyTorch audio-model onboarding framework. Phase 02 implements the canonical
audio-model onboarding skill MVP on top of the Phase 00 typed contracts and Phase 01 environment and
checkpoint cache subsystems.

No real model is integrated yet. The root environment stays lightweight and model-agnostic;
model-specific dependencies, materialized environments, checkpoints, reports, and profiling outputs
belong under ignored `.torch-dae/` runtime state.

Core invariants:

- one model card describes exactly one model-family, variant, and checkpoint tuple;
- public waveform APIs use `[B,C,T]` plus `sample_rate`, with optional `[B]` valid lengths;
- environment creation is explicit and reproducible from committed specifications;
- each environment spec references one canonical committed `sources.json` manifest;
- checkpoints are cached under `.torch-dae/checkpoints/` and are never committed;
- profiling is reserved for later runtime-verified integrations.

Implemented control-plane commands:

```bash
torch-dae card list
torch-dae card show <card-id>
torch-dae card validate <card-id-or-path>

torch-dae env create <card-id>
torch-dae env ensure <card-id>
torch-dae env verify <card-id>
torch-dae env remove <card-id>
torch-dae env info <card-id>
torch-dae env run <card-id> -- <command>

torch-dae checkpoint ensure <card-id>
torch-dae checkpoint info <card-id>
torch-dae checkpoint remove <card-id>
```

Environment commands recreate a verified recommended environment from committed
`environments/<card-id>/` inputs into `.torch-dae/environments/<card-id>/<fingerprint>/`. The
committed `environment_id` may differ from the card ID, but all committed references must agree and
the filesystem layout remains card-based. Checkpoint commands acquire or reuse assets under
`.torch-dae/checkpoints/<checkpoint-id>/<sha256>/`.

Use `--offline` to require cache reuse and `--no-python-downloads` to prevent uv-managed Python
downloads. Offline mode still allows uncached local acquisitions such as `local_path` checkpoints and
package-bundle checkpoints from an already materialized environment. Use
`torch-dae env run <card-id> -- python script.py` instead of relying on shell activation.

Phase 01 model environments install the actual local `torch-dae` distribution as a non-editable wheel
built through the declared build backend. The package identity includes `pyproject.toml`, the
configured README, Python modules, package data, vendored files, and packaging configuration. The
wheel cache records raw hashes and build metadata, and verification checks installed files, package
metadata, console entry points, vendored file inclusion, source-wheel metadata, and explicit source
package versions with `PYTHONPATH` and `PYTHONHOME` removed from model-environment commands. Runtime
command and checkpoint acquisition diagnostics are written only under ignored `.torch-dae/reports/`.
Successful environment and checkpoint metadata reference their report files. Failed environment
materializations are marked failed with report references before quarantine, and failed checkpoint
acquisitions retain sanitized runtime reports without creating valid cache metadata. Checkpoint
failure reports cover typed transport, stream, local I/O, package-bundle, hash-validation,
offline-cache, cache-finalization, metadata-write, cleanup, and response-close outcomes. The
checkpoint CLI exits with code `3` for not-found/offline-unavailable failures and code `4` for
acquisition or hash failures, without displaying tracebacks for expected operational errors.

Model integration commands are still deferred. Phase 02 defines the evidence-driven onboarding
workflow, static inspection utilities, templates, report contracts, synthetic evaluations, and
validation rules that future model integrations must follow. Synthetic golden reports are evaluated
against production-inspector observations, so fixture evidence such as package metadata, dependency
declarations, model symbols, checkpoint helpers, pinned revisions, source strategy evidence, and
embedding tensor candidates must stay truthful.

The canonical onboarding skill lives at `skills/audio-model-onboarding/`. Codex and Claude project
skill entries are relative symlinks to that single directory. The skill supports these modes:

- `analyze`
- `resolve-environment`
- `integrate`
- `verify`
- `card`
- `profile`

`profile` remains reserved. Phase 02 does not integrate PANNs, BYOL-A, EnCodec, or any other real
audio model; PANNs begins only in Phase 03. Phase 02 tests use synthetic repositories only, and each
fixture is marked as synthetic and scientifically meaningless.

The onboarding evidence policy requires every material claim to be classified as a verified upstream
fact, locally observed behavior, reasoned inference, user-provided decision, unresolved ambiguity, or
unsupported claim. Inferences need a rationale and cannot silently become verified facts. Candidate
generation applies the same semantic compatibility checks to dependency records, source strategies,
environment candidates, and decision gates.

Environment candidate generation emits a strict result envelope with evidence items, normalized
dependency provenance, source-strategy context, decision gates, unresolved constraints, and optional
target platform. Official-package candidates carry exact `source_package_name` and
`source_package_version` evidence, source-strategy gates always block promotion, and Phase 01
diagnostics are referenced as `.torch-dae`-relative
`reports/environments/<card-id>/<fingerprint>/<report>.json` paths. Conda ranges, GitHub Actions CI
matrices, `.github/workflows/ci.yml` evidence paths, checkpoint-helper provenance, shared inspection
budgets, and local real-Git evaluation are covered by Phase 02 tests. Real environment
materialization remains a Phase 01 operation.

Additional documentation:

- `docs/model-onboarding-skill.md`
- `docs/onboarding-evidence-policy.md`
- `docs/onboarding-artifacts.md`
