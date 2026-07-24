# torch-dae

`torch-dae` is a PyTorch audio-model onboarding framework. Phase 01 implements the control-plane
environment and checkpoint cache subsystems on top of the Phase 00 typed contracts and schemas.

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

Model integration commands are still deferred because Phase 02+ onboarding and pilot wrappers have
not started.

The canonical onboarding skill lives at `skills/audio-model-onboarding/`. Codex and Claude project skill entries are relative symlinks to that single directory.
