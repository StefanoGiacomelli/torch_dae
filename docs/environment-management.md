# Environment Management

The environment subsystem materializes model-specific environments from committed card references under
`environments/<card-id>/`. The committed inputs are `environment.json`, `pyproject.toml`, `uv.lock`,
`sources.json`, and `verify_environment.py`.

`card_id` and `environment_id` are related but not interchangeable. The requested card ID must match
the model card and the environment specification `model_card_id`; the model card recommended
environment, environment specification, and source manifest must agree on `environment_id`. Directory
layout remains card-based even when the environment ID differs:
`environments/<card-id>/` and `.torch-dae/environments/<card-id>/<fingerprint>/`.

Cross-document paths are validated exactly. The card must reference
`environments/<card-id>/environment.json` and the same lockfile as the environment specification. The
environment specification must reference `environments/<card-id>/uv.lock`,
`environments/<card-id>/pyproject.toml`, `environments/<card-id>/sources.json`, and
`environments/<card-id>/verify_environment.py`.

Runtime state is ignored and created only under:

```text
.torch-dae/environments/<card-id>/<fingerprint>/
.torch-dae/repositories/
.torch-dae/source-builds/
.torch-dae/reports/
```

Use:

```bash
torch-dae env ensure <card-id>
torch-dae env run <card-id> -- python script.py
torch-dae env info <card-id> --json
torch-dae env remove <card-id>
```

`--offline` reuses valid cached interpreters, environments, source checkouts, wheels, and packages.
It fails clearly on cache misses. `--no-python-downloads` prevents uv-managed Python downloads; offline
mode implies it.

The local `torch-deepaudioembedding` distribution is installed into model environments as a cached
non-editable wheel while retaining the `torch_dae` import package and `torch-dae` console command.
The wheel is built with `uv build --wheel` and the root build backend. The package identity always
includes a content digest over `pyproject.toml`, the configured project README, every regular file
under `src/torch_dae/` including Python modules, package data, and vendored files, plus packaging
configuration files when present. Clean Git states use `git:<HEAD>:content:<digest>`; dirty,
staged, unstaged, untracked, and pre-commit states use `content:<digest>`.

The cached wheel metadata records the package identity, filename, raw SHA-256, distribution
name/version, build command, and `SOURCE_DATE_EPOCH`. Verification reloads `wheel.json` and rejects
missing, malformed, stale, or inconsistent metadata.

Git sources keep a canonical checkout under `.torch-dae/repositories/<source-id>/<revision>/`. Online
mode recovers dirty, wrong-revision, wrong-remote, or incomplete checkouts by replacing them with an
atomic clone through a temporary sibling path. Offline mode reports the invalid cache without
modifying it. Builds export the exact revision into a disposable workspace with `git archive`, build a
wheel there, remove the workspace in all outcomes, and then recheck that the canonical checkout is
clean. Git-source wheel caches include strict metadata for URL, revision, build fingerprint, Python
version, platform, lockfile hash, distribution name/version, filename, and wheel hash.

Environment verification checks the installed local wheel files, wheel `RECORD` coverage, the
`torch-dae` console entry point, sanitized importability, explicit package-source versions, Git
wheel/source state, and vendored files against both repository bytes and local wheel members. Model
environment subprocesses remove `PYTHONPATH` and `PYTHONHOME`.

Command diagnostics for materialization are written under
`.torch-dae/reports/environments/<card-id>/<fingerprint>/` and referenced from
`torch-dae-materialization.json` in execution order. Reports cover the commands actually executed for
Python resolution and inspection, `uv venv`, locked `uv sync`, local wheel build/install, Git clone
and checkout validation, Git archive and wheel build/install, dependency checks, installed
distribution inspection, and the verification script. Each report records sanitized arguments,
working directory, timestamps, duration, return code, stdout, stderr, and status.

Failed materializations are marked `status = failed` with `completed_at` and all available report
references before being moved under `.failed`. Reports redact authorization headers, bearer tokens,
token-like arguments, credential-bearing URLs, and secret environment-variable values. Logs are
sanitized runtime state and must not be committed.

The environment and checkpoint integration tests exercise real local Git clone, exact detached checkout, archive,
wheel build, install, isolated import, offline reuse, dirty-cache online recovery, dirty-cache offline
failure without mutation, wrong-HEAD recovery, wrong-remote recovery, and corrupt source-wheel
metadata repair/failure behavior.

Shell activation is optional; the authoritative execution path is `env run` or the returned
`python_executable` from `EnvironmentManager.ensure()`.

No real audio model is currently integrated.
