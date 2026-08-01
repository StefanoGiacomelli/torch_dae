# Environment Management

The environment subsystem materializes environments directly from the accepted definition under
`environments/<environment-id>/`. Its five authorities are `environment.json`, `pyproject.toml`,
`uv.lock`, `sources.json`, and `verify_environment.py`; no model card or checkpoint is needed.

`environment_id` is the logical infrastructure identity. `card_id` identifies a separate
model/variant/checkpoint publication tuple. A card may recommend an environment, but that reference
does not authorize or define materialization. Legacy card-keyed environment directories remain
readable when their `environment_id` resolves uniquely.

Cross-document paths are validated exactly. The specification must reference all four sibling
artifacts, its project and lock must agree on exact direct dependencies and Python constraints, and
the source manifest must agree on environment identity, package pins, and referenced source hashes.

Runtime state is ignored and created only under:

```text
.torch-dae/environments/<environment-id>/<fingerprint>/
.torch-dae/repositories/
.torch-dae/source-builds/
.torch-dae/reports/environments/<environment-id>/<fingerprint>/
```

Use:

```bash
torch-dae env resolve <environment-id>
torch-dae env materialize <environment-id>
torch-dae env verify <environment-id>

# Backward-compatible card conveniences:
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

The deterministic fingerprint hashes canonical environment identity plus the exact specification,
project, lock, source-manifest, and verification-script evidence; exact CPython and normalized
platform identity; exact locked direct dependencies; referenced source hashes; and the local wheel
identity. It excludes card prose, checkpoints, absolute runtime paths, timestamps, usernames,
temporary paths, and mutable logs.

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

Materialization and verification have different result contracts. `EnvironmentMaterializationResult`
records creation/reuse, dependency and source preparation, and managed identity, but never claims
verification. `EnvironmentVerificationResult` records interpreter/platform, installed direct
dependencies, imports/smoke checks, evidence hashes, and the fingerprint. It establishes only
infrastructure compatibility—not checkpoint loading, forward/output/embedding correctness, or
unobserved device support. A passed result has `verification_status = passed`,
`lifecycle_state = environment_verified`, nonempty import and smoke observations, only passed
observations with unique names, and no failure classification. A successful result proves both a
successful global status and complete successful coverage of the evidence required by its authority
contract. Failed or unsupported required observations, or absent required observation collections,
cannot accompany success. A failed result remains valid diagnostic evidence with
`lifecycle_state = materialized` and a non-null failure classification; it may retain partial,
failed, or absent observations and cannot promote lifecycle.

Environment verification checks the installed local wheel files, wheel `RECORD` coverage, the
`torch-dae` console entry point, sanitized importability, explicit package-source versions, Git
wheel/source state, and vendored files against both repository bytes and local wheel members. Model
environment subprocesses remove `PYTHONPATH` and `PYTHONHOME`.

Command diagnostics for materialization are written under
`.torch-dae/reports/environments/<environment-id>/<fingerprint>/` and referenced from
`torch-dae-materialization.json` in execution order. Reports cover the commands actually executed for
Python resolution and inspection, `uv venv`, locked `uv sync`, local wheel build/install, Git clone
and checkout validation, Git archive and wheel build/install, dependency checks, and installed
distribution inspection. Verification commands are recorded separately below
`verification-commands/`. Each report records sanitized arguments,
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

Managed environment-verification results are ignored execution evidence. Verify orchestration
promotes a normalized copy into its accepted phase-local
`onboarding_reports/<workflow-id>/verify/` tree for a final card to consume. The promoted copy must
retain passed status and the matching fingerprint. Managed logs or an unmanaged result path cannot
substitute for that canonical card evidence. Only checkpoint-specific runtime reports belong in
`verification_reports/`.

PANNs adapters are integrated for three AudioSet tuples. Pretrained checkpoint acquisition,
checkpoint-specific runtime verification, and final model cards remain pending.
