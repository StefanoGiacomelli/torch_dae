# Environment Resolution

Environment resolution is evidence-driven. Candidate versions must come from Python constraints,
declared dependencies, lock files, setup files, CI/Docker evidence, imported APIs, release dates,
official compatibility matrices, issue evidence when inspected, and controlled local trial results.
Conda declarations preserve ranges such as `numpy<1.24`, exact Conda assignments such as
`python=3.10`, and build-suffixed declarations such as `pytorch=1.13.1=<build>` without treating the
build string as the version. GitHub Actions CI matrices may contribute exact dependency records from
scalar, inline-list, or block-list matrix values, with `.github/workflows/ci.yml` preserved as the
evidence path. Only values inside static `strategy` → `matrix` definitions create dependency
records. GitHub Actions `${{ ... }}` references in setup actions, environment variables, or commands
are ignored. Invalid declarations remain available as diagnostics but do not affect version
selection, constraint merging, conflicts, unpinned classification, or principal dependencies.

Use `generate_environment_candidates.py` to produce ordered unverified candidates. Do not run an
arbitrary Cartesian search. Trial only explicitly selected candidates and use the environment APIs
and CLI to materialize or verify isolated model-specific environments.

Successful resolution may prepare `environments/<environment-id>/environment.json`, `pyproject.toml`,
`uv.lock`, `sources.json`, and `verify_environment.py`.

Every package imported directly by the selected minimal runtime source surface is a direct
dependency. Do not rely on transitive installation. When dependencies are exactly pinned, the
verification script checks every declared direct dependency and exact version.

When multiple tuples may share future source, identify the minimum files and symbols and compare
candidate revisions for byte identity, symbol identity, and semantic differences. Select one common
revision only when it preserves every required variant; otherwise retain tuple-specific provenance.
Chronological proximity never establishes checkpoint equivalence.

Tuples share environment evidence only when direct dependencies are equivalent, selected source APIs
are compatible, import/constructor trials pass for every tuple, platform and interpreter match, and
the reused evidence and differences are documented.

A constructor trial establishes dependency resolution, source import, class construction, and
parameter-device placement only. It does not establish checkpoint compatibility, forward
correctness, output correctness, embedding correctness, inference equivalence, or runtime
verification.

Resolve-environment has two successful completion states. **Draft resolution complete** means a
candidate was selected, its isolated import/constructor trial passed, environment drafts and locks
validate, production materialization prerequisites are intentionally absent, and no fingerprint or
lifecycle promotion is claimed; `integrate` may be next. **Environment lifecycle resolved** requires
canonical materialization, successful verification, fingerprint and report reference, and every
strict lifecycle prerequisite. Draft completion is not failure and must not be promoted to
`environment_resolved`.

Optional host diagnostics are portable and non-blocking. A failed optional CPU-brand or
platform-detail command is recorded separately from model-trial success. Sandbox, DNS,
package-index, authentication, and rate-limit failures retain their original logs and classification.
One identical evidence-motivated rerun is permitted when policy allows; preserve both outcomes and
do not hide an external failure by changing generic code.

The repository root `.venv` is expected control-plane runtime state. Create or synchronize it with
`uv sync --all-groups --frozen`, and run analysis utilities with `uv run`; this is not model
environment materialization. `pypdf` belongs to the root only as bounded document-analysis tooling.
Model dependencies remain prohibited from the root and are resolved under `.torch-dae/environments/`
through the model-environment subsystem. Report root environment creation when observed, but do not
classify the `.venv` itself as contamination. Bytecode and tool caches remain ignored and excluded
from audit archives.

Official-package resolution requires exact `source_package_name` and `source_package_version`
evidence that matches the selected candidate. Accepted identity provenance is verified upstream
`package_metadata` or locally observed `environments/<environment-id>/pyproject.toml`, `uv.lock`, or
`environment.json`; `sources.json`, `verify_environment.py`, arbitrary files, inference, and runtime
observations cannot prove package identity. Remaining
source-strategy decision gates block `environment_resolved`; resolved choices belong in decision
records. Diagnostic references use `.torch-dae`-relative
`reports/environments/<environment-id>/<fingerprint>/<report>.json`. Accepted normalized
environment evidence belongs in the verify handoff tree and must retain passed status plus the
matching fingerprint; only checkpoint-specific runtime reports use
`verification_reports/<card-id>/<report>.json`. Failed environment evidence remains diagnostic and
cannot promote lifecycle state.
A successful environment result additionally requires nonempty import and smoke observations, only
passed observations, and unique names across both collections. Missing, failed, or unsupported
required observations cannot accompany successful global status.
