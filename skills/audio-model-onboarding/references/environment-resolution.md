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

Successful resolution may prepare `environments/<card-id>/environment.json`, `pyproject.toml`,
`uv.lock`, `sources.json`, and `verify_environment.py`.
Official-package resolution requires exact `source_package_name` and `source_package_version`
evidence that matches the selected candidate. Accepted identity provenance is verified upstream
`package_metadata` or locally observed `environments/<card-id>/pyproject.toml`, `uv.lock`, or
`environment.json`; `sources.json`, `verify_environment.py`, arbitrary files, inference, and runtime
observations cannot prove package identity. Remaining
source-strategy decision gates block `environment_resolved`; resolved choices belong in decision
records. Diagnostic references use `.torch-dae`-relative
`reports/environments/<card-id>/<fingerprint>/<report>.json`, while committed verification reports
use `verification_reports/<card-id>/<report>.json`.
