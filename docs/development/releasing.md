# Releasing

This page is the production runbook for a `torch-dae` release. The repository publishes the
`torch-deepaudioembedding` distribution from a GitHub Release through PyPI Trusted Publishing; no
package-index token is stored in the repository.

## Release invariants

Before publishing a release, all of the following must be true:

- the release-preparation pull request has been merged into protected `main`;
- GitHub Actions is green on the resulting `main` merge commit;
- Read the Docs `latest` has built successfully from that commit;
- `pyproject.toml`, `uv.lock`, `CHANGELOG.md`, and `CITATION.cff` agree on the intended release;
- `scripts/check_release_version.py --current` validates the project version;
- Ruff, mypy, the full test suite, coverage gates, schema checks, repository validation, skill
  validation, Sphinx `-W`, package build, and Twine checks pass;
- the Git worktree is clean.

PyPI release files are immutable. Never attempt to republish a version that already exists.

## Published identifiers

- PyPI project name: `torch-deepaudioembedding`
- TestPyPI project name: `torch-deepaudioembedding`
- GitHub repository: `torch_dae`
- The Read the Docs project and documentation slug remain `torch-dae`.

## Production publishing path

The production path is:

```text
release-preparation PR
        ↓
merge commit on main
        ↓
post-merge CI + Read the Docs latest
        ↓
GitHub Release v<project.version>
        ↓
publish.yml validate-and-build
        ↓
manual approval of environment pypi
        ↓
PyPI Trusted Publishing
        ↓
GitHub Release artifacts
        ↓
PyPI / Read the Docs / Zenodo verification
```

`publish.yml` runs only when a GitHub Release is **published**. A normal push to `main` does not
publish a package. The workflow checks out the release tag, requires the exact
`v<project.version>` tag, executes the complete release validation, builds the wheel and source
distribution once, verifies the wheel in a clean environment, publishes the same artifacts to PyPI,
and attaches them to the GitHub Release.

## 1. Prepare and merge the release

On the release branch:

```bash
uv lock --check
uv sync --python 3.11 --all-groups --extra profiling --frozen
uv run python scripts/check_release_version.py --current
uv run ruff format --check
uv run ruff check
uv run mypy src scripts
uv run pytest
uv run python scripts/generate_schemas.py --check
uv run python scripts/generate_profiling_schema.py --check
uv run python scripts/validate_repository.py
uv run python skills/audio-model-onboarding/scripts/validate_skill_artifacts.py . --json
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
uv run python -m build
uv run python -m twine check dist/*
```

Remove local `dist/` after the check; release artifacts are built again by the trusted GitHub
workflow. Do not commit `dist/`.

Open a pull request against `main`, wait for every required status check and the Read the Docs pull
request build, and merge with a merge commit. After the merge, wait for the new `main` CI and Read
the Docs `latest` build to succeed before creating the release.

## 2. Optional TestPyPI rehearsal

`.github/workflows/test-publish.yml` is manual-only. Use it when a package-index rehearsal is useful,
particularly after packaging or metadata changes. TestPyPI has a separate Trusted Publisher and
environment configuration; it is not part of the production trigger.

A TestPyPI run must still validate the current project version, execute the release gates, build the
distributions once, verify the wheel, and publish only to TestPyPI.

## 3. Create and publish the GitHub Release

Create a GitHub Release targeting the verified `main` commit. The tag must be exactly:

```text
v<project.version>
```

For version `0.2.0`, the production tag is therefore `v0.2.0`. Use the corresponding release-notes
document under `docs/releases/` as the GitHub Release body.

It is safe to prepare the release as a draft. **Publishing** the release is the action that triggers
`.github/workflows/publish.yml`.

## 4. Approve the protected PyPI deployment

Production publishing uses the existing GitHub environment `pypi` and the configured PyPI Trusted
Publisher:

- owner: `StefanoGiacomelli`;
- repository: `torch_dae`;
- workflow: `publish.yml`;
- GitHub environment: `pypi`;
- package: `torch-deepaudioembedding`.

The environment requires manual approval. Review the successful `validate-and-build` job and the
reported release version before approving the deployment. No PyPI API token is required.

## 5. Verify the published release

After the workflow completes, verify all of the following:

1. PyPI lists the intended version and both wheel and source distribution.
2. The GitHub Release contains the same wheel and source distribution.
3. A clean environment can install the public wheel and reports the expected version:

   ```bash
   python -m venv /tmp/torch-dae-release-check
   /tmp/torch-dae-release-check/bin/python -m pip install torch-deepaudioembedding==<version>
   /tmp/torch-dae-release-check/bin/torch-dae --version
   ```

4. Read the Docs has built the release tag and the `stable` documentation alias points to the
   intended release.
5. The Zenodo GitHub integration has archived the release.

## 6. Zenodo DOI synchronization

The stable concept DOI is `10.5281/zenodo.21641390`. A version-specific DOI does not exist until
Zenodo archives the published GitHub Release, so it must never be invented during release
preparation.

`CITATION.cff` in the release-preparation commit records the exact software version/date and the
concept DOI. After Zenodo assigns the immutable DOI for the new release, update the default branch
through a small metadata-only pull request by adding the version DOI to `CITATION.cff`. Do not move
or recreate the already published Git tag.

## Repository and service configuration

The current production configuration is intentionally external to the repository:

- `main` is protected by a ruleset requiring pull requests and CI status checks;
- force pushes and branch deletion are blocked for `main`;
- Codecov uploads use OIDC and fail the CI job if upload fails;
- Read the Docs is connected through the GitHub integration and builds `main` / pull requests;
- PyPI publishing uses OIDC Trusted Publishing through the protected `pypi` environment.

Repository workflows must not contain package-index credentials, user secrets, host-specific
paths, or manual publication tokens.
