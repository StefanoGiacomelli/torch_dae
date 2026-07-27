# Releasing

## Release sequence

1. Complete a human audit of the staged repository and merge through a protected pull request.
2. Confirm CI quality, Python 3.11/3.12 tests, coverage, documentation, and distribution jobs pass.
3. Confirm the `Unreleased` changelog is accurate and `project.version` is the intended version.
4. Create a tag exactly equal to `v<project-version>` and create a GitHub Release from that commit.
5. Publish the GitHub Release. This alone triggers `publish.yml`.
6. Approve the protected `pypi` deployment after reviewing its validation and build artifact.
7. Verify the PyPI project, attached GitHub Release wheel/source distribution, and archived services.

The production workflow validates the release tag, Ruff, mypy, the complete test and coverage gates,
schemas, repository structure, the onboarding skill, and warning-clean documentation. It builds the
wheel and source distribution once, runs Twine, installs the wheel in a clean environment, and
reuses the exact artifact for PyPI and GitHub Release assets.

Do not commit `dist/` or use a manual package-index token.

The package-index distribution is `torch-deepaudioembedding`, which normalizes to
`torch_deepaudioembedding` in wheel and source-distribution filenames. The software title and
documentation slug remain `torch-dae`, the import package remains `torch_dae`, and the console
command remains `torch-dae`. Release automation uses distribution globs rather than a hard-coded
artifact filename.

## GitHub environment: pypi

After the first push, the repository owner must create:

- Environment name: `pypi`
- Required reviewer: Stefano Giacomelli
- Deployment protection: release workflow only

Configure the pending PyPI Trusted Publisher with these exact values:

- PyPI project name: `torch-deepaudioembedding`
- GitHub owner: `StefanoGiacomelli`
- GitHub repository: `torch_dae`
- Workflow filename: `publish.yml`
- GitHub environment: `pypi`

No PyPI API token is required. Two-factor authentication should remain enabled, and recovery codes
must be stored safely. When a pending publisher is used, the project name is not reserved until the
first successful publication.

## GitHub environment: testpypi

Create:

- Environment name: `testpypi`
- Execution: manual only through `workflow_dispatch`
- Manual approval: optional

Configure the TestPyPI Trusted Publisher with:

- TestPyPI project name: `torch-deepaudioembedding`
- GitHub owner: `StefanoGiacomelli`
- GitHub repository: `torch_dae`
- Workflow filename: `test-publish.yml`
- GitHub environment: `testpypi`

TestPyPI uses a separate account and publisher configuration. The manual workflow validates the
current project version and critical gates, builds once, verifies the wheel, uploads a short-lived
artifact, and publishes only to `https://test.pypi.org/legacy/`.

## Read the Docs

After pushing:

1. Sign in to Read the Docs and connect the GitHub App.
2. Import `StefanoGiacomelli/torch_dae`.
3. Request or select the project slug `torch-dae`.
4. Set `main` as the default branch.
5. Enable pull-request builds.
6. Verify the `latest` documentation.
7. Enable tagged versions when releases exist.

The repository's `.readthedocs.yaml` installs the `docs` extra on Python 3.12 and fails on warnings.
Do not commit a Read the Docs token or repository secret.

## Codecov

Connect the public repository to Codecov, confirm OIDC upload support, verify the first coverage
upload, and confirm the README badge resolves. A `CODECOV_TOKEN` is not required unless OIDC fails
and the repository owner explicitly chooses that fallback later.

## Zenodo

Sign in to Zenodo, connect the GitHub account, enable the `torch_dae` repository, publish a GitHub
Release, and verify that the release is archived. Copy the resulting DOI into future software
metadata only after it exists. `CITATION.cff` is the canonical repository citation metadata; do not
invent a DOI.

## Branch protection

Protect `main` after the first push:

- require pull requests before merging;
- require successful CI status checks;
- require the documentation build;
- require the Python test matrix;
- require quality and coverage jobs;
- prevent force pushes;
- prevent branch deletion.

Repository settings are configured by the owner, not by a repository workflow.
