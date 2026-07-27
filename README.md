# torch-dae: an AI skill-based framework for Audio Embedding Models

[![CI](https://github.com/StefanoGiacomelli/torch_dae/actions/workflows/ci.yml/badge.svg)](https://github.com/StefanoGiacomelli/torch_dae/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/gh/StefanoGiacomelli/torch_dae/graph/badge.svg)](https://codecov.io/gh/StefanoGiacomelli/torch_dae)
[![Documentation Status](https://readthedocs.org/projects/torch-dae/badge/?version=latest)](https://torch-dae.readthedocs.io/en/latest/?badge=latest)
[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Status: Pre-release](https://img.shields.io/badge/status-pre--release-orange)](#project-status)

## Project status

torch-dae is pre-release research software. Its typed control plane, isolated environment and
checkpoint subsystems, and evidence-grounded audio-model onboarding skill are implemented and
validated. No model-specific integrations are distributed in the current release. Model support is
added through the canonical onboarding workflow and isolated model-specific environments.

## Overview

`torch-dae` turns an official audio-model implementation and a specific checkpoint into a
reproducible, evidence-backed PyTorch integration. It separates lightweight repository control from
model runtimes so incompatible research dependencies can live in isolated, checkpoint-specific
environments.

The project keeps source provenance, scientific decisions, environment resolution, checkpoint
identity, wrapper behavior, embeddings, and runtime observations explicit and machine-validatable.
One model card always describes exactly one model family, variant, and checkpoint tuple.

## Current capabilities

- Strict typed contracts and generated JSON Schemas for cards, environments, checkpoints,
  embeddings, onboarding reports, and runtime verification.
- Reproducible model-specific environment materialization, verification, reuse, and execution.
- Checkpoint acquisition from HTTPS, GitHub releases, Hugging Face, package resources, and local
  paths, with hashing, cache validation, offline behavior, and sanitized diagnostics.
- Static upstream inspection and evidence-grounded analysis without importing untrusted model code.
- A canonical agent-neutral onboarding skill shared by Codex and Claude.
- Synthetic grounded evaluations, repository safety validation, and strict quality gates.

## Not available yet

- No real audio model or checkpoint is included.
- The `model inspect` and `model verify` CLI placeholders do not execute model workflows yet.
- Profiling is reserved until a model is runtime-verified and a profiling workflow is explicitly
  implemented and invoked.
- The first package-index release is not published yet; source installation remains available.

## Architecture

The root package is a model-agnostic control plane:

- [Typed contracts](src/torch_dae/contracts.py) and [model-card models](src/torch_dae/cards/models.py)
  enforce public identity, lifecycle, waveform, output, and evidence rules.
- [Environment management](docs/environment-management.md) recreates isolated runtimes from
  committed specifications under `environments/<card-id>/`.
- [Checkpoint management](docs/checkpoint-management.md) validates and caches assets only under
  ignored runtime state.
- [Static onboarding inspectors](src/torch_dae/onboarding/inspection.py) collect bounded evidence
  without executing upstream repositories.
- [Evidence and decision gates](docs/onboarding-evidence-policy.md) prevent unsupported facts or
  ambiguous scientific choices from being silently promoted.
- The [canonical onboarding skill](skills/audio-model-onboarding/SKILL.md) drives agent workflows for
  both Codex and Claude.
- [Synthetic grounded evaluation](skills/audio-model-onboarding/references/synthetic-evaluation.md)
  checks reports against concrete fixture observations.

Model cards belong under `model_cards/`, verification reports under `verification_reports/`, and
committed environment inputs under `environments/`. Materialized environments, repositories,
checkpoints, diagnostics, and coverage data remain under ignored `.torch-dae/`.

## Installation

The complete user and API documentation is available on
[Read the Docs](https://torch-dae.readthedocs.io/en/latest/).

Install a published package-index release:

```bash
pip install torch-deepaudioembedding
torch-dae --help
```

Install from source with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/StefanoGiacomelli/torch_dae.git
cd torch_dae
uv sync --all-groups
uv run torch-dae --help
```

## Quick start

Inspect the empty public registry and validate a synthetic card fixture:

```bash
uv run torch-dae card list
uv run torch-dae card validate tests/fixtures/valid/model-card.analyzed.json
uv run torch-dae env --help
uv run torch-dae checkpoint --help
```

Model-specific environment and checkpoint commands require a committed model card and its
environment specification. No such production artifact is included yet.

## Available CLI commands

The implemented control-plane CLI currently exposes:

```text
uv run torch-dae card list
uv run torch-dae card show <card-id>
uv run torch-dae card validate <card-id-or-path>

uv run torch-dae env create <card-id>
uv run torch-dae env ensure <card-id>
uv run torch-dae env verify <card-id>
uv run torch-dae env remove <card-id>
uv run torch-dae env info <card-id>
uv run torch-dae env run <card-id> -- <command>

uv run torch-dae checkpoint ensure <card-id>
uv run torch-dae checkpoint info <card-id>
uv run torch-dae checkpoint remove <card-id>
```

The CLI also exposes `uv run torch-dae model inspect <card-id>` and
`uv run torch-dae model verify <card-id>` as explicit unavailable-feature placeholders. They do not
perform onboarding or runtime verification.

## Audio-model-onboarding skill

The [canonical skill](skills/audio-model-onboarding/SKILL.md) supports these implemented agent
workflows:

- `analyze`: static, evidence-grounded upstream analysis.
- `resolve-environment`: compatibility resolution and controlled isolated trials.
- `integrate`: explicitly authorized wrapper and model-package integration after all prerequisites.
- `verify`: controlled acquisition and runtime verification for one selected model/checkpoint.
- `card`: checkpoint-specific model-card generation from validated evidence and artifacts.

`profile` is documented separately as reserved functionality. Agent workflows are not one-shot CLI
commands and must not be confused with the control-plane commands above. See the
[skill guide](docs/model-onboarding-skill.md), [artifact guide](docs/onboarding-artifacts.md), and
[canonical templates](skills/audio-model-onboarding/templates/README.md).

## Copy-paste agent request

Copy [the canonical request template](skills/audio-model-onboarding/templates/agent-request.md) and
fill in its placeholders:

```text
Use the canonical `audio-model-onboarding` skill available in this repository.

MODE: <analyze | resolve-environment | integrate | verify | card>

MODEL_NAME: <MODEL_NAME>
UPSTREAM_REPOSITORY: <GITHUB_REPOSITORY_URL>
PAPER_OR_TECHNICAL_REFERENCE: <PAPER_URL_OR_NONE>

TARGET_VARIANT: <VARIANT_NAME_OR_AUTO_DISCOVER>
TARGET_CHECKPOINT: <CHECKPOINT_NAME_OR_AUTO_DISCOVER>
PREFERRED_EMBEDDING: <EMBEDDING_NAME_OR_UNRESOLVED>

ADDITIONAL_CONSTRAINTS:
<OPTIONAL_PROJECT_SPECIFIC_CONDITIONING_OR_NONE>
```

The template also carries the required safety, evidence, isolation, validation, mode-scope, and
no-commit instructions.

## Expected agent response

The [canonical response template](skills/audio-model-onboarding/templates/agent-response.md) requires
`Summary`, `Work completed`, `Problems and resolutions`, `Open questions`, `Files`, and `Validation`
sections. It lists only files actually changed for the requested model and uses `None.` when no open
question remains.

## Repository layout

```text
src/torch_dae/                         Root control-plane package
schemas/                               Generated strict JSON Schemas
scripts/                               Schema, coverage, and repository validation
skills/audio-model-onboarding/         Canonical agent workflow
docs/                                  Public subsystem and workflow guides
tests/                                 Contract, subsystem, safety, and synthetic evaluation tests
environments/                          Committed per-card environment inputs
model_cards/                           Checkpoint-specific production cards
verification_reports/                  Committed runtime observations
.torch-dae/                            Ignored runtime state
```

The `.agents/` and `.claude/` skill entries are public relative symlinks to the canonical skill.

## Development and validation

```bash
uv sync --all-groups
uv run ruff format --check
uv run ruff check
uv run mypy src scripts
uv run pytest
uv run python scripts/generate_schemas.py --check
uv run python scripts/validate_repository.py
uv run python skills/audio-model-onboarding/scripts/validate_skill_artifacts.py . --json
```

Run coverage with the same local thresholds as CI:

```bash
mkdir -p .torch-dae
uv run pytest -q \
  --cov=torch_dae \
  --cov-branch \
  --cov-report=term-missing \
  --cov-report=json:.torch-dae/coverage.json
uv run python scripts/check_coverage.py \
  .torch-dae/coverage.json \
  --min-line 85 \
  --min-branch 70
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution and pull-request expectations.

## Roadmap

Model support is added through the canonical onboarding workflow after its evidence, isolation, and
runtime prerequisites are satisfied. Profiling remains unavailable until a model integration is
runtime-verified and a dedicated profiling workflow is implemented.

## Funding

The research project title is:

*Methods of Computational Auditory Scene Analysis and Synthesis supporting eXtended and Immersive
Reality Services*

Funding identifiers: DM 118/2023; Mission 4; Component 1; Investment 4.1; PNRR Research; CUP
E11I23000100001.

Research activities were mainly funded under the Ministerial Decree (DM) 118/2023, Mission 4,
Component 1, Investment 4.1 of the National Recovery and Resilience Plan (PNRR) – “PNRR Research” –
CUP: E11I23000100001.

## Citations

Cite the software entry when citing this repository or package. Also cite the IEEE ISCC paper when
discussing the framework design, standardization rationale, or deployment methodology. The
repository's canonical software metadata is in [CITATION.cff](CITATION.cff).

```bibtex
@software{giacomelli2026torch_dae,
  author  = {Giacomelli, Stefano},
  title   = {{torch-dae}: an AI skill-based framework for Audio Embedding Models},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/StefanoGiacomelli/torch_dae},
  license = {Apache-2.0}
}
```

```bibtex
@inproceedings{giacomelli2025torch_dae,
  author    = {Giacomelli, Stefano and Centofanti, Carlo and
               Graziosi, Fabio and Rinaldi, Claudia},
  title     = {{Design and Deployment of a Standard Framework for
                Audio Neural Networks Embedding Models}},
  booktitle = {2025 IEEE Symposium on Computers and Communications (ISCC)},
  year      = {2025},
  pages     = {1--6},
  doi       = {10.1109/ISCC65549.2025.11326439},
  publisher = {IEEE}
}
```

## License

Licensed under the [Apache License 2.0](LICENSE). Attribution information is provided in
[NOTICE](NOTICE).

## Contact

**Stefano Giacomelli**<br>
ICT - Ph.D. Candidate<br>
Department of Information Engineering, Computer Science and Mathematics (DISIM)<br>
University of L'Aquila, Italy

<img
  src="https://phdict.disim.univaq.it/wp-content/uploads/2024/06/logo-univaq-disim-2-2-768x283.png"
  alt="University of L'Aquila — DISIM"
  width="420"
/>

- **Email:** [stefano.giacomelli@graduate.univaq.it](mailto:stefano.giacomelli@graduate.univaq.it)
- **GitHub:** [StefanoGiacomelli](https://github.com/StefanoGiacomelli)
- **ORCID:** [0009-0009-0438-1748](https://orcid.org/0009-0009-0438-1748)
- **Google Scholar:** [Profile](https://scholar.google.com/citations?user=l-n0hl4AAAAJ&hl=en)
- **LinkedIn:** [Profile](https://www.linkedin.com/in/stefano-giacomelli-811654135)
