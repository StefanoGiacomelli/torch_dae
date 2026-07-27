# torch-dae: an AI skill-based framework for Audio Embedding Models

**Status:** pre-release · **Version:** 0.1.0

`torch-dae` is a model-agnostic control plane and AI skill-based workflow for
evidence-grounded onboarding and integration of audio embedding models. It records source
provenance, checkpoint identity, isolated environment resolution, wrapper behavior, embeddings,
and runtime observations as explicit, validated artifacts.

No model-specific integrations are distributed in the current release. Model support is added
through the canonical onboarding workflow and isolated model-specific environments.

## Installation

Install the package-index distribution:

```bash
pip install torch-deepaudioembedding
torch-dae --help
```

Or install the source checkout:

```bash
git clone https://github.com/StefanoGiacomelli/torch_dae.git
cd torch_dae
uv sync --all-groups
uv run torch-dae --help
```

Start with the {doc}`getting-started/quickstart`, then follow the
{doc}`tutorials/audio-model-onboarding` for the skill workflow. Package users can consult the
hand-curated {doc}`api/index`; contributors should begin with
{doc}`development/architecture` and {doc}`development/contributing`.

```{toctree}
:maxdepth: 2
:caption: Getting started

getting-started/installation
getting-started/quickstart
```

```{toctree}
:maxdepth: 2
:caption: Tutorials

tutorials/audio-model-onboarding
tutorials/agent-interaction
```

```{toctree}
:maxdepth: 2
:caption: User guide

user-guide/model-registry
user-guide/environments
user-guide/checkpoints
user-guide/embeddings
user-guide/model-cards
```

```{toctree}
:maxdepth: 2
:caption: Skill reference

skill/overview
```

```{toctree}
:maxdepth: 2
:caption: API reference

api/index
```

```{toctree}
:maxdepth: 2
:caption: Development

development/architecture
development/contributing
development/testing
development/documentation
development/releasing
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference/cli
reference/schemas
reference/lifecycle
```

```{toctree}
:hidden:

checkpoint-management
environment-management
model-onboarding-skill
onboarding-artifacts
onboarding-evidence-policy
```

## Citation

Cite the software entry in the repository's
[CITATION.cff](https://github.com/StefanoGiacomelli/torch_dae/blob/main/CITATION.cff) when citing the
repository or package. Also cite the IEEE ISCC paper (DOI
[`10.1109/ISCC65549.2025.11326439`](https://doi.org/10.1109/ISCC65549.2025.11326439)) when
discussing the framework design, standardization rationale, or deployment methodology.

## Funding

Research project: *Methods of Computational Auditory Scene Analysis and Synthesis supporting
eXtended and Immersive Reality Services*.

Research activities were mainly funded under the Ministerial Decree (DM) 118/2023, Mission 4,
Component 1, Investment 4.1 of the National Recovery and Resilience Plan (PNRR) – “PNRR Research” –
CUP: E11I23000100001.

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

[Email](mailto:stefano.giacomelli@graduate.univaq.it) ·
[GitHub](https://github.com/StefanoGiacomelli) ·
[ORCID](https://orcid.org/0009-0009-0438-1748) ·
[Google Scholar](https://scholar.google.com/citations?user=l-n0hl4AAAAJ&hl=en) ·
[LinkedIn](https://www.linkedin.com/in/stefano-giacomelli-811654135)
