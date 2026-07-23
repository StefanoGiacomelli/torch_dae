# `torch-dae` Project Specification

**Document status:** Proposed normative baseline
**Specification version:** `0.1.0`
**Project type:** PyTorch audio-model onboarding, reproducibility, integration, and profiling framework
**Repository root:** `torch-dae/`
**Python package:** `torch_dae`

## 1. Normative terminology

The terms **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** define mandatory, recommended, discouraged, and optional requirements.

A requirement marked **MUST** is part of the acceptance criteria for the corresponding implementation phase.

---

# 2. Project objective

`torch-dae` provides a reproducible procedure for transforming an official audio-model repository and a specific pretrained checkpoint into:

1. a technically and scientifically grounded repository-analysis report;
2. a checkpoint-specific JSON model card;
3. a reproducible model-specific Python environment;
4. a unified PyTorch wrapper;
5. verified checkpoint loading and inference;
6. explicit access to all meaningful embedding representations;
7. later architectural, runtime, memory, and energy profiling.

The framework targets audio-related models including, but not limited to:

* general-purpose audio tagging;
* audio representation learning;
* acoustic scene classification;
* sound event detection;
* sound event localization and detection;
* music information retrieval;
* speech representation models;
* neural audio codecs;
* multimodal audio encoders;
* bioacoustic and environmental-audio models.

The first release targets models with either:

1. an official PyTorch implementation; or
2. a technically reliable PyTorch wrapper for the official model.

Automatic ports from TensorFlow, JAX, or other frameworks are outside the initial scope.

---

# 3. Non-goals

The first project version MUST NOT attempt to provide:

* one universal environment containing all models;
* automatic legal eligibility decisions;
* automatic blocking based on source or checkpoint licenses;
* automatic ports from non-PyTorch frameworks;
* automatic redistribution of pretrained weights;
* architecture or inference profiling before runtime verification;
* compatibility with the legacy backbone JSON format;
* execution based on arbitrary Python strings or `exec()`;
* hidden environment creation during model construction;
* a generic certification or deployment lifecycle unrelated to model onboarding;
* production-grade remote model serving.

License information MUST be recorded but MUST NOT automatically prevent local analysis, integration, checkpoint loading, or execution.

---

# 4. Fundamental domain entities

The framework distinguishes the following entities.

## 4.1 Model family

A scientific or technical architecture family, for example:

* PANNs;
* BYOL-A;
* EnCodec;
* AudioCLIP;
* HuBERT.

## 4.2 Model variant

A concrete architecture or configuration within a family, for example:

* `Cnn14`;
* `encodec_48khz`;
* `hubert_base`;
* `passt_s`.

## 4.3 Checkpoint

A concrete set of pretrained weights associated with:

* one architecture variant;
* one training configuration;
* one set of tasks;
* one or more training datasets;
* one specific output interpretation.

## 4.4 Model-card identity

Every model card MUST represent exactly:

```text
model family + architecture variant + checkpoint
```

A single model card MUST NOT represent an entire family containing multiple incompatible checkpoints.

The canonical card identifier SHOULD use:

```text
<family>-<variant>-<checkpoint>
```

For example:

```text
panns-cnn14-audioset
encodec-48khz-default
byol-a-audiontt2020-default
```

## 4.5 Model integration

A model integration consists of:

* one checkpoint-specific model card;
* one reproducible environment specification;
* one wrapper entry point;
* one checkpoint specification;
* one runtime-verification report;
* zero or more embedding specifications.

---

# 5. Model-card lifecycle

Every model card MUST have exactly one lifecycle status:

```text
draft
analyzed
environment_resolved
checkpoint_verified
runtime_verified
profiled
```

## 5.1 `draft`

The card exists, but its repository and scientific metadata have not been fully analyzed.

## 5.2 `analyzed`

The following have been inspected:

* official repository;
* architecture implementation;
* scientific publication;
* checkpoint source;
* preprocessing;
* forward path;
* candidate embeddings;
* package and environment evidence.

Unresolved information MAY remain explicit.

## 5.3 `environment_resolved`

A model-specific environment has been successfully constructed and frozen.

The committed environment specification MUST include:

* resolved Python version;
* resolved direct dependencies;
* lock file;
* source-installation strategy;
* supported platform evidence;
* environment-verification command.

## 5.4 `checkpoint_verified`

The specified checkpoint has been:

* acquired;
* hashed;
* loaded into the intended architecture;
* checked for state-dictionary or serialization compatibility.

## 5.5 `runtime_verified`

The wrapper has passed runtime verification for:

* canonical waveform input;
* model construction;
* checkpoint loading;
* forward inference;
* probability output where supported;
* all declared embeddings;
* declared device behavior;
* gradient behavior where applicable.

## 5.6 `profiled`

The runtime-verified integration has completed the defined profiling protocol.

Profiling details remain outside the initial MVP specification.

## 5.7 Issues

Lifecycle status MUST NOT encode every unresolved problem.

Each card MUST instead support an `issues` collection with records such as:

```json
{
  "issue_id": "checkpoint-checksum-missing",
  "kind": "checkpoint_metadata",
  "status": "open",
  "description": "The upstream project does not publish a checkpoint checksum.",
  "impact": "The locally observed checksum is used."
}
```

Allowed issue states SHOULD include:

```text
open
resolved
accepted
not_applicable
```

---

# 6. Evidence semantics

Information stored in model cards MUST distinguish its provenance.

Every material claim SHOULD be classified as one of:

```text
officially_reported
observed
inferred
unresolved
not_reported
not_applicable
```

## 6.1 `officially_reported`

Explicitly stated by an authoritative source such as:

* official repository;
* official model card;
* package metadata;
* scientific paper;
* official documentation.

## 6.2 `observed`

Directly established through repository inspection or successful execution.

## 6.3 `inferred`

Derived from available evidence but not explicitly stated or directly executed.

Every inference MUST include a concise rationale.

## 6.4 Evidence records

The model-card schema MUST support evidence records containing:

```json
{
  "evidence_id": "ev-panns-forward-001",
  "kind": "repository_source",
  "status": "officially_reported",
  "url": "https://github.com/...",
  "revision": "full-git-commit",
  "path": "pytorch/models.py",
  "symbol": "Cnn14.forward",
  "description": "Defines clipwise logits and embedding output."
}
```

Evidence SHOULD remain concise and targeted. The project does not require a legal-audit or certification-style evidence graph.

---

# 7. Canonical repository structure

```text
torch-dae/
├── .git/
├── .gitignore
├── AGENTS.md
├── CLAUDE.md
├── README.md
├── CHANGELOG.md
├── pyproject.toml
├── uv.lock
│
├── skills/
│   └── audio-model-onboarding/
│       ├── SKILL.md
│       ├── references/
│       │   ├── repository-analysis.md
│       │   ├── scientific-metadata.md
│       │   ├── environment-resolution.md
│       │   ├── checkpoint-resolution.md
│       │   ├── wrapper-implementation.md
│       │   ├── embedding-analysis.md
│       │   ├── model-card-authoring.md
│       │   └── runtime-verification.md
│       ├── scripts/
│       └── templates/
│           ├── analysis-report.md
│           ├── model-card.json
│           └── environment.json
│
├── .agents/
│   └── skills/
│       └── audio-model-onboarding -> ../../skills/audio-model-onboarding
│
├── .claude/
│   └── skills/
│       └── audio-model-onboarding -> ../../skills/audio-model-onboarding
│
├── schemas/
│   ├── model-card.schema.json
│   ├── checkpoint.schema.json
│   ├── environment.schema.json
│   ├── embedding.schema.json
│   └── verification-report.schema.json
│
├── src/
│   └── torch_dae/
│       ├── __init__.py
│       ├── core/
│       │   ├── model.py
│       │   ├── outputs.py
│       │   ├── capabilities.py
│       │   ├── checkpoint.py
│       │   ├── embeddings.py
│       │   ├── preprocessing.py
│       │   ├── errors.py
│       │   └── registry.py
│       │
│       ├── environment/
│       │   ├── manager.py
│       │   ├── specification.py
│       │   ├── fingerprint.py
│       │   ├── materialization.py
│       │   ├── verification.py
│       │   └── subprocess.py
│       │
│       ├── cli/
│       │   ├── main.py
│       │   ├── cards.py
│       │   ├── environment.py
│       │   ├── checkpoints.py
│       │   └── models.py
│       │
│       └── models/
│           └── <model-family>/
│               ├── __init__.py
│               ├── model.py
│               ├── preprocessing.py
│               └── vendor/
│
├── model_cards/
│   └── <family>/
│       └── <card-id>.json
│
├── environments/
│   └── <card-id>/
│       ├── environment.json
│       ├── pyproject.toml
│       ├── uv.lock
│       ├── sources.json
│       └── verify_environment.py
│
├── verification_reports/
│   └── <family>/
│       └── <card-id>.json
│
├── tests/
│   ├── core/
│   ├── environment/
│   ├── skills/
│   ├── schemas/
│   └── models/
│
└── .torch-dae/
    ├── repositories/
    ├── source-builds/
    ├── environments/
    ├── checkpoints/
    ├── reports/
    └── profiling/
```

The complete `.torch-dae/` directory MUST be ignored by Git.

No legacy backbone JSON files or parsed legacy hints MUST be included in the new repository.

---

# 8. Root environment and model environments

## 8.1 Root control-plane environment

The repository root environment exists only to operate:

* the CLI;
* model-card validation;
* environment creation;
* checkpoint resolution;
* registry operations;
* skill scripts;
* tests for the control plane.

The root environment SHOULD NOT install:

* PyTorch;
* TorchAudio;
* Transformers;
* model-specific packages;
* model checkpoints.

This minimizes dependency conflicts and allows the environment manager to operate independently.

## 8.2 Model-specific environments

Every checkpoint-specific integration MUST have its own reproducible environment.

The environment specification is committed under:

```text
environments/<card-id>/
```

The materialized virtual environment is stored under:

```text
.torch-dae/environments/<card-id>/<fingerprint>/
```

Model-specific environments MUST install:

* the correct Python interpreter;
* the local `torch-dae` package;
* model runtime dependencies;
* the selected upstream source or package;
* only dependencies required by that integration.

---

# 9. Environment specification

An environment specification MUST include at least:

```json
{
  "schema_version": "1.0.0",
  "environment_id": "panns-cnn14-audioset",
  "model_card_id": "panns-cnn14-audioset",
  "python": {
    "constraint": "==3.10.16",
    "resolved_version": "3.10.16"
  },
  "platforms": {
    "resolved_on": ["macos-arm64"],
    "expected_compatible": ["linux-x86_64"],
    "verified": ["macos-arm64"]
  },
  "dependency_manager": "uv",
  "lockfile": "environments/panns-cnn14-audioset/uv.lock",
  "project_file": "environments/panns-cnn14-audioset/pyproject.toml",
  "sources_file": "environments/panns-cnn14-audioset/sources.json",
  "verification": {
    "script": "environments/panns-cnn14-audioset/verify_environment.py"
  }
}
```

## 9.1 Environment fingerprint

The environment fingerprint MUST depend on:

* canonical environment specification;
* exact lock-file contents;
* exact Python version;
* source URLs and revisions;
* source-installation strategy;
* target platform;
* relevant local `torch-dae` package version or commit.

Changing any of these inputs MUST produce a different fingerprint.

## 9.2 Environment resolution and recreation

The system MUST distinguish two operations.

### Resolution

Performed during onboarding by Codex or Claude.

Resolution discovers a functioning combination of:

* Python;
* PyTorch;
* TorchAudio;
* NumPy;
* model dependencies;
* source revision;
* package or wheel installation strategy.

### Recreation

Performed after resolution using:

```bash
torch-dae env ensure <card-id>
```

Recreation MUST use the committed environment specification and lock file. It MUST NOT repeat compatibility research.

---

# 10. Environment-management API

Environment creation MUST be a dedicated subsystem and MUST NOT occur implicitly inside model initialization.

## 10.1 Python API

```python
from torch_dae.environment import EnvironmentManager

manager = EnvironmentManager.from_repository_root()

environment = manager.ensure(
    model_card_id="panns-cnn14-audioset",
)
```

The manager MUST expose:

```python
class EnvironmentManager:
    def create(self, model_card_id: str) -> ResolvedEnvironment:
        ...

    def ensure(self, model_card_id: str) -> ResolvedEnvironment:
        ...

    def verify(self, model_card_id: str) -> EnvironmentVerification:
        ...

    def remove(self, model_card_id: str) -> None:
        ...

    def info(self, model_card_id: str) -> EnvironmentInfo:
        ...

    def run(
        self,
        model_card_id: str,
        command: list[str],
    ) -> ManagedProcessResult:
        ...
```

## 10.2 `ResolvedEnvironment`

```python
@dataclass(frozen=True)
class ResolvedEnvironment:
    environment_id: str
    model_card_id: str
    root: Path
    python_executable: Path
    fingerprint: str
    python_version: str
    platform: str
    installed_packages: Mapping[str, str]
    installed_sources: tuple[InstalledSource, ...]
    valid: bool
```

## 10.3 CLI

The root CLI MUST expose:

```bash
torch-dae env create <card-id>
torch-dae env ensure <card-id>
torch-dae env verify <card-id>
torch-dae env remove <card-id>
torch-dae env info <card-id>
torch-dae env run <card-id> -- <command>
```

Required semantics:

| Command  | Behavior                                                                                 |
| -------- | ---------------------------------------------------------------------------------------- |
| `create` | Creates a new environment and fails if a valid or invalid materialization already exists |
| `ensure` | Reuses a valid environment or creates/rebuilds it                                        |
| `verify` | Validates an existing environment without changing it                                    |
| `remove` | Deletes only local cached environment state                                              |
| `info`   | Displays committed specification and local materialization status                        |
| `run`    | Executes a child process inside the ensured model environment                            |

A running Python process MUST NOT attempt to mutate itself into another virtual environment.

---

# 11. Upstream-source installation policy

The skill and environment manager MUST apply the following priority order.

## 11.1 Priority 1: official package

Use a published official package with an exact resolved version.

Example:

```json
{
  "source_id": "encodec-package",
  "role": "model_implementation",
  "installation": "package",
  "package": "encodec",
  "version": "0.1.1"
}
```

## 11.2 Priority 2: pinned official repository

When no appropriate package exists, use an immutable repository revision.

The environment manager SHOULD:

1. clone the repository into ignored cache;
2. checkout the exact revision;
3. build a wheel;
4. install the wheel into the model environment.

Example:

```json
{
  "source_id": "official-repository",
  "role": "model_implementation",
  "installation": "git",
  "url": "https://github.com/...",
  "revision": "full-commit-sha",
  "build": "wheel"
}
```

## 11.3 Priority 3: minimal vendored adaptation

Vendoring MAY be used only when the upstream source cannot be installed reproducibly.

The vendored code MUST include:

* upstream URL;
* upstream revision;
* copied file list;
* adaptation description;
* justification;
* tests demonstrating intended equivalence.

The framework MUST avoid unnecessary architecture reimplementation.

---

# 12. Checkpoint specification and management

## 12.1 `CheckpointSpec`

```python
@dataclass(frozen=True)
class CheckpointSpec:
    checkpoint_id: str
    source_type: CheckpointSourceType
    url: str | None
    repository_id: str | None
    revision: str | None
    filename: str | None
    expected_sha256: str | None
    observed_sha256: str | None
    format: str
    loader: str
    license: LicenseRecord
```

Supported checkpoint sources MUST include:

```text
https
github_release
huggingface
package_bundle
local_path
```

## 12.2 Cache

Resolved checkpoints MUST be stored under:

```text
.torch-dae/checkpoints/<checkpoint-id>/<sha256>/
```

Checkpoint files MUST NOT be committed.

## 12.3 Checkpoint manager

The checkpoint manager MUST:

1. resolve the source;
2. acquire or locate the asset;
3. compute SHA-256;
4. compare it with the expected hash when available;
5. persist acquisition metadata;
6. return a local immutable path.

If no upstream checksum exists, the first verified local hash MAY become the committed observed hash.

## 12.4 License behavior

Checkpoint and source licenses MUST be recorded.

An absent, ambiguous, or restrictive license MUST NOT automatically prevent:

* analysis;
* local checkpoint acquisition;
* local integration;
* runtime verification.

The framework MUST NOT present legal conclusions.

---

# 13. Canonical model input contract

All public model wrappers MUST accept waveform input with:

```text
waveform shape: [B,C,T]
sample_rate: integer Hz
```

where:

* `B` is batch size;
* `C` is channel count;
* `T` is sample count.

Mono audio MUST use:

```text
C = 1
```

The default waveform dtype SHOULD be:

```text
float32
```

## 13.1 Valid lengths

Public waveform operations SHOULD accept:

```python
valid_lengths: torch.Tensor | None
```

with shape:

```text
[B]
```

Each value represents the number of valid, unpadded samples for the corresponding item.

When `valid_lengths` is `None`, all `T` samples are valid.

After resampling, wrappers MUST update valid lengths consistently.

## 13.2 Wrapper responsibilities

Every wrapper MUST internally perform or delegate:

* shape validation;
* dtype conversion where scientifically appropriate;
* channel adaptation;
* mono downmixing where required;
* resampling;
* padding or truncation;
* waveform normalization;
* native feature extraction;
* upstream layout conversion.

The public user MUST NOT be required to construct model-specific spectrograms.

## 13.3 Resampling

Automatic resampling MUST be enabled by the standard public waveform API.

The preprocessing interface MUST also permit strict control:

```python
model.preprocess(
    waveform,
    sample_rate,
    allow_resample=False,
)
```

When `allow_resample=False`, a mismatched sample rate MUST raise a precise error.

The model card MUST document:

* native sample rate;
* resampling implementation;
* channel policy;
* padding policy;
* normalization;
* duration constraints.

---

# 14. Public PyTorch wrapper API

Every integration MUST expose an ordinary `torch.nn.Module`.

```python
class AudioModel(torch.nn.Module):
    @classmethod
    def from_random(
        cls,
        *,
        variant: str | None = None,
        **architecture_kwargs,
    ) -> "AudioModel":
        ...

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: CheckpointSpec | str | Path | None = None,
        *,
        variant: str | None = None,
        **kwargs,
    ) -> "AudioModel":
        ...

    def load_checkpoint(
        self,
        checkpoint: CheckpointSpec | str | Path,
        *,
        strict: bool = True,
        map_location: str | torch.device = "cpu",
    ) -> None:
        ...

    def preprocess(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        *,
        valid_lengths: torch.Tensor | None = None,
        allow_resample: bool = True,
    ) -> PreprocessingOutput:
        ...

    def forward(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        *,
        valid_lengths: torch.Tensor | None = None,
    ) -> AudioModelOutput:
        ...

    def predict_probability(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        *,
        valid_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ...

    def available_embeddings(self) -> tuple[EmbeddingSpec, ...]:
        ...

    def compute_embedding(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        *,
        embedding_id: str | None = None,
        valid_lengths: torch.Tensor | None = None,
    ) -> EmbeddingOutput:
        ...
```

## 14.1 Random initialization

`from_random()` MAY be unsupported when the upstream implementation cannot reliably construct the architecture without checkpoint-bound configuration.

The model card MUST then state:

```json
{
  "random_initialization": {
    "supported": false,
    "reason": "..."
  }
}
```

The framework MUST NOT require unnecessary architecture reimplementation solely to provide random initialization.

## 14.2 `forward()`

`forward()` MUST return native differentiable task outputs.

It MUST NOT automatically apply:

* sigmoid;
* softmax;
* thresholding;
* label decoding;
* temporal event decoding.

For classification models, the primary output SHOULD normally be logits.

For representation models, codecs, SED, SELD, or other structures, the output MAY contain different typed components.

## 14.3 `predict_probability()`

`predict_probability()` MUST return only a probability tensor.

The activation MUST correspond to the official task definition:

* sigmoid for multilabel outputs;
* softmax for mutually exclusive classes;
* another documented transformation where officially defined.

Models without probabilistic outputs MUST raise:

```python
UnsupportedCapabilityError
```

Class labels MUST be exposed separately, for example:

```python
model.class_labels
```

## 14.4 No mandatory `predict()`

A universal `predict()` method is not part of the base API.

Task-specific decoding MAY later be exposed through specialized methods such as:

```text
predict_labels
decode_events
decode
reconstruct
```

---

# 15. Output types

## 15.1 `AudioModelOutput`

The universal output MUST preserve differentiability and model-specific topology.

A recommended structure is:

```python
@dataclass
class AudioModelOutput:
    primary: torch.Tensor | object
    tensors: Mapping[str, torch.Tensor]
    lengths: torch.Tensor | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    native_output: object | None = None
```

Requirements:

* all training-relevant tensors MUST remain attached to the autograd graph;
* `primary` identifies the model’s main native result;
* named tensors provide stable access;
* `native_output` MAY preserve the original upstream return object.

## 15.2 `EmbeddingOutput`

```python
@dataclass
class EmbeddingOutput:
    embedding_id: str
    tensor: torch.Tensor
    layout: str
    lengths: torch.Tensor | None = None
    timestamps: torch.Tensor | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
```

---

# 16. Embedding specification

Every scientifically meaningful embedding candidate SHOULD remain accessible.

## 16.1 `EmbeddingSpec`

Each candidate MUST describe:

```json
{
  "embedding_id": "cnn14.global_embedding",
  "name": "Global pooled CNN14 representation",
  "description": "...",
  "officially_defined": true,
  "default": true,
  "network_location": "...",
  "layout": "B,D",
  "dimension": 2048,
  "granularity": "clipwise",
  "temporal_hop_seconds": null,
  "pooling": "global max plus average pooling",
  "projection": "none",
  "normalization": "none",
  "task_head_relation": "before classifier",
  "dtype": "float32",
  "status": "verified",
  "selection_rationale": "...",
  "evidence_ids": []
}
```

## 16.2 Multiple embeddings

After a default embedding has been selected:

* all other valid candidates MUST remain accessible;
* `available_embeddings()` MUST return all supported candidates;
* `compute_embedding()` MUST accept an explicit `embedding_id`.

## 16.3 Default selection

If the upstream project formally defines an embedding, it SHOULD be the default unless incompatible with the requested use.

If no formal definition exists, the onboarding skill MUST:

1. identify all meaningful candidates;
2. explain their architectural positions;
3. explain pooling, projection, normalization, dimensionality, and temporal granularity;
4. explain downstream implications;
5. request user selection;
6. record the selected `default_embedding_id`.

Classifier logits and task decisions MUST NOT be presented as embeddings.

---

# 17. Device semantics

A wrapper MUST behave as a standard PyTorch module:

```python
model.to(device)
```

Internal parameters, buffers, temporary tensors, and preprocessing operations MUST follow the selected device whenever supported.

The model card MUST separate:

```json
{
  "device_support": {
    "upstream_declared": ["cpu", "cuda"],
    "locally_tested": ["cpu", "mps"],
    "known_limitations": []
  }
}
```

Initial development and verification will occur on:

* Apple CPU;
* Apple MPS where supported.

The architecture MUST remain capable of supporting CUDA environments later.

MPS or CUDA support MUST NOT be inferred solely from generic PyTorch availability.

---

# 18. Model-card schema

A complete checkpoint-specific model card MUST contain the following top-level sections:

```json
{
  "schema_version": "1.0.0",
  "card_id": "...",
  "card_status": "runtime_verified",
  "identity": {},
  "sources": {},
  "scientific_reference": {},
  "description": {},
  "tasks": {},
  "datasets": {},
  "reported_metrics": [],
  "usage": {},
  "input": {},
  "outputs": {},
  "embeddings": {},
  "capabilities": {},
  "device_support": {},
  "architectural_profiling": {},
  "inference_profiling": {},
  "energy_profiling": {},
  "limitations": [],
  "issues": [],
  "evidence": []
}
```

## 18.1 Identity

```json
{
  "model_name": "PANNs",
  "model_family": "PANNs",
  "variant": "Cnn14",
  "checkpoint_name": "AudioSet Cnn14",
  "framework": "pytorch",
  "wrapper_entry_point": "torch_dae.models.panns:PannsCnn14"
}
```

## 18.2 Sources

Sources MUST distinguish:

* official scientific repository;
* implementation repository or package;
* checkpoint source;
* wrapper source where different.

Every Git repository SHOULD record an immutable revision.

## 18.3 Scientific reference

```json
{
  "title": "...",
  "doi": "...",
  "official_publication": "...",
  "authors": [
    "Q. Kong",
    "Y. Cao",
    "T. Iqbal",
    "Y. Wang",
    "W. Wang",
    "M. D. Plumbley"
  ],
  "year": 2020
}
```

Author names MUST be complete and formatted consistently in IEEE bibliography style.

One canonical publication year MUST be used.

## 18.4 Description

The description MUST separately discuss:

* architectural design;
* preprocessing;
* training objective;
* checkpoint-specific behavior;
* implementation characteristics.

## 18.5 Tasks

Tasks MUST be separated into:

```json
{
  "pretraining": [],
  "finetuning": [],
  "official_evaluation": [],
  "supported_inference": []
}
```

## 18.6 Datasets

Datasets MUST distinguish:

```json
{
  "training": [],
  "validation": [],
  "testing": []
}
```

A dataset record SHOULD support:

* name;
* version;
* subset;
* split;
* role;
* official evidence.

## 18.7 Reported metrics

Metrics MUST be represented as records, not a flat dictionary:

```json
{
  "task": "audio_tagging",
  "dataset": "AudioSet",
  "split": "evaluation",
  "metric": "mAP",
  "value": 0.431,
  "unit": null,
  "protocol": "...",
  "checkpoint_specific": true,
  "source_status": "officially_reported",
  "evidence_ids": []
}
```

## 18.8 Usage

Usage MUST reference the committed environment specification:

```json
{
  "recommended_environment": {
    "environment_id": "panns-cnn14-audioset",
    "specification": "environments/panns-cnn14-audioset/environment.json",
    "lockfile": "environments/panns-cnn14-audioset/uv.lock",
    "verified": true
  },
  "installation_commands": [],
  "checkpoint_loading": [],
  "smoke_test_command": "torch-dae model verify panns-cnn14-audioset"
}
```

## 18.9 Profiling sections

Before profiling, each section MUST remain valid with:

```json
{
  "status": "not_profiled"
}
```

Model-card creation and runtime verification MUST NOT depend on profiling completion.

---

# 19. Repository-analysis skill

The project MUST provide one canonical skill:

```text
audio-model-onboarding
```

The same canonical skill directory MUST be exposed to:

* Codex;
* Claude Code.

## 19.1 Skill inputs

The minimum user input is:

* official repository URL.

Optional inputs include:

* requested variant;
* requested checkpoint;
* official paper URL;
* intended task;
* intended default embedding;
* target local platform.

The skill MUST remain independent of the previous project’s backbone JSON files.

## 19.2 Skill modes

The skill MUST support the following internal modes.

### `analyze`

Produces a technical report covering:

* repository identity;
* architecture;
* available variants;
* checkpoints;
* tasks;
* datasets;
* metrics;
* environment evidence;
* preprocessing;
* forward outputs;
* candidate embeddings;
* unresolved questions.

### `resolve-environment`

Determines a functioning environment and freezes it.

### `integrate`

Creates the unified PyTorch wrapper.

### `verify`

Loads the checkpoint and verifies runtime behavior.

### `card`

Creates or updates the checkpoint-specific model card.

### `profile`

Reserved for the later profiling subsystem.

## 19.3 Chat report

After repository analysis, the skill MUST report its findings in the chat before silently making scientific choices that require user judgement.

The report SHOULD clearly distinguish:

* established upstream facts;
* locally observed behavior;
* inferences;
* unresolved decisions.

When embedding selection is ambiguous, the skill MUST present all meaningful candidates and request the user’s decision.

---

# 20. Environment-resolution protocol

Environment resolution MUST be evidence-driven rather than based on a small arbitrary number of attempts.

## 20.1 Evidence collection

The skill MUST inspect, when available:

* repository date and commit history;
* package metadata;
* `requirements.txt`;
* `pyproject.toml`;
* `setup.py`;
* Conda environments;
* Dockerfiles;
* CI workflows;
* documentation;
* framework APIs used by the source;
* checkpoint release date;
* package release history.

## 20.2 Candidate compatibility space

The agent MUST derive a plausible compatibility space over:

```text
Python × PyTorch × TorchAudio × NumPy × principal dependencies
```

The historical period and source APIs MUST influence the initial candidates.

The newest package versions MUST NOT be assumed to be the correct starting point.

## 20.3 Failure classification

Every failed attempt SHOULD be classified as one of:

```text
python_version_unavailable
package_version_unavailable
dependency_conflict
binary_or_abi_incompatibility
removed_api
torch_torchaudio_mismatch
numpy_compatibility
source_build_failure
repository_defect
checkpoint_incompatibility
missing_asset
unsupported_platform
runtime_error
```

The next attempt MUST be motivated by previous evidence or failure analysis.

## 20.4 Termination

Resolution terminates when:

* a fully working environment is found;
* a required source or checkpoint is unavailable;
* the plausible compatibility space has been methodically exhausted;
* an upstream defect prevents execution;
* the model is demonstrably unsupported on the current platform.

The framework MUST NOT impose an arbitrary limit such as four attempts.

## 20.5 Freeze

After success, the onboarding process MUST:

* fix the Python version;
* fix direct package versions;
* generate the model-specific lock file;
* record platform information;
* freeze source revisions;
* execute an environment verification;
* commit only specifications and lock data.

---

# 21. Runtime-verification protocol

A card reaches `runtime_verified` only after the wrapper successfully verifies the applicable capabilities.

## 21.1 Required checks

```python
model = Model.from_random(...)
random_output = model(
    waveform,
    sample_rate,
    valid_lengths=valid_lengths,
)

model.load_checkpoint(checkpoint_spec)

pretrained_output = model(
    waveform,
    sample_rate,
    valid_lengths=valid_lengths,
)

embedding = model.compute_embedding(
    waveform,
    sample_rate,
    embedding_id=model.default_embedding_id,
    valid_lengths=valid_lengths,
)
```

When random initialization is unsupported, the verification report MUST record that explicitly.

## 21.2 Runtime checks

The verification MUST cover:

* canonical `[B,C,T]` input;
* batch size greater than one where feasible;
* mono input;
* mismatched input sample rate and automatic resampling;
* strict no-resampling mode;
* variable valid lengths;
* forward output topology;
* `predict_probability()` where supported;
* every declared embedding candidate;
* checkpoint loading;
* CPU execution;
* MPS execution where supported and locally available;
* movement through `.to(device)`;
* differentiability for training-relevant outputs;
* offline reuse after dependencies and checkpoint have been cached.

## 21.3 Verification report

Each model integration MUST have a committed verification report containing:

* environment ID and fingerprint;
* platform;
* device;
* checkpoint hash;
* input contracts;
* output names;
* tensor ranks and shapes;
* dtypes;
* embedding results;
* passed and unsupported capabilities;
* known limitations.

Large runtime outputs and checkpoints MUST remain ignored.

---

# 22. Registry

The project SHOULD derive the model registry from validated model cards.

The registry MUST support:

```python
from torch_dae import registry

cards = registry.list_cards()
card = registry.get_card("panns-cnn14-audioset")
model_class = registry.get_model_class("panns-cnn14-audioset")
```

The registry MUST NOT import model-specific packages while merely listing model cards.

Wrapper imports SHOULD be lazy.

---

# 23. Root CLI

The first stable CLI SHOULD expose:

```bash
torch-dae card list
torch-dae card show <card-id>
torch-dae card validate <card-id>

torch-dae env create <card-id>
torch-dae env ensure <card-id>
torch-dae env verify <card-id>
torch-dae env remove <card-id>
torch-dae env info <card-id>
torch-dae env run <card-id> -- <command>

torch-dae checkpoint ensure <card-id>
torch-dae checkpoint info <card-id>
torch-dae checkpoint remove <card-id>

torch-dae model inspect <card-id>
torch-dae model verify <card-id>
```

The control-plane CLI MUST remain usable without installing PyTorch in the root environment.

---

# 24. Schema and validation requirements

All committed JSON artifacts MUST validate against strict Draft 2020-12 schemas.

Principal schemas MUST use:

* explicit nested properties;
* enums;
* URI formats;
* full Git revision patterns;
* SHA-256 patterns;
* conditional requirements;
* `additionalProperties: false`, except for explicitly designed extension maps.

Validation MUST cover:

* model cards;
* environments;
* checkpoints;
* embeddings;
* verification reports.

Pydantic or an equivalent typed Python model SHOULD mirror each principal schema.

Schema fixtures MUST include both valid and semantically invalid cases.

---

# 25. Testing strategy

## 25.1 Core tests

Core tests MUST verify:

* canonical input validation;
* valid-length validation;
* capability errors;
* output dataclasses;
* embedding selection;
* checkpoint specifications;
* registry lazy loading.

## 25.2 Environment tests

Environment tests MUST verify:

* deterministic fingerprints;
* `create`, `ensure`, `verify`, `remove`, and `info`;
* environment reuse;
* rebuild after specification change;
* model environment isolation;
* root environment non-contamination;
* managed child-process execution.

Fixture environments MAY use lightweight packages instead of real models.

## 25.3 Schema tests

Every schema MUST be tested with:

* valid fixtures;
* missing required fields;
* invalid enum values;
* malformed hashes;
* inconsistent lifecycle fields;
* inconsistent counts or references where applicable.

## 25.4 Skill tests

The skill MUST be evaluated against repository fixtures covering:

* installable official package;
* installable Git repository;
* repository requiring minimal vendoring;
* ambiguous embedding definitions;
* unpinned requirements;
* checkpoint hidden behind a helper;
* non-PyTorch repository with an available PyTorch wrapper.

## 25.5 Model tests

Every integration MUST have:

* card validation;
* environment recreation test;
* wrapper construction test;
* checkpoint loading test;
* forward smoke test;
* probability test where supported;
* all-embedding test;
* device test;
* verification-report consistency test.

---

# 26. MVP pilot models

The initial implementation SHOULD be validated using three heterogeneous integrations.

## 26.1 PANNs Cnn14

Purpose:

* classification;
* multilabel probabilities;
* wrapper/package analysis;
* clipwise embedding;
* AudioSet checkpoint.

## 26.2 BYOL-A

Purpose:

* representation-learning model;
* explicit preprocessing;
* multiple internal representations;
* custom repository reconstruction.

## 26.3 EnCodec 48 kHz

Purpose:

* neural codec;
* non-classification forward output;
* continuous encoder latent;
* quantized latent;
* discrete codes;
* optional probability capability unsupported.

The three pilots MUST be integrated sequentially. Lessons from each SHOULD update the core API and skill before bulk onboarding.

---

# 27. Profiling boundary

Profiling is explicitly deferred until the pilot wrappers are runtime-verified.

The future profiling subsystem will distinguish:

* architecture-only analysis;
* model-forward profiling;
* preprocessing profiling;
* end-to-end profiling;
* CPU;
* CUDA;
* MPS;
* RAM;
* accelerator memory;
* latency;
* throughput;
* real-time factor;
* energy.

The initial schemas MUST reserve profiling sections, but no profiling implementation is required in the bootstrap or skill MVP.

---

# 28. Implementation sequence

The new project SHOULD be implemented through the following bounded phases.

## Phase 00 — Repository bootstrap and normative contracts

Deliver:

* fresh repository scaffold;
* root control-plane package;
* project instructions;
* strict schemas;
* typed domain models;
* CLI skeleton;
* test and quality configuration;
* canonical skill links;
* ignored runtime layout.

No real model integration.

## Phase 01 — Environment and checkpoint core

Deliver:

* environment specifications;
* fingerprinting;
* environment manager;
* root environment CLI;
* source-installation hierarchy;
* checkpoint manager;
* fixture-based integration tests.

## Phase 02 — Skill MVP

Deliver:

* canonical `SKILL.md`;
* focused reference workflows;
* report template;
* model-card template;
* repository-inspection utilities;
* environment-resolution protocol;
* skill evaluations for both agents.

## Phase 03 — PANNs pilot

Deliver complete onboarding from repository analysis through runtime verification.

## Phase 04 — BYOL-A pilot

Stress representation learning, preprocessing, and embedding alternatives.

## Phase 05 — EnCodec pilot

Stress non-classification outputs and multiple latent forms.

## Phase 06 — Core stabilization

Freeze:

* public API;
* schemas;
* environment workflow;
* skill workflow;
* onboarding conventions.

## Phase 07 — Model-family expansion

Reanalyze and integrate additional models directly from their original repositories.

## Phase 08 — Profiling subsystem

Implement architectural, inference, memory, and energy profiling.

---

# 29. Phase 00 acceptance criteria

The repository-bootstrap phase is accepted only if:

1. the root folder is `torch-dae`;
2. the root Git repository is clean after commit;
3. no legacy backbone files are included;
4. the control-plane environment contains no model-specific dependency;
5. the canonical skill exists once under `skills/audio-model-onboarding/`;
6. Codex and Claude project skill paths resolve to the canonical skill;
7. all principal JSON schemas are strict and valid;
8. typed Python models exist for all principal schemas;
9. model-card lifecycle states are enforced;
10. the canonical `[B,C,T]` contract is represented;
11. `valid_lengths` semantics are represented;
12. the public API is defined but contains no model implementation;
13. the environment-manager interface is defined;
14. the checkpoint-manager interface is defined;
15. the root CLI loads without PyTorch;
16. `.torch-dae/` is fully ignored;
17. tests, Ruff, mypy, build, and schema validation pass;
18. documentation reflects this specification;
19. profiling remains a declared future capability;
20. no pilot model implementation has started.

---

# 30. Project invariants

The following invariants apply throughout development:

1. one model card represents one model–variant–checkpoint tuple;
2. all public audio inputs use `[B,C,T]` plus sample rate;
3. wrappers own model-specific preprocessing;
4. `forward()` returns raw differentiable output;
5. `predict_probability()` returns only a probability tensor;
6. every valid embedding remains accessible;
7. environments are model-specific and reproducible;
8. environment creation is explicit;
9. checkpoint assets are cached but never committed;
10. licenses are recorded but non-blocking;
11. the root control-plane environment remains lightweight;
12. official package is preferred, then pinned repository, then minimal vendoring;
13. repository analysis uses primary upstream evidence;
14. legacy backbone JSON files are not project inputs;
15. profiling begins only after runtime verification;
16. unresolved information is represented explicitly rather than rhetorically strengthened.

---

# 31. Specification freeze

The following interfaces are considered frozen for the first implementation cycle:

* checkpoint-specific model cards;
* model-card lifecycle states;
* `[B,C,T]` waveform input;
* optional `valid_lengths`;
* automatic resampling with strict opt-out;
* `from_random()`;
* `from_pretrained()`;
* `load_checkpoint()`;
* `forward()`;
* `predict_probability()`;
* `available_embeddings()`;
* `compute_embedding()`;
* `CheckpointSpec`;
* model-specific environments;
* `EnvironmentManager`;
* root environment CLI;
* source-installation priority;
* licenses as informational metadata;
* one canonical onboarding skill for Codex and Claude;
* no legacy backbone dependency;
* profiling postponed until verified pilot integrations.
