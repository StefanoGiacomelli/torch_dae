# `torch-dae` Project Specification

**Document status:** Proposed normative baseline
**Specification version:** `0.2.0`
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
7. optional repeatable architectural, runtime, memory, and energy profiling through immutable Technical Cards.

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

Profiling evidence is not part of model integration identity. An accepted integration MAY have zero
or more independent Technical Cards. Technical Cards reference the accepted Model Card and
checkpoint but MUST NOT mutate, enrich, or lifecycle-promote that Model Card.

---

# 5. Model-card lifecycle

For all new onboarding workflows, the normative Model Card lifecycle is:

```text
draft
analyzed
environment_resolved
checkpoint_verified
runtime_verified
```

`runtime_verified` is the terminal onboarding state. Profiling is optional evidence collected
outside Model Card lifecycle progression.

The current Model Card schema `1.0.0` and Python enum still accept `profiled` and embedded profiling
placeholders for backward compatibility with the original bootstrap contract. Those fields are
**legacy/reserved compatibility surface**. New profiling workflows MUST NOT promote a Model Card to
`profiled`, MUST NOT rewrite accepted Model Card profiling placeholders, and MUST store profiling
evidence only in independent Technical Cards.

## 5.1 `draft`

The card exists, but repository/scientific metadata analysis is incomplete.

## 5.2 `analyzed`

Official repository, architecture implementation, scientific publication, checkpoint source,
preprocessing, forward path, candidate embeddings, package/environment evidence, and unresolved
facts have been inspected.

## 5.3 `environment_resolved`

A referenced model environment has been successfully constructed, verified, and frozen before the
card claims this status. The card references the matching environment fingerprint and passed
environment-verification evidence.

## 5.4 `checkpoint_verified`

The selected checkpoint has been acquired, hashed, loaded into the intended architecture, and
checked for serialization/state compatibility.

## 5.5 `runtime_verified`

The public wrapper has passed strict target-aware runtime verification for canonical waveform input,
construction, checkpoint loading, forward inference, declared outputs/embeddings, declared device
behavior, and gradient behavior where required.

The card references its strict runtime-verification target and matching checkpoint-specific report
by repository path and SHA-256.

## 5.6 Legacy/reserved `profiled` compatibility

`profiled` remains readable in Model Card schema `1.0.0` only for backward compatibility. It is not
a legal target state for new onboarding or Technical Card profiling. A future major Model Card
schema revision MAY remove that compatibility surface through an explicit migration.

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
│   └── <environment-id>/
│       ├── environment.json
│       ├── pyproject.toml
│       ├── uv.lock
│       ├── sources.json
│       └── verify_environment.py
│
├── onboarding_reports/
│   └── <workflow-id>/
│       ├── workflow.json
│       ├── analyze/
│       │   ├── handoff.json
│       │   └── <canonical phase outputs>
│       ├── resolve-environment/
│       │   ├── handoff.json
│       │   └── <canonical phase outputs>
│       ├── integrate/
│       │   ├── handoff.json
│       │   └── <canonical phase outputs>
│       └── card/
│           ├── handoff.json
│           └── <canonical phase outputs>
│
├── verification_reports/
│   └── <family>/
│       └── <card-id>.json
│
├── technical_cards/
│   └── <model-id>/
│       ├── <technical-card-id>.json
│       └── <technical-card-id>.npz
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
    ├── workspaces/
    └── profiling/
```

The complete `.torch-dae/` directory MUST be ignored by Git.

No legacy backbone JSON files or parsed legacy hints MUST be included in the new repository.

`onboarding_reports/` contains accepted, bounded, reviewable static-analysis, planning, and
environment-resolution handoffs produced before runtime verification. Phase directories MAY be
absent until accepted. Canonical JSON and Markdown files MUST remain decompressed and diffable.
Checkpoint payloads, complete command logs, source clones, virtual environments, caches, coverage
output, and duplicate transport archives MUST NOT be committed there.

`verification_reports/` retains one meaning only: committed checkpoint-specific runtime
observations created by `verify`. Analysis reports, environment-resolution reports, integration
plans, handoff manifests, and audit bundles MUST NOT be stored there. Ignored diagnostic and
execution logs belong under `.torch-dae/reports/`. External review bundles are on-demand transport
and audit artifacts and MUST be generated outside the repository.

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

Every integration tuple MUST reference a reproducible environment. Multiple tuples MAY reference
one logical environment ID when their accepted dependency and source contracts are identical.

The environment specification is committed under:

```text
environments/<environment-id>/
```

The materialized virtual environment is stored under:

```text
.torch-dae/environments/<environment-id>/<fingerprint>/
```

Model-specific environments MUST install:

* the correct Python interpreter;
* the local `torch-dae` package;
* model runtime dependencies;
* the selected upstream source or package;
* only dependencies required by that integration.

Before materialization, the environment manager MUST build or reuse the deterministic local
package wheel without network access and validate that every active `Requires-Dist` requirement is
satisfiable from the accepted lock under the canonical installation policy. Marker evaluation MUST
use the environment's selected Python version and platform, optional extras MUST remain inactive
unless explicitly selected, and only direct or transitive lock entries reachable from the locked
environment project may satisfy the check. Missing and incompatible versions MUST remain distinct
structured evidence. This preflight is environment-lock completeness evidence only; it MUST NOT
create an environment, import a model, require a model card, or establish environment verification,
checkpoint compatibility, or runtime verification.

The local package identity MUST be content-addressed exclusively over deterministic wheel build
inputs and MUST use a stable form such as `content-sha256:<digest>`. Identical package inputs MUST
have the same identity before and after a Git commit, after a HEAD-only change, and outside a Git
repository. Git HEAD and cleanliness MAY be recorded separately as `repository_head` and
`repository_dirty`, with `package_content_sha256`, but this provenance MUST NOT affect package
identity, wheel-cache keys, environment fingerprints, offline reuse, or eligibility.

Concurrent requests for one local package identity MUST coordinate through a bounded,
inter-process-safe managed-runtime lock. At most one authoritative builder may publish; waiters MUST
reuse the completed validated cache entry. Process-specific build paths, safe stale-lock handling,
failure cleanup, and atomic cache publication are required. One process MUST NOT delete another
process's valid entry, and distinct identities MUST NOT be serialized by a global lock. Manual
serialization of environment materialization is not the required race workaround.

---

# 9. Environment specification

An environment specification MUST include at least:

```json
{
  "schema_version": "1.0.0",
  "environment_id": "panns-cnn14-audioset",
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

`model_card_id` is an optional legacy association. It MUST NOT authorize materialization and is not
required for new environment definitions. The five accepted files under the environment-ID
directory are the complete materialization authority.

## 9.1 Environment fingerprint

The environment fingerprint is a lowercase SHA-256 of canonically serialized, deterministically
ordered evidence and MUST depend on:

* canonical environment ID and exact `environment.json` SHA-256;
* exact `pyproject.toml`, `uv.lock`, `sources.json`, and `verify_environment.py` SHA-256 values;
* exact Python implementation and version;
* normalized target platform;
* exact direct-dependency names and locked versions;
* every referenced integrated or vendored source hash;
* relevant local `torch-dae` build identity.

Changing any of these inputs MUST produce a different fingerprint.
Model-card prose, checkpoint bytes, absolute paths, timestamps, usernames, temporary paths, and
mutable log paths MUST NOT contribute to the fingerprint. Git HEAD and repository cleanliness are
informational provenance and MUST NOT contribute. Logical environment IDs remain distinct even when
their dependency evidence is otherwise equivalent.

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

Performed after resolution using the card-independent interface:

```bash
torch-dae env materialize <environment-id>
torch-dae env verify <environment-id>
```

Recreation MUST use the committed environment specification and lock file. It MUST NOT repeat compatibility research.
The older `env create` and `env ensure` card-oriented commands remain convenience adapters: they
resolve the card's recommended environment ID and delegate to the same direct primitives.

## 9.3 Onboarding phase handoffs

Every cross-conversation onboarding workflow MUST have a stable canonical workflow ID and MAY commit
accepted phase outputs under `onboarding_reports/<workflow-id>/`. The workflow record and each
accepted handoff MUST use strict Draft 2020-12 contracts with closed objects, canonical identifiers,
repository-relative paths, and validated SHA-256 values.

A workflow record MUST identify the model family, target variant/checkpoint/card scopes, repository
commit at creation, accepted phase paths, current accepted phase, and active or completed status.
Each handoff MUST identify its workflow and phase, draft/accepted/superseded status, producing
repository commit, `project_spec.md` hash, canonical skill fingerprint, hashed input and output
artifacts, consumed user decisions, carried unresolved items, validation result, target scopes, and
allowed next modes. An accepted handoff MUST have passed its required validation. A superseding
handoff MUST record the SHA-256 of the prior accepted handoff.

Each handoff MAY carry a strict `artifact_supersessions` collection for shared repository outputs
that legally evolve in a later workflow phase. Every entry MUST record one repository-relative
path, its latest accepted originating phase and SHA-256, the new SHA-256, a non-empty reason, and an
optional prior handoff SHA-256 when needed for unambiguous lineage. The containing later phase MUST
declare that path as its own output with the new hash. Duplicate, stale, ambiguous, cyclic, forked,
or backward transitions MUST be rejected.

Historical handoffs, canonical phase reports, and their recorded artifact declarations MUST remain
immutable evidence. Full-workflow validation MUST preserve their hashes while checking the current
filesystem against the latest accepted declaration in each model-integration artifact chain. A
changed workflow-owned artifact without a valid supersession MUST fail. Shared control-plane files
MAY evolve outside a model workflow; their accepted declarations remain historical evidence and
their current state is validated by repository-wide schema, test, and static validation instead of
being misreported as a model-artifact mutation. Discovery of an earlier accepted phase after a later
transition MUST report the affected external artifact as superseded, not corrupted.

Accepted handoff control-plane hashes are immutable historical provenance. Validation MUST require
their presence and valid SHA-256 syntax, preserve them byte-for-byte, and report separately whether
the recorded canonical-skill fingerprint or project-specification hash differs from the current
repository. Such drift is informational and MUST NOT invalidate an accepted handoff. A historical
hash proves the control-plane identity declared by the phase; by itself it does not retain or
reconstruct those bytes. A producing Git revision MUST NOT be claimed as the source of uncommitted
control-plane bytes merely because the handoff records that repository commit.

Pending handoff control-plane hashes are current-repository validation inputs. Before promotion, the
pending phase MUST record and exactly match the current canonical-skill fingerprint and current
`project_spec.md` SHA-256. It MUST NOT copy a prerequisite handoff's historical control-plane hashes.
A mismatch MUST fail before atomic promotion changes accepted workflow bytes. A new phase MAY consume
accepted prerequisites from older control planes without rewriting them; its own current hashes and
the reported historical/current identities make that control-plane transition explicit.

Artifact supersession MUST NOT target accepted handoffs, canonical phase reports, workflow history,
`project_spec.md`, generated schemas, verification reports, checkpoints, credentials, or ignored
runtime state. External attachments MAY be retained as digest-only evidence, but MUST NOT become
required future local paths. Handoffs from different workflows MUST NOT be combined silently. A
handoff MUST NOT claim a model-card lifecycle promotion that its underlying report does not claim.

The root control plane MUST provide deterministic `discover`, `validate`, `promote`, `bundle`,
`cleanup`, `finalize`, and `run-manifest create` operations through
`scripts/onboarding_handoff.py`. Discovery MUST search only the
committed `onboarding_reports/` root. Promotion MUST validate before mutation, use an atomic
temporary sibling and rename, reject partial or arbitrary runtime content, and require explicit
supersession before replacing an accepted handoff.
Promotion of a later phase with artifact supersessions MUST validate the complete accepted history
plus the pending handoff, exact prior and new hashes, the later output declaration, and the current
repository file before mutation. Failure MUST leave the accepted workflow byte-identical and MUST
NOT weaken the exact promotion allowlist.

Every phase execution MUST use either a context-managed system temporary directory or:

```text
.torch-dae/workspaces/<workflow-id>/<phase>/<run-id>/
```

A managed run manifest MUST record its run/workflow/phase identities, start time, repository commit,
created, reused, external, and retained paths, retained reasons, and cleanup result. Successful
closure MUST validate in the workspace, promote accepted canonical artifacts, generate the review
bundle, remove recorded ephemeral workspaces and trial environments, verify removal, and report
retained managed caches. Failure handling MUST preserve only bounded diagnostics required to
classify the failure and MUST leave no unidentified temporary paths.

Cleanup categories are: ephemeral workflow workspaces; failed or completed trial environments;
reusable repository/package caches; materialized model environments; checkpoint caches; and external
audit outputs. Default cleanup MAY remove only the first two categories for the selected workflow
and completed run. All deletion targets MUST be recorded by a managed run manifest. Reusable caches,
materialized environments, checkpoints, and external audit outputs require explicit flags or
external user action and MUST NOT be removed implicitly.

Cleanup and finalize MUST distinguish a genuinely legacy workflow from unmanaged workspace content
using existing lifecycle/control-plane evidence, never a hard-coded workflow identity. A workflow
whose `.torch-dae/workspaces/<workflow-id>/` scope contains no phase/run directories at all — the
legacy case, for example a workflow completed before run manifests were adopted — MUST NOT raise a
hard failure; cleanup MUST return a structured, successful non-action result reporting
`status: "not_applicable"` and `reason: "no_managed_run_manifests"` with an empty removal set, and
exit successfully. A run directory that is present under that scope but carries no manifest of its
own is unmanaged content, not the legacy case: it MUST be reported and MUST block cleanup and
finalize from succeeding, rather than being silently classified as `not_applicable` or deleted.
Ownership MUST NOT be guessed from unrecorded paths.

The root control plane MUST provide one canonical `run-manifest create` entry point that is the
allocator for a managed workspace run: it atomically creates the run's own workspace directory and
writes its manifest in the same call and returns the absolute run-root and manifest paths. Every
lifecycle mode that needs a managed workspace MUST call it before performing any managed workspace
work and MUST perform that work inside the returned run root, rather than creating an independent
workspace directory first and registering it afterward.

The root control plane MUST also provide one canonical `finalize` operation composing `validate`,
the bounded required repository gates, `cleanup`, and `bundle` for a completed lifecycle phase — it
MUST NOT duplicate any validator's logic. It MUST, in order: re-validate every accepted phase's
declared artifacts against the current repository; run the bounded required repository-level gates
(at minimum repository validation, canonical skill validation, staged-equivalent worktree
validation, and `git diff --check`); run a cleanup preflight, which blocks the entire call when it
reports unmanaged workspace content or any other blocking condition; generate the deterministic
review bundle through the requested phase, embedding the required-gate results, lifecycle state,
evidence-invariance summary, and cleanup preflight as archive metadata so the archive is
self-sufficient for independent review; and, only after that immutable archive exists, optionally
perform real cleanup execution, whose result and durable receipt are recorded — by absolute path and
hash — in one machine-readable `finalize-result.json` alongside every other generated artifact's
absolute path. Any required-gate failure, unaccepted phase, evidence drift, or unmanaged workspace
content MUST make the whole call fail rather than produce a successful finalization result. The full
test suite is a separate development/CI gate and MUST NOT be required by normal per-phase
finalization.

Because full-workflow validation already recomputes and compares the recorded SHA-256 of every
accepted handoff's declared artifacts, validates every declared `artifact_supersessions` transition,
and matches every latest external-artifact declaration, `finalize` MUST fail outright — not merely
warn — when any undeclared mutation is present. Evidence-invariance reporting MUST distinguish
unchanged phase-local accepted evidence, validated current external evidence, and validated declared
historical supersessions from each other; it MUST NOT describe a legally superseded shared artifact
as if every historical byte were unchanged.

Review bundles MUST be deterministic, produced exclusively with Python `tarfile`, and normalize
ownership, names, timestamps, modes, gzip metadata, and extended archive metadata. They MUST contain
the workflow and accepted artifacts through the requested phase, referenced committed artifacts,
repository identity and status, staged/unstaged/untracked inventories and diffs, a tracked-file
manifest, specification and skill fingerprints, artifact hashes and sizes, declared and actual
archive inventories, and a bundle result. When requested, the bundle MUST include a working-tree
snapshot excluding Git metadata and ignored runtime/build/cache/checkpoint state. Declared and
actual inventories MUST be compared independently.
Bundle metadata and results MUST report every included phase's historical canonical-skill
fingerprint and project-specification SHA-256, the current values, and separate skill,
specification, and aggregate drift states. Historical and current values MUST NOT be presented as
identical when drift exists.
For a bundle through a superseding phase, every historical handoff and declaration MUST remain
present, supersession chains and affected paths MUST be explicit, and each shared external artifact
MUST appear with its latest accepted repository bytes without claiming that an earlier hash equals
the current file.

The final validation matrix for a phase MUST begin only after the current handoff and every
repository evidence artifact have been generated. From that point, mutation of source, tests,
schemas, documentation, environment definitions, reports, workflow records, handoffs, or superseded
handoff archives invalidates the entire final matrix and requires it to be rerun. The final
staged-equivalent inventory and exact current handoff SHA-256 MUST be recorded and rechecked after
the matrix. Deterministic audit archives MUST then be generated automatically and read-only outside
the repository; no archive member may come from a repository file absent from the recorded final
inventory.

---

# 10. Environment-management API

Environment creation MUST be a dedicated subsystem and MUST NOT occur implicitly inside model initialization.

## 10.1 Python API

```python
from torch_dae.environment import EnvironmentManager

manager = EnvironmentManager.from_repository_root()

definition = manager.resolve_environment("panns-cnn14-audioset")
materialization = manager.materialize_environment("panns-cnn14-audioset")
verification = manager.verify_environment(
    "panns-cnn14-audioset",
    expected_fingerprint=definition.environment_fingerprint,
)
```

The manager MUST expose:

```python
class EnvironmentManager:
    def resolve_environment(
        self, environment_id: str
    ) -> ResolvedEnvironmentDefinition:
        ...

    def materialize_environment(
        self,
        environment_id: str,
        *,
        expected_spec_sha256: str | None = None,
    ) -> EnvironmentMaterializationResult:
        ...

    def preflight_environment(
        self, environment_id: str
    ) -> EnvironmentDependencyClosureResult:
        ...

    def verify_environment(
        self,
        environment_id: str,
        *,
        expected_fingerprint: str | None = None,
    ) -> EnvironmentVerificationResult:
        ...

    # Backward-compatible card-oriented convenience methods.
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
torch-dae env resolve <environment-id>
torch-dae env preflight <environment-id>
torch-dae env materialize <environment-id>
torch-dae env verify <environment-id>
torch-dae env remove <card-id>
torch-dae env info <card-id>
torch-dae env run <card-id> -- <command>
```

Required semantics:

| Command  | Behavior                                                                                 |
| -------- | ---------------------------------------------------------------------------------------- |
| `create` | Creates a new environment and fails if a valid or invalid materialization already exists |
| `ensure` | Reuses a valid environment or creates/rebuilds it                                        |
| `resolve` | Validates and fingerprints accepted environment artifacts without materializing         |
| `preflight` | Proves the accepted lock can satisfy active local-wheel runtime requirements offline |
| `materialize` | Creates or reuses accepted locked environment infrastructure                       |
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
    schema_version: str
    checkpoint_id: str
    source_type: CheckpointSourceType
    url: str | None
    repository_id: str | None
    revision: str | None
    filename: str | None
    expected_sha256: str | None
    observed_sha256: str | None
    authority: CheckpointAuthority | None
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

Legacy checkpoint specifications use schema `1.0.0` and remain readable. Authority-complete
specifications use schema `2.0.0`. They MUST bind a structured `CheckpointAuthority` to the exact
requested filename and MUST NOT treat an arbitrary direct HTTPS URL as authoritative evidence.

The dedicated checkpoint filename grammar MUST accept safe provider names such as
`Cnn14_16k_mAP=0.438.pth` and safe nested relative resource paths. It MUST reject absolute paths,
drive paths, traversal, empty or doubled path segments, backslashes, NULs, URL queries or fragments,
and paths that escape their package or cache root. The stricter repository-artifact path grammar is
not a substitute for this provider-filename grammar.

```python
@dataclass(frozen=True)
class PublishedChecksum:
    algorithm: str
    digest: str

@dataclass(frozen=True)
class CheckpointAuthority:
    provider: str
    record_id: str
    filename: str
    expected_size_bytes: int
    published_checksums: tuple[PublishedChecksum, ...]
    record_url: str | None
    provenance_status: str
```

Authority identifiers MUST be canonical, expected size MUST be positive and exact, checksum
algorithms MUST be unique, and digests MUST validate for their declared algorithms. MD5 and SHA-256
MUST be supported. Provider metadata MUST resolve before payload acquisition through an injectable,
bounded metadata transport. Zenodo resolution MUST use the official record API derived from the
structured record ID, select exactly one equal filename, reject malformed or duplicate metadata,
and accept only provider-controlled HTTPS metadata and payload URLs.

## 12.2 Cache

Resolved checkpoints MUST be stored under:

```text
.torch-dae/checkpoints/<checkpoint-id>/<sha256>/
```

Checkpoint files MUST NOT be committed.

## 12.3 Checkpoint manager

The checkpoint manager MUST:

1. resolve structured authority metadata before payload acquisition when authority is declared;
2. resolve the source;
3. acquire or locate the asset under an independent maximum-byte safety ceiling;
4. stream and compute observed byte count, SHA-256, and every authority-published digest algorithm;
5. require observed bytes to equal the authoritative exact expected size;
6. compare every published checksum and any expected SHA-256;
7. persist metadata-response and payload provenance;
8. install content-addressed cache state only after all required checks pass;
9. return a local immutable path.

Published checksum means evidence declared by the authoritative provider. Observed checksum means a
digest calculated from acquired or cached local bytes. Exact expected size is an authority identity
and integrity constraint. Maximum bytes is a resource-safety ceiling. These concepts MUST remain
distinct.

When an authority publishes only MD5, acquisition MAY succeed if policy permits and exact size plus
the published MD5 match. The manager MUST compute SHA-256 locally, use that observed SHA-256 as the
content-addressed cache identity, and MUST NOT relabel it as provider-published evidence.

Offline reuse MUST revalidate checkpoint ID, specification fingerprint, exact file size, observed
SHA-256/cache identity, every published checksum, and cached metadata provenance. File existence
alone MUST NOT establish a valid cache hit. An invalid cache or authority mismatch MUST fail
explicitly offline.

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
  "runtime_verification_target": "onboarding_reports/.../verify/runtime-targets/target.json",
  "runtime_verification_target_sha256": "...",
  "verification_report": "verification_reports/.../report.json",
  "verification_report_sha256": "...",
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
    "verified": true,
    "fingerprint": "...",
    "verification_result": "onboarding_reports/.../verify/environment-results/environment.json",
    "verification_result_sha256": "..."
  },
  "installation_commands": [],
  "checkpoint_loading": [],
  "smoke_test_command": "torch-dae model verify --target <runtime-target.json> --offline"
}
```

When `verified` is false, the fingerprint and verification-result fields MUST be absent. When it is
true, all three fields are required and MUST match a promoted canonical
`EnvironmentVerificationResult` under `onboarding_reports/<workflow-id>/verify/`. That result MUST
have `verification_status = passed`, `lifecycle_state = environment_verified`, and no failure
classification. It MUST also contain nonempty import and environment-smoke observations, every
observation MUST have passed, and observation names MUST be nonempty and unique across both
collections. A hash-correct failed or incomplete result is diagnostic evidence and MUST NOT satisfy
a verified model-card claim.

## 18.9 Legacy profiling placeholders

Model Card schema `1.0.0` retains `architectural_profiling`, `inference_profiling`, and
`energy_profiling` for compatibility with the original bootstrap contract.

New accepted Model Cards MUST leave those placeholders at `not_profiled`. Profiling completion MUST
NOT mutate them and MUST NOT change `card_status` away from `runtime_verified`. The independent
Technical Card contract defined in Section 27 is the normative representation for new profiling
evidence.

Model-card creation, runtime verification, release acceptance, and model contribution MUST NOT
depend on profiling completion.

---

# 19. Repository-analysis skill---

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

* stable workflow ID;
* requested variant;
* requested checkpoint;
* official paper URL;
* intended task;
* intended default embedding;
* target local platform.

The skill MUST remain independent of the previous project’s backbone JSON files.

When a workflow ID is supplied, the skill MUST discover and validate accepted prerequisite
handoffs locally before requesting attachments. Without a workflow ID, automatic selection is
permitted only when exactly one compatible active workflow exists. Recorded decisions and unresolved
items MUST be consumed and carried forward. A duplicate attachment MUST match the canonical digest;
a mismatch MUST stop the phase rather than select one silently. An accepted phase MUST be promoted
before it is declared complete, after which the canonical `finalize` operation MUST perform the
required repository gates, evidence-invariance checks, cleanup preflight, deterministic review
packaging, and any explicitly requested scoped cleanup.

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

Consumes an accepted integrate handoff, creates strict runtime-verification targets, resolves and
verifies their environments, then acquires and verifies each checkpoint. It does not require a
model card.

### `card`

Consumes accepted analysis, environment, integration, environment-verification, and
checkpoint-runtime evidence to create or update the checkpoint-specific final model card. A card is
an evidence consumer and publication surface, never a bootstrap prerequisite.

### `profile`

Reserved compatibility entry point inside the onboarding skill. The onboarding skill MUST NOT
execute profiling or mutate Model Cards. The independent profiling workflow defined in Section 27
will consume accepted `runtime_verified` Model Cards and emit Technical Cards after its implementation.

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

Every package imported directly by the selected minimal runtime source surface MUST be declared as a
direct environment dependency. Direct imports MUST NOT rely only on transitive installation. When
exact pins are used, the environment verification script MUST verify every declared direct
dependency and its exact version.

The local `torch-deepaudioembedding` wheel is installed only after locked-project synchronization
and with dependency resolution disabled (`--no-deps`). Its active package-runtime requirements
therefore MUST already be reachable and version-compatible in the accepted environment lock. A
reachable transitive entry MAY satisfy a wheel requirement because locked synchronization installs
that closure; an orphan lock entry MUST NOT satisfy it. Direct declaration remains required when no
reachable dependency path otherwise installs the requirement. Dependency-closure preflight MUST run
before environment creation and wheel installation and MUST emit a deterministic strict result.

When variants or checkpoints may share a future source substrate, resolution MUST identify the
minimum required files and symbols and compare candidate revisions at byte, symbol, and semantic
levels. One common revision MAY be selected only when it preserves every required variant;
otherwise tuple-specific provenance is required. Checkpoint equivalence MUST NOT be inferred from
chronological proximity.

Multiple tuples MAY reuse one environment stack only when their direct dependencies are equivalent,
the selected source APIs are compatible, constructor/import trials pass for each tuple, platform and
interpreter are identical, and all differences and reused evidence are documented.

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

## 20.6 Constructor trials and completion states

A constructor trial MAY establish dependency resolution, source import, class construction, and
parameter-device placement only. It MUST explicitly record that it does not establish checkpoint
compatibility, forward or output correctness, embedding correctness, inference equivalence, or
runtime verification.

A successful resolve-environment phase has two distinct completion states:

1. **draft resolution complete**: an evidence-supported candidate is selected; isolated import and
   constructor trials pass; environment drafts and locks validate; production source, wrapper, or
   card prerequisites are intentionally absent; no fingerprint or lifecycle promotion is claimed;
   and `integrate` MAY be recommended next;
2. **environment lifecycle resolved**: canonical materialization and verification pass; the
   fingerprint and report reference exist; every lifecycle contract is satisfied; and promotion to
   `environment_resolved` is valid.

The first state is successful phase completion but MUST NOT be promoted to `environment_resolved`.
The strict `EnvironmentResolutionReport` lifecycle contract remains authoritative.

Optional host metadata probes MUST be portable and non-blocking. Failure of an optional CPU-brand or
platform-detail probe MUST be recorded separately and MUST NOT invalidate an otherwise successful
constructor trial.

Sandbox, DNS, package-index, authentication, and rate-limit failures MUST retain their original
logs, be classified separately from model or dependency incompatibility, and MAY receive one
identical evidence-motivated rerun when execution policy permits. Both outcomes MUST be preserved;
generic code MUST NOT be changed merely to hide an external execution failure.

---

# 21. Runtime-verification protocol

A card reaches `runtime_verified` only after the wrapper successfully verifies the applicable capabilities.

## 21.0 Authority and lifecycle separation

The generic authority chain is:

```text
EnvironmentSpecification
  -> EnvironmentMaterializationResult
  -> EnvironmentVerificationResult(status=passed) and matching environment fingerprint
  -> RuntimeVerificationTarget
  -> checkpoint-specific VerificationReport(status=passed)
  -> final ModelCard
```

Failed evidence remains diagnostic evidence and never promotes lifecycle state. The existence of a
result or report, a valid hash, or a self-declared passed status is not proof of success.

A successful result proves both:

* successful global status;
* complete successful coverage of the evidence required by its authority contract.

An environment is `draft` when accepted definition and lock artifacts exist, `materialized` after
locked dependencies and sources are prepared, and `environment_verified` only after infrastructure
verification passes. These states belong to the environment. `runtime_verified` belongs to one
model/variant/checkpoint/environment tuple and requires checkpoint acquisition, hashing, loading,
forward checks, and a checkpoint-specific report. Environment-only evidence MUST NOT promote a
checkpoint or card.

`RuntimeVerificationTarget` is a strict executable request. It associates workflow, accepted
integration handoff, integrated variant/adapter, checkpoint specification and acquisition policy,
future card identity, public entry point, environment definition and source manifest, waveform and
sample-rate contract, expected outputs/probability/embedding semantics, permitted devices,
execution limits, unresolved items, and ordered `required_check_ids` and `optional_check_ids`. New
verification uses target schema `2.0.0`. Required check IDs MUST be nonempty, canonical, and unique;
optional check IDs MUST be canonical and unique; the two sets MUST be disjoint. Their input order is
part of the deterministic request. The target MUST make no success claim and MUST NOT require an
existing final card.

The required-check vocabulary MUST be capable of representing checkpoint acquisition and SHA-256,
checkpoint deserialization, state-dictionary compatibility, canonical waveform input,
checkpoint-loaded forward execution, every expected output, declared probability behavior, the
default embedding, deterministic repeated execution, and required CPU execution. Optional checks
represent bounded diagnostics rather than prerequisites.

Verify mode MUST perform these operations in order:

1. consume the accepted integrate handoff;
2. create or resolve strict runtime-verification targets;
3. resolve each target environment by environment ID;
4. materialize from the accepted environment definition;
5. run environment verification and record the fingerprint;
6. acquire the checkpoint according to its explicit policy;
7. verify checkpoint provenance and SHA-256;
8. deserialize and check model/state-dictionary compatibility;
9. perform bounded forward, output, probability, embedding, and device checks;
10. create the checkpoint-specific verification report;
11. promote the accepted verify handoff.

Environment verification results are infrastructure evidence. Managed execution copies live under
`.torch-dae/reports/environments/<environment-id>/<fingerprint>/`; accepted verify orchestration
MUST promote the normalized result under its phase-local `onboarding_reports/<workflow-id>/verify/`
tree. The promoted result MUST retain `verification_status = passed`,
`lifecycle_state = environment_verified`, and the exact target environment fingerprint. A failed
environment result uses `verification_status = failed`, `lifecycle_state = materialized`, and a
non-null failure classification. A passed result MUST contain nonempty import and smoke collections,
only passed observations, unique nonempty observation names across both collections, and no failure
classification. Failed results MAY retain partial, failed, or absent observations when failure
preceded observation completion. Generic environment results MUST NOT be stored under
`verification_reports/`. Managed command logs and materialization records remain runtime evidence,
but a final card MUST reference the promoted canonical environment result rather than an unmanaged
file.

Target-aware verification reports use schema `2.0.0`, repeat their target's exact ordered required
and optional check IDs, and MUST declare `verification_status = passed | failed`. Checks MUST be
nonempty, canonical, unique, and target-declared. Every required check MUST appear exactly once and
MUST pass for an overall passed report; a required check MUST NOT be unsupported. A passed report
contains no failed check. A failed report contains at least one failed declared required or optional
check and cannot support `runtime_verified`. An optional check may pass or be explicitly unsupported
without false failure. Every unsupported optional check MUST include nonempty details, appear in
`unsupported_capabilities`, and have a recorded limitation. Repository validation MUST compare the
report with its referenced target and enforce the exact check contracts and all existing identity,
hash, fingerprint, checkpoint, environment, source, handoff, and entry-point associations.

Legacy schema `1.0.0` reports remain readable without rewriting: any failed check makes them
unsuccessful, while a report with no failed checks retains the existing conservative compatibility
interpretation. A legacy report or schema `1.0.0` target MUST NOT be silently reinterpreted as
completeness-aware evidence for a new `runtime_verified` card. Existing target fixtures migrate by
setting schema `2.0.0` and adding explicit ordered required and optional check ID fields.

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

* runtime-verification target and workflow identity;
* integrated variant, checkpoint, public entry point, and accepted integration-handoff hash;
* environment-specification and source-manifest hashes;
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

Committed verification reports are created only by `verify` after checkpoint-specific runtime
observation. Pre-runtime handoffs and audit/transport bundles are not verification reports.

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
torch-dae env resolve <environment-id>
torch-dae env materialize <environment-id>
torch-dae env verify <environment-id>
torch-dae env remove <card-id>
torch-dae env info <card-id>
torch-dae env run <card-id> -- <command>

torch-dae checkpoint ensure <card-id>
torch-dae checkpoint info <card-id>
torch-dae checkpoint remove <card-id>
torch-dae checkpoint resolve --spec <checkpoint-spec.json>
torch-dae checkpoint ensure-spec --spec <checkpoint-spec.json>
torch-dae checkpoint info-spec --spec <checkpoint-spec.json>

torch-dae model inspect <card-id>
torch-dae model verify --target <runtime-target.json> [--offline] [--json]
```

The control-plane CLI MUST remain usable without installing PyTorch in the root environment.

Card-independent checkpoint commands MUST strictly load a `CheckpointSpec`, resolve authoritative
metadata without acquiring payload bytes, acquire only through `CheckpointManager.ensure_checkpoint`,
inspect authority/cache state without a model card, emit machine-readable JSON, and honor offline
policy. Card-based `checkpoint ensure` MUST delegate to the same explicit manager primitive.

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
* environment materialization results;
* environment verification results;
* runtime-verification targets;
* checkpoints;
* embeddings;
* verification reports;
* onboarding workflow records;
* onboarding phase-handoff manifests.

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

# 26. MVP pilot model family

The MVP pilot is completed using the PANNs family only. The pilot covers three fixed
checkpoint-specific integrations so that the full onboarding lifecycle can be exercised without
expanding scope to additional model families.

## 26.1 PANNs checkpoints

The pilot comprises:

* `panns-cnn14-16k-map-0438`;
* `panns-resnet38-map-0434`;
* `panns-wavegram-logmel-cnn14-map-0439`.

Together they exercise classification, multilabel probabilities, checkpoint authority and strict
loading, clipwise embeddings, environment isolation, runtime verification, and checkpoint-specific
model-card generation. BYOL-A, EnCodec, and other model families remain optional future expansion
targets and are not required to complete the MVP pilot.

The three PANNs checkpoint integrations complete the MVP pilot. Lessons from this pilot MAY update the core API and skill before optional future model-family expansion.

---

# 27. Profiling and Technical Card specification

## 27.1 Evidence boundary

Profiling is optional empirical evidence collected only after a checkpoint-specific Model Card has
reached `runtime_verified`. It is not required for onboarding acceptance, release inclusion, or
model support.

One accepted immutable Model Card MAY have zero or more independent Technical Cards:

```text
accepted Model Card
  +-- Technical Card: CPU
  +-- Technical Card: MPS
  +-- Technical Card: CUDA
  +-- ...
```

A Model Card states what the model/checkpoint is and what torch-dae verified. A Technical Card states
how that exact public wrapper behaved under one profiling protocol, device/backend, and execution
context. Profiling MUST NOT mutate or enrich the Model Card.

The future website/analytics layer is derived presentation only. It MUST reconstruct claims from
Model Cards, Technical Cards, and hash-addressed raw assets.

## 27.2 Terminology

**Profiling Campaign**: one user invocation of the profiling CLI.

**Profiling Run**: one device/backend-specific attempt inside a campaign.

**Technical Card**: immutable evidence produced by exactly one successful Profiling Run.

**Profiling Protocol**: versioned normative measurement methodology.

**Profiling Plan**: locally resolved execution plan derived from Model Card, protocol, requested
devices, and detected hardware.

**Raw Measurement Asset**: losslessly compressed `.npz` containing bounded supporting observations.

**Execution Context**: privacy-safe hardware/software/runtime configuration for one Profiling Run.

**Profile Finalization**: validation and review packaging of a profiling campaign; it never advances
Model Card onboarding.

## 27.3 Immutability and granularity

Accepted Model Card bytes MUST remain unchanged by profiling.

A contributed Technical Card and its raw `.npz` asset are immutable. Corrections require a new card
that MAY declare `supersedes` references.

One Technical Card represents exactly:

```text
one exact Model Card/checkpoint
x one profiling protocol version
x one device/backend
x one execution environment/context
x one complete profiling session
```

A campaign using automatic device discovery MAY therefore generate separate CPU, MPS, and CUDA
Technical Cards.

## 27.4 Devices and CLI contract

Profiling v1 SHALL support:

```text
auto
cpu
mps
cuda
cuda:<index>
```

`--device` SHOULD be repeatable.

Planned public CLI:

```text
torch-dae model profile \
  --model <model-id> \
  --device <auto|cpu|mps|cuda|cuda:index> \
  --protocol <protocol-id> \
  --energy <auto|off> \
  --output-dir <path>
```

Recommended defaults:

```text
--device auto
--protocol audio-inference-v1
--energy auto
--output-dir ./technical_cards
```

`auto` includes CPU, discovers locally available supported accelerators, performs a bounded
public-wrapper smoke inference on each, and profiles each successful device/backend. Failed
accelerator attempts are campaign diagnostics and do not generate Technical Cards.

Explicit device requests MUST NOT be silently replaced with CPU unless CPU was also requested.

Planned utilities:

```text
torch-dae technical-card validate <card>
torch-dae technical-card inspect <card>
torch-dae technical-card list
```

These commands remain specification-only until implementation lands.

## 27.5 Precision

Protocol v1 uses only the accepted wrapper's verified/native precision. It does not automatically
enable FP16, BF16, AMP, quantization, or `torch.compile`.

Technical Cards record observed parameter dtype, execution dtype, autocast state, and quantization
state.

## 27.6 Canonical profiling input

Profiling v1 is dataset-independent and uses only deterministic seeded synthetic white noise.

The canonical waveform generator produces `float32` IID uniform samples:

```text
x[n] ~ U(-1, 1)
```

The protocol records PRNG implementation/algorithm, seed, sample rate, sample count, batch size,
channel count, and generated tensor/waveform SHA-256 where practical.

No speech, music, soundscape, benchmark dataset, or external audio file participates in protocol v1.
Input generation/allocation occur outside canonical timed inference.

## 27.7 Canonical duration and minimum supported input

Every Profiling Run resolves a canonical benchmark duration from the accepted model/runtime
contract. When the wrapper is variable-length and no fixed duration exists, protocol v1 uses
`10.0` seconds as the reference duration.

For `batch=1`, the profiler SHALL empirically determine the minimum supported waveform length in
integer sample-count space:

1. initial probe at 10 seconds;
2. on success, repeatedly halve until a failing/lower boundary is found;
3. on failure, repeatedly double until success or 120 seconds;
4. binary-search the failing/passing bracket;
5. recheck the candidate minimum and immediately smaller sample count where valid.

Non-monotonic behavior MUST be reported as `non_monotonic`; no minimum may be fabricated.

The observed minimum is device/backend profiling evidence and never rewrites the Model Card input
contract.

## 27.8 Benchmark matrix and repetitions

At canonical duration:

```text
batch_size = [1, 2, 4, 8]
```

The empirical minimum duration is additionally profiled at `batch=1` when distinct from canonical
duration.

Every steady-state condition uses:

```text
warmup_inferences = 10
measured_inferences = 50
```

There is no outer repetition loop in protocol v1.

Tested resource/input failures are explicit `unsupported` observations rather than silently omitted
conditions.

## 27.9 Timing scope and statistics

Canonical latency measures:

```text
prepared waveform tensor
-> public torch-dae wrapper
-> requested public output
```

Internal wrapper DSP is included. Waveform generation, filesystem I/O, serialization, campaign
orchestration, and report generation are excluded.

Timing uses a monotonic high-resolution clock plus appropriate device synchronization around the
public-wrapper call.

Required summaries:

* mean;
* standard deviation;
* minimum;
* maximum;
* p50/median;
* p90;
* p95;
* p99;
* coefficient of variation;
* throughput;
* real-time factor;
* speed factor.

The 50 raw latency observations MUST be stored losslessly, preferably as integer nanoseconds.

## 27.10 Cold-start evidence

Cold evidence is separate from steady-state inference. Where separable, record wrapper/model
initialization, checkpoint/model loading, device placement/readiness, and first-inference latency.
The first inference is never part of the 50 steady-state observations.

## 27.11 Instrumentation separation

Canonical latency uses minimal instrumentation.

Memory and energy measurement occur in a separate bounded resource pass so process sampling,
allocator inspection, CodeCarbon, or other instrumentation cannot contaminate canonical latency.

## 27.12 CPU thread regimes

One CPU Technical Card contains two thread regimes:

```text
single_thread
native_default
```

`single_thread` sets torch intra-op threads to 1. `native_default` observes the original/default
runtime configuration. Thread-sensitive regimes SHOULD execute in isolated subprocesses.

Record torch intra-op/inter-op observations and relevant BLAS/OpenMP settings where available.

## 27.13 Architecture evidence

Mandatory:

* total parameters;
* trainable parameters;
* non-trainable parameters;
* parameter bytes;
* buffer bytes;
* state-dict tensor bytes;
* dtype distribution;
* module count.

Optional capability-dependent evidence:

* FLOPs;
* MACs;
* operator/module census.

FLOP/MAC results MUST record backend, backend version, counting convention, unsupported-operation
coverage, and `complete`/`partial`/`unavailable` status. Partial counts MUST NOT be presented as
complete totals.

## 27.14 Host RAM and accelerator memory

Host RAM is a first-class profiling domain.

Where available, record:

* total host RAM capacity;
* process RSS before model load;
* process RSS after model load;
* process RSS immediately before resource measurement;
* sampled peak process RSS;
* process RSS after the resource pass.

USS/PSS MAY be recorded with explicit semantics.

CUDA cards SHOULD record native allocator current/peak allocated and reserved memory where
available.

MPS cards SHOULD record the exact exposed tensor-allocation and Metal-driver-allocation surfaces.
On unified-memory systems, host RSS and accelerator/driver allocation MUST NOT be added into a
fabricated independent-memory total.

Cross-backend memory surfaces retain their backend-specific semantics.

## 27.15 Energy and CodeCarbon

Energy is optional capability-dependent evidence and is not required for Technical Card validity.

Protocol v1 integrates CodeCarbon as the first energy backend when available.

Public selection:

```text
--energy auto|off
```

Energy evidence MUST distinguish:

```text
hardware_measured
software_estimated
unavailable
failed
```

Record backend/version, measurement kind, scope, duration, sampling interval, privilege usage,
CPU energy, accelerator/GPU energy, RAM energy, total energy, and average power where produced.

When privileged hardware counters can improve measurement quality, an interactive CLI MAY request
explicit user consent. It MUST NOT edit sudoers, store credentials, or silently elevate. Declined or
non-interactive elevation falls back to the best non-privileged measurement/estimation path or
reports `unavailable`.

Carbon/geolocation functionality is not required for Technical Cards and MUST NOT silently
geolocate contributors.

## 27.16 Profiling tooling versus accepted model environments

Profiling MUST NOT silently mutate accepted onboarding environment definitions or historical
environment evidence.

Additional instrumentation dependencies such as CodeCarbon MUST preserve accepted model-runtime
dependency versions and be recorded separately as profiling-tooling provenance. Model environment
fingerprint, torch-dae package identity, profiler implementation version, and profiling-tooling
identity remain distinct provenance fields.

## 27.17 Identity, privacy, and contributor fields

Technical Card IDs are globally unique and hash-derived. The canonical digest binds at least:

* Technical Card schema version;
* Model Card ID and SHA-256;
* checkpoint SHA-256;
* profiling protocol ID/version;
* torch-dae package identity;
* source revision/dirty state when available;
* hardware-configuration fingerprint;
* execution-context fingerprint;
* device/backend;
* random run nonce.

Measurement values are not part of ID construction. Human IDs SHOULD use
`tc-<truncated-identity-sha256>` while retaining the full SHA-256.

Hardware-configuration fingerprints MAY include CPU model/architecture/core counts, RAM capacity,
accelerator vendor/model, and accelerator memory where meaningful. They MUST NOT include hostname,
username, IP address, serial number, MAC address, machine UUID, credentials, tokens, or home paths.

Execution-context fingerprints extend hardware class with OS/version, Python, PyTorch, torch-dae
package identity, device/backend, native precision, and CPU thread regime. They represent equivalent
technical contexts, not unique physical machines.

Contributor display name and GitHub handle are optional explicit fields and never affect technical
validity, identity, or comparability.

## 27.18 Runtime classification and versioning

Technical Cards MAY be:

```text
canonical
modified_runtime
unknown_runtime
```

Modified/local torch-dae runs are accepted when internally valid and fully identified, but default
reference analytics SHOULD restrict to `canonical` unless explicitly broadened.

Profiling distinguishes:

```text
technical_card_schema_version
profiling_protocol_version
profiler_implementation_version
```

These versions MUST NOT be conflated.

## 27.19 Raw evidence storage

Committed profiling evidence uses:

```text
technical_cards/
  <model-id>/
    <technical-card-id>.json
    <technical-card-id>.npz
```

The JSON stores identity, provenance, summaries, limitations, and the raw-array manifest. The `.npz`
stores bounded, lossless observations and MUST be loadable with pickle disabled.

Raw evidence MAY contain the 50 latency observations per successful condition, bounded host-memory
samples, bounded accelerator-memory samples, and bounded energy/power samples when exposed.

Large profiler traces, Chrome traces, checkpoint payloads, system dumps, and unbounded logs MUST NOT
be committed as Technical Card evidence. Protocol v1 uses normal Git for compact assets.

## 27.20 Technical Card top-level contract

Technical Card schema v1 is conceptually:

```text
schema_version
technical_card_id
identity_sha256
created_at_utc

model_reference
protocol
profiler
run
contributor

hardware
software
device
execution_context

precision
input_provenance

architecture
cold_start
conditions
memory
energy

raw_measurements
measurement_coverage
comparability
limitations
supersedes
```

## 27.21 Technical Card validity

A Technical Card is valid only when:

* the referenced Model Card exists and validates;
* Model Card SHA-256 matches;
* checkpoint identity/hash matches the Model Card;
* the profiling protocol is recognized;
* the public wrapper succeeds on the declared device/backend baseline;
* raw timing evidence validates against summaries;
* required hardware/software provenance is present;
* Technical Card identity recomputes;
* raw `.npz` SHA-256 and array manifest validate;
* privacy validation finds no prohibited identifiers.

Energy, FLOP/MAC evidence, accelerator-specific memory details, or optional scaling conditions MAY be
unavailable. Measurement coverage distinguishes missing evidence from tested unsupported
conditions.

## 27.22 Campaign result and profile finalization

A Campaign Result is orchestration metadata, not a contributed Technical Card. It records requested,
detected, attempted, failed, and successful devices plus generated Technical Cards, validation
outcomes, and finalization/review artifacts.

Profiling is repeatable and MUST NOT reuse or mutate the model-onboarding workflow ID as a new
lifecycle phase.

The profiler SHALL create an independent managed campaign/run identity and reuse the repository's
hardened run-manifest, validation, deterministic review-bundle, evidence-invariance, and cleanup
principles where applicable.

Profile finalization proves the referenced Model Card remained unchanged while validating Technical
Cards and raw assets. It never promotes Model Card lifecycle.

## 27.23 GitHub contribution and future analytics

Profiling v1 contribution is a normal pull request containing only:

```text
technical_cards/<model-id>/<technical-card-id>.json
technical_cards/<model-id>/<technical-card-id>.npz
```

CI SHALL verify schema, identity digest, raw hash/array manifest, Model Card association, checkpoint,
protocol, raw/summary consistency, provenance, privacy, duplicate IDs, and supersession references.

Contributors MUST NOT modify the referenced Model Card when adding Technical Cards.

Future strict analytics groups MUST distinguish at least model/checkpoint, protocol compatibility,
benchmark condition, duration/sample count, batch, precision, device/backend, CPU thread regime, and
runtime classification.

Default summaries SHOULD emphasize median, IQR, min/max, Technical Card count, and unique
execution-context count. Repeated cards from one execution context SHOULD first be aggregated within
that context so one contributor/context cannot dominate cross-platform summaries.

# 28. Implementation sequence

Development SHOULD proceed through the following bounded phases.

## Phase 00 — Repository bootstrap and normative contracts

Completed repository scaffold, strict schemas, typed contracts, CLI skeleton, environment/checkpoint
interfaces, tests, CI, and documentation.

## Phase 01 — Runtime infrastructure

Completed environments, source hierarchy, checkpoint manager, and integration-test foundations.

## Phase 02 — Skill MVP

Completed canonical onboarding skill, evidence workflows, templates, and agent-neutral handoff
contracts.

## Phase 03 — PANNs pilot

Completed three checkpoint-specific PANNs integrations through immutable `runtime_verified` Model
Cards.

## Phase 04 — Profiling and Technical Card normative specification

Freeze the v1 evidence model, protocol, storage, privacy/provenance, CLI, contribution, and analytics
contracts before implementation.

## Phase 05 — Profiling subsystem implementation

Implement generic Technical Card schemas, deterministic synthetic-input profiling,
architecture/timing/RAM/device-memory profilers, CodeCarbon integration, compact raw `.npz` evidence,
campaign orchestration, validation, CLI, and profile finalization.

## Phase 06 — PANNs profiling campaign

Exercise the generic profiler on the three accepted PANNs Model Cards across locally available CPU
and accelerator backends. Validate and contribute the first canonical Technical Cards without
modifying the Model Cards.

## Phase 07 — Model + Technical Card presentation and analytics

Build the site locally from validated Model Cards and Technical Cards, including strict
protocol-aware filtering, robust statistics, and cross-platform comparison. Deploy GitHub Pages only
after local validation.

## Phase 08 — Core stabilization and optional model-family expansion

Stabilize public profiling contracts after the PANNs profiling campaign, then onboard additional
model families independently.

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
19. profiling v1 is normatively specified as an optional Technical Card capability even while implementation remains pending;
20. no pilot model implementation has started.

---

# 30. Project invariants

The following invariants apply throughout development:

1. one Model Card represents one model-variant-checkpoint tuple;
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
15. profiling begins only from an accepted `runtime_verified` Model Card;
16. profiling is optional for model onboarding, acceptance, and release inclusion;
17. accepted Model Cards are immutable profiling references and are never enriched in place;
18. one Technical Card represents one model/checkpoint, protocol, device/backend, execution context,
    and profiling session;
19. Technical Cards are immutable append-only empirical evidence;
20. profiling v1 uses deterministic seeded synthetic white noise only;
21. canonical steady-state conditions use 10 warmups and 50 measured inferences;
22. raw profiling observations are stored as bounded lossless `.npz` assets plus strict JSON
    Technical Cards;
23. host RAM and accelerator-memory evidence retain backend-specific semantics;
24. energy evidence distinguishes hardware measurement from software estimation and MAY be
    unavailable without invalidating the Technical Card;
25. unresolved information is represented explicitly rather than rhetorically strengthened;
26. accepted pre-runtime phase handoffs are committed under `onboarding_reports/`;
27. `verification_reports/` remains checkpoint-specific runtime evidence only;
28. temporary onboarding work uses recorded managed workspaces and scoped cleanup;
29. draft environment resolution is distinct from lifecycle promotion;
30. accepted environment definitions, not model cards, authorize materialization;
31. environment verification and checkpoint-runtime verification remain separate evidence;
32. runtime-verification targets preserve explicit model/checkpoint/environment identity;
33. final Model Cards may claim verification only by referencing matching hash-addressed evidence;
34. only passed environment and runtime evidence can promote Model Card lifecycle state;
35. failed environment or runtime evidence remains diagnostic and never authorizes promotion;
36. profiling campaigns use independent managed identities and never advance onboarding lifecycle;
37. Technical Card contributions never modify their referenced Model Card;
38. contributor identity is optional and prohibited personal/machine identifiers are never inferred;
39. analytics aggregate only explicitly compatible profiling conditions by default.

# 31. Specification freeze

The following interfaces are frozen for the profiling implementation cycle:

* checkpoint-specific Model Cards as immutable onboarding evidence;
* `runtime_verified` as terminal for new Model Card onboarding;
* schema-1 `profiled`/embedded profiling fields as legacy compatibility until a future major schema
  migration;
* `[B,C,T]` waveform input and explicit sample rate;
* optional `valid_lengths`;
* automatic resampling with strict opt-out;
* `from_random()`, `from_pretrained()`, and `load_checkpoint()`;
* `forward()`, `predict_probability()`, `available_embeddings()`, and `compute_embedding()`;
* model-specific environments and checkpoint management;
* Technical Cards as independent immutable profiling evidence;
* one Technical Card per device/backend profiling session;
* deterministic seeded white-noise protocol v1 input;
* canonical batch matrix `[1,2,4,8]`;
* 10 warmups and 50 measured steady-state inferences per condition;
* host-RAM profiling and backend-specific accelerator-memory semantics;
* CodeCarbon as the initial optional energy backend with measured/estimated provenance;
* compact JSON + lossless `.npz` Technical Card storage;
* strict compatible-group aggregation for future analytics.

Implementation details not fixed by this specification MAY evolve without changing these evidence
semantics.
