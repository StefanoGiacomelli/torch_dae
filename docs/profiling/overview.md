# Profiling overview

Profiling v1 measures how an already accepted `torch-dae` model behaves in a specific execution
context. It is optional empirical evidence: a model does **not** need a Technical Card to be
supported, and profiling never changes the accepted Model Card.

```text
runtime_verified Model Card
          |
          +---- Technical Card: CPU / single_thread
          +---- Technical Card: CPU / native_default
          +---- Technical Card: MPS
          +---- Technical Card: CUDA
          `---- ... later independent sessions
```

A Model Card answers **what model/checkpoint is supported and what behavior was verified**. A
Technical Card answers **how that exact accepted wrapper behaved under one profiling protocol,
device/backend, execution context, and session**.

## Quick command

From the full repository workspace with the profiling dependencies installed:

```bash
uv sync --python 3.11 --all-groups --extra profiling --frozen

uv run torch-dae model profile \
  --model panns-cnn14-16k-map-0438 \
  --device cpu \
  --protocol audio-inference-v1 \
  --energy off \
  --output-dir profiling_candidates/cnn14-cpu \
  --json
```

The target Model Card must be `runtime_verified`. `--output-dir` must be a repository-local
candidate/workspace directory and must **not** be the repository root or any path inside canonical
`technical_cards/`.

See {doc}`../tutorials/profiling` before running a real campaign.

## What Profiling v1 measures

The current protocol records, where available:

- architecture size: parameters, buffers, state-dict bytes, module count, dtype distribution;
- cold-start components such as initialization, checkpoint loading, device placement, and first
  inference;
- empirical minimum supported waveform length;
- steady-state latency and derived throughput/real-time metrics;
- host process memory;
- backend-specific accelerator memory;
- CodeCarbon-backed energy evidence when enabled and available;
- hardware/software/execution-context metadata needed to interpret the session.

FLOPs/MACs are optional capability-dependent evidence and may legitimately be unavailable.

## What you control

The public CLI lets you select:

| Control | Meaning |
|---|---|
| `--model` | accepted Model Card ID to profile |
| `--device` | `auto`, `cpu`, `mps`, `cuda`, or `cuda:<index>`; repeatable |
| `--protocol` | currently `audio-inference-v1` |
| `--energy` | `auto` or `off` |
| `--allow-privileged-energy` | explicitly permits local privileged hardware-counter requests |
| `--output-dir` | candidate evidence directory inside the repository |
| `--json` | machine-readable campaign summary on stdout |

## What the protocol fixes

For comparability, Profiling v1 does **not** expose every measurement parameter as a tuning knob.
It fixes:

- deterministic seeded IID uniform white-noise input in `[-1, 1]`;
- native/verified execution precision;
- 10-second canonical duration for variable-length models without an explicit fixed duration;
- canonical batch sizes `1, 2, 4, 8`;
- empirical minimum-input measurement at batch 1 when distinct;
- exactly 10 warmups and 50 measured steady-state inferences;
- CPU `single_thread` and `native_default` regimes;
- canonical summary statistics and raw latency preservation.

Changing these would define a different protocol rather than a personalized v1 run.

## Device discovery

`--device auto` always includes CPU, adds MPS only when the model environment reports it usable, and
adds every discovered CUDA device as `cuda:<index>`. Each concrete device must pass a bounded public
wrapper smoke inference before a Technical Card can be generated. A failed accelerator is reported
as campaign diagnostics; it is not silently replaced by CPU.

## Candidate evidence versus canonical evidence

A profiling command writes candidate Technical Card JSON/NPZ pairs to the chosen `--output-dir`.
It also writes local campaign orchestration metadata to:

```text
.torch-dae/profiling/<campaign-id>/campaign-result.json
```

The `.torch-dae/` campaign workspace is runtime state and is not the contributed Technical Card.
Canonical accepted evidence lives only under:

```text
technical_cards/<model-id>/<technical-card-id>.json
technical_cards/<model-id>/<technical-card-id>.npz
```

Promotion is a separate human-reviewed operation. The profiler never writes directly into the
canonical tree.

## Energy is optional evidence

`--energy auto` uses the configured CodeCarbon backend. The resulting measurement is classified as
`hardware_measured`, `software_estimated`, `unavailable`, or `failed`.

A numeric total does not automatically imply complete hardware coverage. Always inspect
`coverage_complete` and `unaccounted_components`. For example, a run may contain valid CPU/RAM
energy while an accelerator component remains unmeasured.

Privileged counters are never authorized implicitly. On a platform where better measurement needs a
local privileged path, supply `--allow-privileged-energy` only after deliberately deciding to permit
that request. The profiler does not edit sudoers, persist credentials, or require geolocation.

## Where to go next

- {doc}`../tutorials/profiling` — run a campaign end to end;
- {doc}`protocol` — exact measurement methodology;
- {doc}`technical-cards` — storage, validation, and evidence model;
- {doc}`../user-guide/technical-cards` — interpret fields and metrics;
- {doc}`contributing` — review and contribution workflow.
