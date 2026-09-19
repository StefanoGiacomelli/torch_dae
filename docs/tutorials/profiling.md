# Profile an accepted model

This tutorial runs Profiling v1 from an accepted Model Card to validated candidate Technical Card
evidence. It does not promote candidate files into the canonical `technical_cards/` tree.

## 1. Prepare the repository workspace

Profiling requires the full repository plus the optional profiling dependencies:

```bash
uv sync --python 3.11 --all-groups --extra profiling --frozen
```

Confirm the model exists and is accepted:

```bash
uv run torch-dae card show panns-cnn14-16k-map-0438
```

The target must be `runtime_verified`.

You can inspect existing canonical Technical Cards independently:

```bash
uv run torch-dae technical-card list
```

## 2. Decide campaign scope

For a first run, CPU with energy disabled is the simplest deterministic choice:

```text
model       panns-cnn14-16k-map-0438
device      cpu
protocol    audio-inference-v1
energy      off
output      profiling_candidates/cnn14-cpu
```

The CPU campaign produces two Technical Cards because protocol v1 separates:

```text
single_thread
native_default
```

If you use `--device auto`, the profiler additionally discovers usable MPS and every CUDA device.

## 3. Run the profiler

```bash
uv run torch-dae model profile \
  --model panns-cnn14-16k-map-0438 \
  --device cpu \
  --protocol audio-inference-v1 \
  --energy off \
  --output-dir profiling_candidates/cnn14-cpu \
  --json
```

A real campaign performs environment resolution/verification, checkpoint resolution, bounded device
smoke tests, minimum-input search, architecture/cold-start measurement, steady-state timing, and a
separate resource pass.

Profiling may take materially longer than a normal inference because every successful steady-state
condition uses 10 warmups plus 50 measured calls.

## 4. Find the outputs

Candidate evidence appears under your selected directory:

```text
profiling_candidates/cnn14-cpu/
  tc-<id-a>.json
  tc-<id-a>.npz
  tc-<id-b>.json
  tc-<id-b>.npz
```

Campaign orchestration metadata is written separately under ignored runtime state:

```text
.torch-dae/profiling/<campaign-id>/campaign-result.json
```

The campaign result tells you which devices were requested/detected/attempted, which concrete runs
succeeded, which card/raw paths were produced, and why any smoke-tested device failed.

## 5. Validate every candidate card

For each JSON emitted by the campaign:

```bash
uv run torch-dae technical-card validate \
  profiling_candidates/cnn14-cpu/<technical-card-id>.json
```

Or request JSON output:

```bash
uv run torch-dae technical-card validate \
  profiling_candidates/cnn14-cpu/<technical-card-id>.json \
  --json
```

Do not continue to promotion if validation fails.

## 6. Read the Technical Card

Pretty-print its normalized content:

```bash
uv run torch-dae technical-card inspect \
  profiling_candidates/cnn14-cpu/<technical-card-id>.json
```

For a first review, focus on:

1. `model` — exact Model Card/checkpoint being measured;
2. `device` and `execution_context` — where the run occurred;
3. `canonical_duration_seconds` and `minimum_input_search`;
4. `conditions[*].timing` — latency/throughput/RTF/speed factor;
5. `host_memory` and `accelerator_memory`;
6. `energy` — especially measurement kind and coverage;
7. `coverage` and `limitations`;
8. `raw_measurements` — path/hash/array manifest for the NPZ.

Use {doc}`../user-guide/technical-cards` for field interpretation.

## 7. Inspect raw latency samples

```python
from pathlib import Path
import numpy as np

npz_path = Path("profiling_candidates/cnn14-cpu/<technical-card-id>.npz")

with np.load(npz_path, allow_pickle=False) as raw:
    print(raw.files)
    samples_ns = raw["raw_ns__canonical-b1"]
    print(samples_ns.shape)   # (50,)
    print(samples_ns.dtype)   # int64 for latency arrays
```

Only arrays actually produced by the run are present. The Technical Card's raw manifest is the
source of truth for array names and shapes.

## 8. Enable energy intentionally

To request normal CodeCarbon evidence:

```bash
uv run torch-dae model profile \
  --model panns-cnn14-16k-map-0438 \
  --device cpu \
  --energy auto \
  --output-dir profiling_candidates/cnn14-energy
```

On platforms where the backend can use privileged hardware counters, `torch-dae` permits that path
only when you explicitly add:

```text
--allow-privileged-energy
```

Do not equate `total_energy_kwh` with complete system/run energy unless the card also states:

```text
coverage_complete = true
```

When it is false, inspect `unaccounted_components`.

## 9. Use automatic device discovery

```bash
uv run torch-dae model profile \
  --model panns-cnn14-16k-map-0438 \
  --device auto \
  --energy off \
  --output-dir profiling_candidates/cnn14-auto
```

`auto` expands to CPU plus locally usable accelerators. A failed accelerator smoke test becomes a
campaign diagnostic and does not invalidate successful cards from other backends.

You can also repeat explicit selectors:

```bash
uv run torch-dae model profile \
  --model <model-card-id> \
  --device cpu \
  --device cuda:0 \
  --energy off \
  --output-dir profiling_candidates/<campaign-name>
```

## 10. Stop at candidate evidence

Successful generation and validation do not make the files canonical. Independent review should
consider raw timing, variability, unsupported conditions, memory semantics, energy coverage,
runtime classification, and any limitations.

Only after that review should a maintainer consider byte-preserving promotion into
`technical_cards/<model-id>/` and a normal pull request.

For an agent-assisted campaign, use the prompts in {doc}`../skill/prompt-library` or the profiling
skill template at `skills/audio-model-profiling/templates/agent-request.md`.
