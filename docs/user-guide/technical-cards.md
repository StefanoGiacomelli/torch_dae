# Reading Technical Cards

A Technical Card is a structured measurement record, not a leaderboard score. Read it together with
the referenced Model Card, protocol, device/backend, execution context, and coverage fields.

This guide focuses on the fields developers normally need when deciding whether a model/runtime is
appropriate for an application.

## Start with four questions

Before comparing numbers, establish:

1. **Which exact model/checkpoint was measured?** — `model`
2. **Which device and runtime produced the measurement?** — `device`, `execution_context`, `software`
3. **Which protocol and runtime classification apply?** — `profiler`, `comparability`
4. **Which evidence is missing or unsupported?** — `coverage`, `limitations`, `energy.coverage_complete`

Only then interpret performance values.

## Field guide

| Section | What it tells you | What to look at first |
|---|---|---|
| `model` | exact accepted Model Card/checkpoint/wrapper | card ID, checkpoint ID/hash, wrapper entry point |
| `profiler` | profiler/protocol and torch-dae provenance | protocol version, runtime classification |
| `hardware` | privacy-safe hardware class | CPU/accelerator model, cores, RAM |
| `software` | execution software stack | OS, Python, PyTorch, CUDA/MPS info |
| `device` | concrete backend | `cpu`, `mps`, `cuda`, device index |
| `execution_context` | comparability-relevant context | thread regime, native precision, fingerprints |
| `synthetic_input` | generated waveform used by v1 | sample rate/count, batch, seed, dtype |
| `architecture` | model/storage size | parameters, bytes, module count, FLOPs status |
| `cold_start` | one-time startup costs | loading/placement/first inference |
| `minimum_input_search` | empirical minimum for this run/backend | status, minimum samples/seconds, boundary checks |
| `conditions` | steady-state benchmark conditions | batch, duration, status, timing |
| `host_memory` | process/host RAM evidence | RSS after load and sampled peak |
| `accelerator_memory` | backend-native device memory surfaces | allocator/driver fields; backend semantics |
| `energy` | resource-pass energy evidence | kind, total/components, coverage, privilege |
| `raw_measurements` | NPZ integrity/array manifest | filename, arrays, shapes, hash |
| `coverage` | tested/unsupported/unavailable summary | unsupported/not-tested conditions, evidence status |
| `comparability` | analysis guardrails | protocol/context/runtime classification |
| `limitations` | explicit caveats | read before drawing conclusions |

## Timing conditions

Each successful `conditions` entry identifies a duration, batch size, sample count, and timing
summary. Typical IDs include:

```text
canonical-b1
canonical-b2
canonical-b4
canonical-b8
minimum-b1
```

A condition may instead be `unsupported` with an explicit reason.

### Latency percentiles

`p50_ns` is the median of the 50 steady-state observations. Higher percentiles (`p90`, `p95`,
`p99`) expose slow-tail behavior that the mean alone can hide.

`coefficient_of_variation` is standard deviation divided by mean. A larger value indicates greater
relative session variability; it is diagnostic context, not an automatic pass/fail criterion.

### Throughput

```text
throughput_items_per_second
```

is the number of batch items processed per second using mean latency. It is useful when the workload
is naturally item-oriented.

### Real-time factor

For batch size `B`, per-item audio duration `D`, and mean call latency `L`:

```text
RTF = L / (B * D)
```

Interpretation:

```text
RTF < 1    faster than real time
RTF = 1    equal to represented real time
RTF > 1    slower than real time
```

RTF uses the total audio duration represented by the batch, not only one item.

### Speed factor

```text
speed_factor = (B * D) / L = 1 / RTF
```

A speed factor of `20` means the measured call processed 20 seconds of represented audio per second
of wall-clock inference time under that exact benchmark condition.

Do not compare speed factors from incompatible durations, protocols, precisions, or execution
contexts as if they were interchangeable.

## Minimum input search

`minimum_input_search` is empirical backend/session evidence. A successful search may report:

- minimum sample count and duration;
- all probe sample counts;
- ordered success/failure probe records;
- whether the passing boundary was rechecked;
- whether the immediately smaller point was verified to fail.

A `non_monotonic` result means the profiler could not represent support as one simple minimum. It is
more informative than fabricating a threshold.

The result does not alter the Model Card's authoritative public input contract.

## Architecture evidence

Parameter count, parameter bytes, buffer bytes, and state-dict tensor bytes describe different
quantities. They should not be substituted for runtime memory.

`flops_macs_status` may be:

```text
complete
partial
unavailable
```

Never quote a partial FLOP/MAC count as if it were complete.

## Memory evidence

### Host memory

Useful fields include:

- `rss_before_model_load_bytes`;
- `rss_after_model_load_bytes`;
- `rss_before_resource_pass_bytes`;
- `rss_sampled_peak_bytes`;
- `rss_after_resource_pass_bytes`.

`rss_sampled_peak_is_sampled` makes explicit that peak RSS is based on sampling rather than a perfect
continuous maximum.

### Accelerator memory

CUDA and MPS expose different memory surfaces. CUDA allocator values and MPS tensor/driver
allocations are backend-specific evidence.

On Apple unified memory, do not add host RSS and MPS driver/tensor allocations into a fabricated
"total memory" value. The surfaces may overlap.

## Energy evidence

Start with:

```text
measurement_kind
coverage_complete
unaccounted_components
privilege_used
```

`measurement_kind` is one of:

```text
hardware_measured
software_estimated
unavailable
failed
```

When numeric energy is present, `total_energy_kwh` is the sum of components represented by the
backend evidence. If `coverage_complete` is false, it is **not** a whole-run energy claim. Read
`unaccounted_components` to learn what is missing.

A card can be scientifically valid with incomplete energy coverage. That incompleteness must simply
remain explicit.

## Cold start versus steady state

`cold_start` captures one-time startup components where separable. The `conditions` timing summaries
capture steady-state calls after warmup. Do not use first-inference latency as a steady-state
latency statistic or vice versa.

## Raw measurements

The Technical Card JSON contains summaries; the NPZ contains raw bounded observations. Before a
sensitive comparison or publication-quality analysis, inspect the raw latency samples rather than
relying only on one aggregate.

```python
import numpy as np

with np.load("<technical-card>.npz", allow_pickle=False) as raw:
    x = raw["raw_ns__canonical-b1"]
    print(x.shape)  # normally (50,) for one successful timing condition
```

The card's `raw_measurements.arrays` manifest defines which arrays actually exist.

## Comparability

A Technical Card records protocol version, execution-context fingerprint, and runtime
classification because measurements are only meaningful within compatible conditions.

For a careful comparison, align at least:

- exact model/checkpoint;
- profiling protocol/version;
- duration/sample count;
- batch size;
- native precision;
- device/backend;
- CPU thread regime where applicable;
- runtime classification;
- execution context.

A difference between two Technical Cards is an observation about those recorded contexts, not a
universal ranking of hardware or models.

## Inspect canonical evidence

List cards:

```bash
uv run torch-dae technical-card list
```

Inspect one:

```bash
uv run torch-dae technical-card inspect \
  technical_cards/<model-id>/<technical-card-id>.json
```

Validate it independently:

```bash
uv run torch-dae technical-card validate \
  technical_cards/<model-id>/<technical-card-id>.json
```

See {doc}`../profiling/technical-cards` for storage/immutability and
{doc}`../profiling/protocol` for the measurement methodology.
