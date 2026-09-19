# Profiling protocol v1

Protocol ID: `audio-inference-v1`<br>
Protocol version: `1.0.0`

Profiling v1 is the implemented deterministic benchmark protocol used by `torch-dae model profile`.
The project specification remains normative; this page explains the operational contract.

## Eligibility

The target must be an accepted checkpoint-specific Model Card with:

```text
card_status = runtime_verified
```

The profiler materializes/verifies the card's recommended environment and resolves the accepted
checkpoint before device discovery.

## Synthetic input

Only deterministic seeded IID uniform white noise is used:

```text
x[n] ~ U(-1, 1)
dtype = float32
```

No speech, music, soundscape, external audio file, or benchmark dataset participates in v1. Input
generation/allocation is outside canonical timed inference.

Each card records the relevant generator provenance, seed, sample rate, sample count, batch size,
channel count, and synthetic tensor digest.

## Device selectors

Implemented selectors are:

```text
auto
cpu
mps
cuda
cuda:<index>
```

`--device` is repeatable. `auto` expands to CPU plus usable MPS and every discovered CUDA device.
Each resolved device is smoke-tested through the public wrapper before profiling.

CPU produces two run contexts:

```text
single_thread
native_default
```

Each accelerator produces its own device/backend run. Explicit device failures remain diagnostics;
they are not silently substituted with another backend.

## Precision

Protocol v1 uses the wrapper's accepted native/verified execution precision. It does not
automatically enable FP16, BF16, AMP, quantization, or `torch.compile`.

The Technical Card records dtype/autocast information as observed evidence.

## Canonical duration

If the accepted runtime contract declares a fixed/canonical duration, that value is used. Otherwise
v1 uses:

```text
10.0 seconds
```

The current variable-length PANNs integrations therefore use the 10-second protocol default.

## Minimum supported input search

At batch 1, the profiler empirically searches integer sample-count space:

1. probe 10 seconds;
2. after success, repeatedly halve until a lower failing boundary is found;
3. after failure, repeatedly double until success or the 120-second upper bound;
4. binary-search the failing/passing bracket;
5. recheck the candidate minimum and the immediately smaller sample count when valid.

The ordered probe sequence is retained. If support is non-monotonic, the result is reported as
`non_monotonic`; no fabricated minimum is emitted.

This measurement does not rewrite the Model Card's input contract. It describes one profiling
session on one backend.

## Benchmark matrix

At canonical duration:

```text
batch_size = 1, 2, 4, 8
```

If the empirical minimum duration differs from the canonical duration, it is additionally measured
at batch 1.

A tested condition that cannot execute because of resource/input limits is recorded as
`unsupported` with a reason rather than being silently omitted.

## Warmup and repetitions

Every successful steady-state condition uses exactly:

```text
warmup_inferences   = 10
measured_inferences = 50
```

There is no adaptive repetition count and no outer repetition loop in v1.

## Timing scope

Canonical latency measures:

```text
prepared waveform tensor
        -> public torch-dae wrapper
        -> requested public output
```

Wrapper-owned DSP is included. Input generation, file I/O, serialization, campaign orchestration,
and report generation are excluded. Device synchronization is applied where required.

The first inference belongs to cold-start evidence and is never one of the 50 steady-state samples.

## Timing summaries

The 50 integer-nanosecond observations are preserved in the raw NPZ and summarized as:

- mean and population standard deviation;
- minimum and maximum;
- p50, p90, p95, and p99;
- coefficient of variation;
- throughput in items/s;
- forward calls/s;
- real-time factor (RTF);
- speed factor.

For mean inference latency `L` seconds, batch size `B`, and per-item audio duration `D` seconds:

```text
throughput_items_per_second = B / L
forward_calls_per_second    = 1 / L
real_time_factor            = L / (B * D)
speed_factor                = (B * D) / L
```

For a successful positive-duration condition, `speed_factor` is the reciprocal of RTF. Lower RTF
and higher speed factor indicate faster processing relative to represented audio duration.

## Cold-start evidence

Where separable, v1 records:

- wrapper/model initialization;
- checkpoint loading;
- device placement/readiness;
- first inference.

These measurements are distinct from steady-state latency.

## Instrumentation separation

Canonical latency runs with minimal instrumentation. Memory and energy are collected in a separate,
bounded resource pass so sampling and energy instrumentation do not contaminate the timing samples.

## Architecture evidence

Mandatory evidence includes:

- total/trainable/non-trainable parameter counts;
- parameter bytes and buffer bytes;
- state-dict tensor bytes;
- dtype distribution;
- module count.

FLOPs/MACs are optional. When unavailable or only partially covered, the card records that status
instead of presenting a partial count as a complete total.

## Host memory

Where available, the card records total RAM plus process RSS around model load and the resource
pass, including sampled peak RSS. USS/PSS may also be present.

A sampled process peak is not the same concept as a model's parameter bytes and should not be
compared as if they measured the same quantity.

## Accelerator memory

CUDA and MPS retain backend-native semantics. CUDA may report current/peak allocated and reserved
allocator values. MPS may report current tensor allocations and driver allocations.

On unified-memory systems, host RSS and accelerator/driver allocation surfaces overlap conceptually
and must not be summed into a fabricated independent-memory total.

## Energy

Public policy:

```text
--energy auto
--energy off
```

When enabled, current v1 uses CodeCarbon as the energy backend and classifies evidence as:

```text
hardware_measured
software_estimated
unavailable
failed
```

The card may record CPU, accelerator, and RAM energy, total energy over accounted components,
average power, measurement duration/interval, privilege usage, limitations, and failure reason.

Interpretation requires two additional fields:

```text
coverage_complete
unaccounted_components
```

`total_energy_kwh` is the sum of components actually represented by the evidence. It is not a claim
of whole-run energy when coverage is incomplete.

Privileged local counters are allowed only when the user explicitly supplies
`--allow-privileged-energy`. No sudoers modification, credential persistence, or silent geolocation
is part of the protocol.

## Raw evidence

Each successful Technical Card references one compact `.npz` asset. The raw manifest stores the
relative filename, SHA-256, media type, and array names/dtypes/shapes. The asset must load with
`allow_pickle=False`.

For normal successful timing conditions, arrays are named from the condition, for example:

```text
raw_ns__canonical-b1
raw_ns__canonical-b2
raw_ns__canonical-b4
raw_ns__canonical-b8
raw_ns__minimum-b1
```

Only arrays actually produced by the session are present.

## Protocol-controlled versus user-controlled settings

| User chooses | Protocol v1 chooses |
|---|---|
| Model Card ID | synthetic input distribution |
| device selector(s) | canonical-duration rule |
| energy `auto` / `off` | batch matrix |
| privileged-energy consent | minimum-input algorithm |
| candidate output directory | warmup count |
| JSON/text CLI presentation | measured iteration count |
|  | CPU thread regimes |
|  | summary statistics |

See {doc}`overview` for the operational overview and {doc}`../user-guide/technical-cards` for result
interpretation.
