# Profiling protocol v1

Protocol ID: `audio-inference-v1`.

`project_spec.md` is authoritative.

## Input

Only deterministic seeded IID uniform white noise in `[-1, 1]`, generated as `float32`.

No speech, music, soundscape, dataset, or external audio file participates in protocol v1.

## Devices

Planned selectors:

```text
auto
cpu
mps
cuda
cuda:<index>
```

`auto` includes CPU and attempts locally supported accelerators. A device must pass a bounded public
wrapper smoke inference before a Technical Card can be emitted.

## Precision

Only verified/native execution precision is benchmarked in v1.

## Minimum input

Start at 10 seconds, bracket a failing/passing boundary by halving/doubling, then binary-search in
integer sample-count space. Maximum probe duration is 120 seconds. Non-monotonic support is reported
rather than forced into a fabricated minimum.

## Batch matrix

At canonical duration:

```text
1, 2, 4, 8
```

The empirical minimum duration is additionally measured at batch 1 when distinct.

## Repetitions

Every steady-state condition uses 10 warmups and 50 measured inferences.

## Timing

Canonical latency is synchronized end-to-end execution of the public waveform wrapper from prepared
tensor to requested public output. Internal wrapper DSP is included; input generation and file I/O
are excluded.

Store raw observations and summarize mean, standard deviation, min/max, p50, p90, p95, p99,
coefficient of variation, throughput, real-time factor, and speed factor.

Cold initialization/first inference remains separate.

## CPU regimes

CPU cards contain both `single_thread` and `native_default` regimes.

## Memory

Host RAM/process RSS is first-class evidence. Where supported record total RAM, RSS before/after
model load, pre-resource-pass RSS, sampled peak RSS, and post-pass RSS.

CUDA/MPS allocator surfaces retain their native backend semantics. On unified-memory systems, host
RSS and accelerator/driver allocation are not summed into a fabricated total.

## Energy

`--energy auto` uses CodeCarbon when available; `--energy off` disables energy instrumentation.

Classify evidence as `hardware_measured`, `software_estimated`, `unavailable`, or `failed`.
Interactive privilege elevation may be offered only with explicit consent. Never edit sudoers, store
credentials, or silently geolocate contributors.

Latency and resource instrumentation use separate passes.
