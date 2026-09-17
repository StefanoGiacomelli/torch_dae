# Technical Cards

A Technical Card is immutable profiling evidence for one exact Model Card/checkpoint, one profiling
protocol, one device/backend, one execution context, and one complete profiling session.

It is not a Model Card extension.

## Storage

```text
technical_cards/
  <model-id>/
    <technical-card-id>.json
    <technical-card-id>.npz
```

JSON stores identity, provenance, summaries, limitations, and the raw-array manifest. `.npz` stores
bounded lossless raw observations and must be readable with pickle disabled.

## Identity

IDs are hash-derived from schema/protocol/model/checkpoint/runtime/hardware/execution-context/device
identity plus a random run nonce. Measurement values are not part of ID construction.

## Privacy

Hardware fingerprints describe configuration classes, not physical machines. Never capture
hostname, username, IP, serial, MAC address, machine UUID, credentials, tokens, or home paths.

Contributor display name/GitHub handle is optional and does not affect technical validity.

## Runtime classification

Cards may be `canonical`, `modified_runtime`, or `unknown_runtime`. Modified/local-code cards may be
valid evidence, but default reference analytics should use canonical cards unless broadened.

## Raw evidence

Raw `.npz` may include 50 latency observations per successful condition plus bounded RAM,
accelerator-memory, and energy/power samples. Do not commit large profiler traces, system dumps,
checkpoint payloads, or unbounded logs.

## Immutability

Accepted Technical Cards are append-only. Corrections create a new card and may declare
`supersedes`; original bytes remain unchanged.
