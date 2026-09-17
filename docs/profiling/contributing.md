# Contributing profiling evidence

Technical Card contribution is specified but not enabled yet. Do not hand-author profiling evidence
before the generator and validators are implemented.

Planned workflow:

```text
accepted runtime-verified Model Card
-> torch-dae model profile ...
-> local Technical Card validation
-> normal GitHub pull request with JSON + NPZ
-> CI validation
-> merge
-> later site/analytics ingestion
```

A profiling pull request contains only:

```text
technical_cards/<model-id>/<technical-card-id>.json
technical_cards/<model-id>/<technical-card-id>.npz
```

Do not modify the referenced Model Card.

Planned CI validation includes schema/ID/raw hash/array manifest, Model Card/checkpoint association,
protocol, raw-vs-summary consistency, provenance, privacy, duplicate IDs, and supersession validity.

Future analytics aggregate only explicitly compatible protocol/input/batch/precision/device/thread
conditions by default and should emphasize median, IQR, min/max, card count, and execution-context
count.
