# Contributing profiling evidence

The Profiling v1 generator and validators are implemented (`torch-dae model profile`,
`torch-dae technical-card validate|inspect|list`; see `src/torch_dae/profiling/`). Candidate
Technical Cards produced by a run are **not** automatically part of the repository: promotion into
the official `technical_cards/<model-id>/` tree is a separate, later, human-reviewed step that
this implementation deliberately does not automate. Do not hand-author profiling evidence, and do
not copy candidate output into `technical_cards/` yourself.

Workflow:

```text
accepted runtime-verified Model Card
-> torch-dae model profile --model <card-id> --device auto --energy auto \
     --output-dir <repository-local-candidate-workspace>   # never technical_cards/ directly
-> torch-dae technical-card validate <candidate>.json
-> independent review of the candidate JSON + NPZ
-> (separate, later) promotion into technical_cards/<model-id>/
-> normal GitHub pull request with the promoted JSON + NPZ
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

`torch-dae technical-card validate` already covers schema/ID/raw hash/array manifest, Model
Card/checkpoint association, protocol recognition, raw-vs-summary consistency, provenance,
privacy, duplicate IDs, and supersession validity (`src/torch_dae/profiling/validation.py`). CI
wiring for pull requests that promote cards into `technical_cards/` is not part of this pass.

Future analytics aggregate only explicitly compatible protocol/input/batch/precision/device/thread
conditions by default and should emphasize median, IQR, min/max, card count, and execution-context
count.
