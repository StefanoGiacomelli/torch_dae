# Technical Cards

A Technical Card is immutable profiling evidence for one exact accepted Model Card/checkpoint, one
Profiling v1 protocol version, one device/backend, one execution context, and one profiling session.

It is independent from the Model Card and never extends Model Card lifecycle state.

## Storage model

Canonical accepted evidence uses:

```text
technical_cards/
  <model-id>/
    <technical-card-id>.json
    <technical-card-id>.npz
```

Candidate evidence uses the same JSON/NPZ pair structure in the repository-local directory selected
with `--output-dir`. The pair is relocation-safe: the JSON references the NPZ relative to its own
directory.

The profiler refuses to write directly into the canonical `technical_cards/` tree.

## What is in the JSON

The JSON contains structured, human-inspectable evidence:

- exact Model Card/checkpoint reference;
- profiler/protocol/runtime classification;
- privacy-safe hardware and software context;
- concrete device/backend and execution context;
- execution precision;
- synthetic input provenance;
- canonical duration;
- architecture evidence;
- cold-start observations;
- minimum-input search;
- timed benchmark conditions;
- host and accelerator memory;
- energy evidence;
- raw NPZ manifest;
- measurement coverage and comparability metadata;
- limitations, campaign reference, and optional supersession.

For a field-by-field developer-oriented explanation, use
{doc}`../user-guide/technical-cards`.

## What is in the NPZ

The `.npz` contains bounded lossless raw observations needed to independently reconstruct important
summaries. In current canonical PANNs cards this includes the 50 integer-nanosecond latency samples
for each successful condition.

Inspect it safely with:

```python
from pathlib import Path
import numpy as np

path = Path("technical_cards/<model-id>/<technical-card-id>.npz")

with np.load(path, allow_pickle=False) as raw:
    print(raw.files)
    latency_ns = raw["raw_ns__canonical-b1"]
    print(latency_ns.shape, latency_ns.dtype)
```

Do not use pickle-enabled loading for Technical Card raw assets.

## Candidate campaign metadata

One profiling invocation also writes orchestration metadata to:

```text
.torch-dae/profiling/<campaign-id>/campaign-result.json
```

That file records requested/detected/attempted devices, successful run/card paths, failed device
smoke diagnostics, protocol versions, energy mode, and workspace identity. It is useful for local
campaign review but is not itself a contributed Technical Card.

## Validate a card

```bash
uv run torch-dae technical-card validate \
  profiling_candidates/<campaign>/<technical-card-id>.json
```

Machine-readable validation:

```bash
uv run torch-dae technical-card validate \
  profiling_candidates/<campaign>/<technical-card-id>.json \
  --json
```

Validation checks the card contract, identity, Model Card/checkpoint association, protocol,
provenance/privacy constraints, raw asset hash/manifest, raw-vs-summary consistency, duplicate IDs,
and supersession rules.

Optional evidence may correctly be unavailable. Missing FLOPs/MACs, unavailable energy, or absent
accelerator memory on a CPU run are not failures when the card represents that coverage honestly.

## Inspect and list

Print a Technical Card as normalized JSON:

```bash
uv run torch-dae technical-card inspect \
  technical_cards/<model-id>/<technical-card-id>.json
```

List canonical cards:

```bash
uv run torch-dae technical-card list
uv run torch-dae technical-card list --json
```

`list` searches only the canonical repository tree, not arbitrary candidate workspaces.

## Runtime classification

A card records one of:

```text
canonical
modified_runtime
unknown_runtime
```

This classification describes the `torch-dae` runtime/source state that produced the evidence.
A structurally valid `modified_runtime` card is not automatically comparable to a canonical card;
analysis must respect the recorded execution context and protocol metadata.

## Immutability and correction

Accepted Technical Card JSON/NPZ bytes are append-only evidence. If a session or interpretation
needs correction, produce a new card. The later card may declare a `supersedes` relationship and a
reason; it must not overwrite the original evidence.

## Privacy

Hardware metadata describes reproducibility-relevant configuration classes, not machine identity.
Technical Cards must not contain hostname, username, IP address, serial number, MAC address, machine
UUID, credentials, tokens, or home-directory paths.

Optional contributor name/GitHub handle does not affect technical validity or card identity.

## Model Card versus Technical Card

```text
MODEL CARD                            TECHNICAL CARD
what is supported?                    how did it behave here?
------------------                    -----------------------
model / variant / checkpoint          one accepted Model Card
waveform contract                     one protocol version
outputs and embeddings                one device/backend
verified environment                  one execution context
runtime verification                  one profiling session
known model limitations               latency/memory/energy/etc.

                one Model Card -> zero or many Technical Cards
```

See {doc}`overview`, {doc}`protocol`, and {doc}`contributing` for the complete profiling workflow.
