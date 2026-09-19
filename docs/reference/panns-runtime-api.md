# PANNs runtime API

This page is the user-facing contract for the three accepted PANNs wrappers. It supplements the
Python docstrings without modifying the immutable PANNs integration artifact captured by the
accepted onboarding evidence.

The wrappers are:

```python
from torch_dae.models.panns import (
    PannsCnn14_16kMap0438,
    PannsResNet38Map0434,
    PannsWavegramLogmelCnn14Map0439,
)
```

All three expose the same public runtime surface and differ in fixed identity, native sample rate,
architecture, checkpoint, and input-length constraints. See {doc}`../models/panns-runtime` for the
model table and {doc}`../tutorials/panns-inference` for an end-to-end example.

## Construction

### `from_random()`

```python
model = ModelClass.from_random(variant=None)
```

Constructs the fixed architecture with upstream random initialization. These wrappers do not accept
arbitrary architecture keyword arguments. If `variant` is supplied, it must equal the wrapper's
fixed accepted variant id.

### `from_pretrained()`

```python
model = ModelClass.from_pretrained(
    checkpoint=checkpoint_path,
    variant=None,
)
```

`checkpoint` is required. It may be a local path or a local-path `CheckpointSpec`. A non-local
checkpoint specification must first be materialized through `CheckpointManager` or the checkpoint
CLI. The wrapper never downloads a default checkpoint.

Raises `ValueError` when the checkpoint is omitted or the requested variant disagrees with the
fixed wrapper identity, and `TypeError` for unsupported construction keywords.

### `load_checkpoint()`

```python
model.load_checkpoint(
    checkpoint_path,
    strict=True,
    map_location="cpu",
)
```

Loads the official PANNs `checkpoint["model"]` state-dictionary convention. The payload must exist
locally and contain a mapping under `model`. `strict` is forwarded to PyTorch state-dict loading.

## Input preprocessing

### `preprocess()`

```python
prepared = model.preprocess(
    waveform,
    sample_rate=native_sample_rate,
    valid_lengths=None,
)
```

`waveform` must be a floating-point PyTorch tensor shaped `[B, 1, T]`. The method validates the
input contract and returns the upstream `[B, T]` waveform through a `PreprocessingOutput`.

The wrapper does **not** automatically:

- resample audio;
- downmix multichannel input;
- normalize amplitude;
- pad, crop, or truncate time samples.

`valid_lengths`, when supplied, must have shape `[B]`. The current PANNs integration does not define
padded-batch semantics, so every value must equal `T`.

`allow_resample` is retained by the common model interface but has no effect here: non-native sample
rates are rejected.

## Inference

### `forward()`

```python
output = model(
    waveform,
    sample_rate=native_sample_rate,
)
```

The returned `AudioModelOutput` contains:

| Access | Shape | Meaning |
|---|---:|---|
| `output.primary` | `[B, 527]` | raw AudioSet logits |
| `output.tensors["logits"]` | `[B, 527]` | same raw logits |
| `output.tensors["probabilities"]` | `[B, 527]` | native sigmoid probabilities |
| `output.tensors["embedding"]` | `[B, 2048]` | accepted post-fc1 clip embedding |

The classifier is multi-label. The probability tensor already applies sigmoid; do not replace it
with softmax. No decision threshold is applied.

### `predict_probability()`

```python
probabilities = model.predict_probability(
    waveform,
    sample_rate=native_sample_rate,
)
```

Returns `[B, 527]` native sigmoid probabilities without thresholding or class aggregation.

## Embeddings

### `available_embeddings()`

Returns a one-element tuple containing the accepted embedding specification:

```text
panns-official-post-fc1-embedding
layout: B,D
dimension: 2048
granularity: clipwise
normalization: none
```

### `compute_embedding()`

```python
embedding = model.compute_embedding(
    waveform,
    sample_rate=native_sample_rate,
    embedding_id="panns-official-post-fc1-embedding",
)

assert embedding.tensor.shape == (batch_size, 2048)
```

Omit `embedding_id` to use the accepted default. Any other id raises `KeyError`. The embedding is
obtained from the same native forward path as the classifier output.

See {doc}`../user-guide/embeddings` for semantic interpretation and downstream-use guidance.

## Labels and default embedding

```python
labels = model.class_labels
default_id = model.default_embedding_id
```

`class_labels` contains the 527 AudioSet display names in classifier-index order.
`default_embedding_id` is `panns-official-post-fc1-embedding`.

## Device placement and evaluation mode

The wrappers are ordinary PyTorch modules inside the isolated model environment. Move the model and
input tensor to a supported device using standard PyTorch operations and use `model.eval()` for
pretrained inference. Device support is distinct from empirical profiling coverage: consult the
canonical Technical Cards when you need measured behavior for a specific execution context.
