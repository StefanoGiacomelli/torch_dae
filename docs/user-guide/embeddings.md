# Embeddings

An embedding in `torch-dae` is a named public representation with declared semantics and tensor
layout. The Model Card records which embeddings are accepted, which one is the default, and what its
network location and dimension mean.

## Generic embedding interface

Wrappers enumerate accepted choices with
{meth}`~torch_dae.core.AudioModelProtocol.available_embeddings`:

```python
for specification in model.available_embeddings():
    print(specification.embedding_id, specification.layout, specification.dimension)
```

Compute one with {meth}`~torch_dae.core.AudioModelProtocol.compute_embedding`:

```python
result = model.compute_embedding(
    waveform,
    sample_rate,
    embedding_id=None,
)

print(result.embedding_id)
print(result.layout)
print(result.tensor.shape)
```

Passing `embedding_id=None` selects the Model Card default.

{class}`torch_dae.core.EmbeddingOutput` carries:

- `embedding_id` — the selected public identifier;
- `tensor` — the runtime tensor;
- `layout` — named axis layout such as `B,D`;
- optional `lengths` and `timestamps` for embeddings whose contract defines them;
- metadata specific to the integration.

Unknown IDs fail explicitly instead of falling back to an arbitrary internal representation.

## Current PANNs embedding

All three accepted PANNs Model Cards expose exactly one verified public embedding:

```text
embedding_id = panns-official-post-fc1-embedding
layout       = B,D
dimension    = 2048
granularity  = clipwise
dtype        = float32 in the verified runtime
```

The representation is the upstream-named tensor after `ReLU(fc1)` and the following dropout call,
immediately before the final `fc_audioset` classifier. In evaluation mode the dropout call is the
identity.

```python
embedding = model.compute_embedding(waveform, sample_rate)
assert embedding.tensor.shape == (waveform.shape[0], 2048)
```

The same tensor is also available from a normal forward pass:

```python
output = model(waveform, sample_rate)
embedding_tensor = output.tensors["embedding"]
```

For the accepted PANNs wrappers, runtime verification established equality between the public
embedding path and the upstream model's native `embedding` output in evaluation mode.

## Clipwise semantics

The PANNs default embedding is clipwise: each input item yields one 2,048-dimensional vector.
There is no public time axis and therefore no embedding timestamps or temporal hop in this contract.

The internal network contains other temporal or branch-specific activations, but those
representations are not exposed as accepted public embeddings because their extraction/timing
semantics have not been verified as part of the current integration.

## Input preparation still matters

`compute_embedding()` uses the same public waveform path as `forward()`. It therefore inherits the
same model-specific input contract. For current PANNs models:

- waveform shape is `[B,1,T]`;
- dtype must be floating point;
- sample rate must exactly match 16 kHz or 32 kHz depending on the Model Card;
- automatic resampling/downmixing/normalization/padding/cropping/truncation is not provided;
- model-specific minimum-duration rules apply;
- unequal padded `valid_lengths` are not supported.

See {doc}`model-execution` and {doc}`../models/panns-runtime` before treating the embedding function as a
generic arbitrary-audio frontend.

## Choosing an embedding in future integrations

A Model Card may expose more than one embedding, but only one may be declared the default. Each
{class}`torch_dae.core.EmbeddingSpec` records the public identifier, network location, layout,
dimension, granularity, pooling/projection/normalization semantics, status, and supporting evidence.

The onboarding workflow does not promote an internal tensor merely because it is easy to hook. A
representation becomes public only when its meaning and extraction contract are supported by
upstream evidence or controlled runtime verification.

See {doc}`../api/outputs-embeddings` for the typed contracts and {doc}`../skill/overview` for the
integration workflow that establishes them.
