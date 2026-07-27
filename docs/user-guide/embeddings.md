# Embeddings

Embedding declarations record the public key, semantic meaning, tensor rank, dimensions, dtype,
gradient behavior, and whether an embedding is the selected default. A card may expose multiple
representations, but only one is the default.

The public waveform input contract is `[B,C,T]` plus an integer sample rate in hertz, with optional
`[B]` valid lengths. Wrapper outputs must preserve batch identity and validate declared tensor
shapes. Candidate representations remain unresolved until upstream evidence or controlled runtime
observations establish their meaning.

An **embedding** is a named representation declared by
{class}`torch_dae.core.EmbeddingSpec`. Integrators enumerate choices with
{meth}`~torch_dae.core.AudioModelProtocol.available_embeddings` and compute one with
{meth}`~torch_dae.core.AudioModelProtocol.compute_embedding`:

```python
result = model.compute_embedding(waveform, 16_000, embedding_id=None)
print(result.embedding_id, result.layout, result.tensor.shape)
```

Here `None` selects the declared default. {class}`torch_dae.core.EmbeddingOutput` carries the
runtime tensor, axis layout, optional output lengths and timestamps, and metadata. Unknown IDs and
unsupported embedding capability fail explicitly; rank and temporal units come from the declaration
and runtime evidence. See {doc}`../api/model-execution`, {doc}`../api/outputs-embeddings`, and
{doc}`../api/capabilities`.
