# Model execution

This page describes the public execution contract for accepted model wrappers. Model-specific pages
remain authoritative for exact shapes, sample rates, minimum durations, and optional capabilities.

## Execution happens in the model runtime

The root `torch_dae` package is intentionally lightweight. Importing a model wrapper that depends on
PyTorch should happen inside the model's accepted isolated environment, not by adding every model
runtime dependency to the root environment.

The high-level path is:

```text
Model Card
   │
   ├── environment → materialize / verify
   ├── checkpoint  → acquire / validate
   └── wrapper     → import inside model environment
                         │
                         └── forward / probability / embedding
```

For an executable PANNs walkthrough, see {doc}`../tutorials/panns-inference`.

## Resolving the wrapper

A Model Card stores `identity.wrapper_entry_point`. Use
{class}`torch_dae.ModelCardRegistry` to resolve it when execution is intended:

```python
from pathlib import Path
from torch_dae import ModelCardRegistry

registry = ModelCardRegistry(Path.cwd())
model_class = registry.get_model_class("panns-cnn14-16k-map-0438")
```

Registry listing and card inspection do not import model wrappers. `get_model_class()` is the
explicit import boundary and therefore requires the model runtime dependencies to be available.

## Construction methods

The generic execution protocol distinguishes random construction from checkpoint-loaded
construction.

### Random initialization

For wrappers that declare the capability:

```python
model = model_class.from_random()
```

This is useful for architecture checks and synthetic boundary tests. It does not establish
pretrained behavior.

### Pretrained initialization

For the accepted PANNs wrappers:

```python
model = model_class.from_pretrained(checkpoint_path).eval()
```

The checkpoint must already exist locally. `from_pretrained()` does not download a default asset.
Use:

```bash
uv run torch-dae checkpoint ensure <card-id> --json
```

to materialize the checkpoint through the accepted acquisition path and obtain its local path.

## Waveform input

The generic public audio interface uses a batched waveform plus an integer sample rate. Exact
constraints are model-specific.

For all current PANNs wrappers:

```text
waveform.shape = [B, 1, T]
waveform dtype  = floating point
sample_rate     = exact native integer sample rate
```

The wrapper explicitly rejects:

- non-tensor waveform input;
- a rank other than 3;
- channel count other than one;
- non-floating waveform dtype;
- a sample rate different from the accepted native rate;
- durations below the model-specific minimum;
- unsupported Wavegram branch-alignment lengths;
- padded batches whose `valid_lengths` values differ from `T`.

The wrapper performs no automatic resampling, downmixing, amplitude normalization, cropping,
padding, or truncation.

## `preprocess()`

For PANNs, `preprocess()` validates the public `[B,1,T]` contract and converts the waveform to the
upstream `[B,T]` layout expected by the selected implementation. It preserves the original waveform
in the returned preprocessing metadata and records that no resampling, normalization, or
padding/truncation occurred.

The `allow_resample` argument exists on the generic interface but does not enable resampling for the
current PANNs integration.

## `forward()`

Call the wrapper like a normal PyTorch module or invoke `forward()` explicitly:

```python
with torch.no_grad():
    output = model(waveform, sample_rate)
```

For PANNs, {class}`torch_dae.core.AudioModelOutput` contains:

```text
primary                         raw logits [B,527]
tensors["logits"]               raw logits [B,527]
tensors["probabilities"]        sigmoid probabilities [B,527]
tensors["embedding"]            clipwise embedding [B,2048]
metadata["probability_activation"] = "sigmoid"
metadata["class_count"]            = 527
```

The native upstream output is also retained in `native_output` for controlled integration and
verification work. Normal application code should prefer the public fields above.

## `predict_probability()`

When the Model Card declares probability support:

```python
probabilities = model.predict_probability(waveform, sample_rate)
```

For PANNs this is exactly the native `sigmoid(logits)` tensor. No thresholding or class aggregation
is applied. Because the task is multi-label, the probabilities do not sum to one and should not be
interpreted as a softmax distribution.

## `compute_embedding()`

```python
embedding = model.compute_embedding(
    waveform,
    sample_rate,
    embedding_id=None,
)
```

`None` selects the Model Card default. For the current PANNs cards the only accepted public
embedding is `panns-official-post-fc1-embedding`, with layout `[B,2048]`.

See {doc}`embeddings` for details.

## Device placement

Model wrappers are PyTorch modules and follow normal `.to(device)` semantics inside a compatible
runtime:

```python
model = model.to("cpu")
```

or, on a supported Apple environment:

```python
model = model.to("mps")
waveform = waveform.to("mps")
```

The accepted PANNs Model Cards record CPU and MPS as locally tested. CUDA is upstream-declared but
not locally verified by the accepted cards. Do not treat an upstream device path as equivalent to
local `torch-dae` runtime verification.

## Labels

The PANNs wrappers expose 527 AudioSet display labels in classifier-index order:

```python
labels = model.class_labels
print(len(labels))
print(labels[0])
```

The index of a logit/probability therefore maps directly to the corresponding entry in
`model.class_labels`.

## Error behavior is part of the contract

`torch-dae` prefers explicit rejection to silent input repair. If a wrapper raises because the
sample rate, channel count, duration, valid lengths, or checkpoint is incompatible, fix the caller's
input or select the correct Model Card rather than bypassing the validation.
