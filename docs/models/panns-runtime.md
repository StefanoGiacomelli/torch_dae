# PANNs integration

`torch-dae` currently provides three accepted checkpoint-specific integrations from the PANNs
(AudioSet) model family:

| Model Card ID | Upstream variant | Sample rate | Reported upstream AudioSet mAP | Wrapper |
|---|---|---:|---:|---|
| `panns-cnn14-16k-map-0438` | `Cnn14_16k` | 16 kHz | 0.438 | `PannsCnn14_16kMap0438` |
| `panns-resnet38-map-0434` | `ResNet38` | 32 kHz | 0.434 | `PannsResNet38Map0434` |
| `panns-wavegram-logmel-cnn14-map-0439` | `Wavegram_Logmel_Cnn14` | 32 kHz | 0.439 | `PannsWavegramLogmelCnn14Map0439` |

The mAP values above are officially reported for the checkpoint identities and were **not** locally
reproduced by `torch-dae` runtime verification.

All three cards are `runtime_verified`. The repository records authoritative checkpoint identity,
accepted isolated environments, runtime-verification reports, and canonical Profiling v1 Technical
Cards. Checkpoint payload bytes are acquired on demand and are not committed to the repository.

## Public waveform contract

All three wrappers require a floating-point PyTorch tensor with shape:

```text
[B, 1, T]
```

where:

- `B` is batch size;
- the channel dimension must be exactly `1`;
- `T` is the waveform sample count.

The `sample_rate` argument must exactly match the model's native rate. The wrapper does not
silently repair input data.

### No automatic preprocessing outside the model frontend

The integration performs **no automatic**:

- resampling;
- stereo-to-mono downmixing;
- amplitude normalization;
- input padding;
- cropping;
- truncation.

The caller must provide an already prepared floating-point mono waveform at the exact native sample
rate.

`valid_lengths` may be supplied as a tensor shaped `[B]`, but every value must equal `T`. Variable
valid lengths inside padded batches are not currently supported.

## Input-length constraints

| Model | Minimum samples | Minimum duration | Additional constraint |
|---|---:|---:|---|
| `Cnn14_16k` | 4,960 | 0.310 s at 16 kHz | none beyond minimum |
| `ResNet38` | 9,920 | 0.310 s at 32 kHz | none beyond minimum |
| `Wavegram_Logmel_Cnn14` | 10,236 | ≈0.3199 s at 32 kHz | `T % 640` must be `0..319` or `636..639` |

The Wavegram constraint preserves alignment between its learned waveform and log-mel branches. An
incompatible length raises an explicit error; the wrapper does not pad or truncate to force branch
alignment.

## Output contract

`forward()` returns an {class}`torch_dae.core.AudioModelOutput` with raw classification logits as the
primary tensor.

| Access path | Semantic meaning | Shape | Notes |
|---|---|---|---|
| `output.primary` | raw AudioSet logits | `[B,527]` | pre-sigmoid |
| `output.tensors["logits"]` | raw AudioSet logits | `[B,527]` | same tensor role as primary |
| `output.tensors["probabilities"]` | multi-label probabilities | `[B,527]` | `sigmoid(logits)` |
| `output.tensors["embedding"]` | clipwise representation | `[B,2048]` | post-`fc1`, pre-classifier |

`predict_probability()` returns the same native sigmoid probabilities exposed in
`output.tensors["probabilities"]`. The wrapper applies no softmax, threshold, or class aggregation.

The package also includes the authoritative ordered 527-row AudioSet label resource. The
`model.class_labels` property returns display names in classifier-index order.

## Default embedding

All three cards expose exactly one verified default embedding:

```text
panns-official-post-fc1-embedding
```

Its runtime layout is `[B,2048]`. It is the upstream-named representation after `ReLU(fc1)` and the
following dropout call, immediately before `fc_audioset`. In `eval()` mode the dropout call is the
identity.

Use:

```python
embedding = model.compute_embedding(waveform, sample_rate)
print(embedding.embedding_id)
print(embedding.layout)
print(embedding.tensor.shape)
```

Other internal temporal, Wavegram-branch, log-mel-branch, or fused representations are not exposed
as accepted public embeddings.

## Construction and checkpoint loading

Random initialization is available for architecture-level use:

```python
model = PannsCnn14_16kMap0438.from_random()
```

Pretrained construction requires an explicit local checkpoint path or a local-path
{class}`torch_dae.core.CheckpointSpec`:

```python
model = PannsCnn14_16kMap0438.from_pretrained(checkpoint_path).eval()
```

`from_pretrained()` never downloads a default checkpoint. Acquire accepted bytes through
`CheckpointManager` or:

```bash
uv run torch-dae checkpoint ensure panns-cnn14-16k-map-0438 --json
```

## Runtime isolation

Model identity and label discovery are lazy and do not require importing the tensor runtime. Tensor
execution requires the accepted model environment. The current PANNs environments use Python
3.12.13 and an isolated dependency stack including NumPy 2.4.6, PyTorch 2.13.0, and TorchLibrosa
0.1.0.

Managed materialization is currently constrained to macOS on arm64. The accepted cards record CPU
and Apple MPS as locally tested; CUDA remains upstream-declared but locally unverified.

## Accepted evidence in the repository

For each of the three models, the repository contains:

- one `runtime_verified` Model Card under `model_cards/panns/`;
- one accepted runtime-verification report under `verification_reports/<model-id>/`;
- the accepted environment definition under `environments/<model-id>/`;
- three canonical Technical Cards under `technical_cards/<model-id>/` covering CPU
  `single_thread`, CPU `native_default`, and Apple MPS profiling contexts;
- one raw `.npz` measurement asset paired with each Technical Card.

## Scientific/source limitations

The accepted cards preserve two source-level limitations rather than hiding them:

1. exact historical checkpoint-to-source revision equivalence remains unresolved;
2. the PANNs paper describes a Hamming analysis window, while the selected official source uses a
   Hann window.

The integration therefore validates the selected implementation/checkpoint behavior without
claiming to resolve that historical discrepancy.

See {doc}`../tutorials/panns-inference` for an end-to-end usage example and
{doc}`../profiling/overview` for empirical profiling.
