# Run a pretrained PANNs model

This tutorial executes the accepted `panns-cnn14-16k-map-0438` integration end to end. The same
pattern applies to the other accepted PANNs cards after changing the Model Card ID and native sample
rate.

The current accepted PANNs managed environments are constrained to macOS on arm64. On another
platform, you can still inspect the cards and repository evidence, but managed environment
materialization will reject the unsupported host.

## 1. Choose the Model Card

```bash
CARD="panns-cnn14-16k-map-0438"
uv run torch-dae card show "$CARD"
```

This checkpoint expects 16 kHz mono floating-point waveform input shaped `[B,1,T]`.

## 2. Materialize and verify the model environment

```bash
uv run torch-dae env ensure "$CARD"
```

`env ensure` reuses an already valid matching runtime or materializes and verifies the accepted
environment when required. The environment is stored under ignored `.torch-dae/` runtime state.

## 3. Acquire the accepted checkpoint

Acquire the checkpoint explicitly and capture the resolved local path:

```bash
CHECKPOINT="$(
  uv run torch-dae checkpoint ensure "$CARD" --json \
  | uv run python -c 'import json, sys; print(json.load(sys.stdin)["path"])'
)"

printf '%s\n' "$CHECKPOINT"
```

The wrapper does not perform this download by itself. The path returned above identifies the local,
integrity-checked cache entry.

After successful acquisition, repeat the command with `--offline` when you want to require local
cache reuse:

```bash
uv run torch-dae checkpoint ensure "$CARD" --offline
```

## 4. Create a small inference script

Create `/tmp/torch-dae-panns-demo.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

import torch

from torch_dae import ModelCardRegistry

card_id = sys.argv[1]
checkpoint_path = Path(sys.argv[2])

registry = ModelCardRegistry(Path.cwd())
card = registry.get_card(card_id)
model_class = registry.get_model_class(card_id)

sample_rate = card.input.sample_rate_hz
model = model_class.from_pretrained(checkpoint_path).eval()

# One second of silence is sufficient for this API demonstration.
# Replace it with your own already-prepared float waveform tensor.
waveform = torch.zeros(1, 1, sample_rate, dtype=torch.float32)

with torch.no_grad():
    output = model(waveform, sample_rate)
    embedding = model.compute_embedding(waveform, sample_rate)

logits = output.tensors["logits"]
probabilities = output.tensors["probabilities"]

print("logits:", tuple(logits.shape))
print("probabilities:", tuple(probabilities.shape))
print("embedding:", tuple(embedding.tensor.shape))
print("embedding id:", embedding.embedding_id)

values, indices = probabilities[0].topk(5)
print("top-5 probabilities for this demonstration input:")
for value, index in zip(values.tolist(), indices.tolist(), strict=True):
    print(f"  {index:3d}  {model.class_labels[index]:40s}  {value:.6f}")
```

The silence input is used only to demonstrate the API. Its predicted labels are not a meaningful
model-quality example.

## 5. Run the script inside the accepted environment

```bash
uv run torch-dae env run "$CARD" -- \
  python /tmp/torch-dae-panns-demo.py "$CARD" "$CHECKPOINT"
```

Expected tensor layouts are:

```text
logits:        (1, 527)
probabilities: (1, 527)
embedding:     (1, 2048)
```

The exact numeric values depend on the input waveform and checkpoint.

## 6. Replace the demonstration waveform with your audio

The wrapper expects your loader or preprocessing pipeline to produce:

```text
floating tensor
shape: [B, 1, T]
sample rate: exactly the Model Card native rate
```

For this Cnn14 checkpoint:

```text
sample rate = 16000 Hz
minimum T   = 4960 samples
```

The wrapper does not resample, downmix, normalize amplitude, pad, crop, or truncate. Prepare these
properties before calling the model.

For `ResNet38` use 32 kHz and at least 9,920 samples. For `Wavegram_Logmel_Cnn14` use 32 kHz, at
least 10,236 samples, and satisfy the additional `T % 640` branch-alignment constraint documented
in {doc}`../models/panns-runtime`.

## 7. Understand the results

`forward()` returns raw logits as `output.primary`. The same result object also contains:

```python
output.tensors["logits"]         # [B,527]
output.tensors["probabilities"]  # [B,527], sigmoid(logits)
output.tensors["embedding"]      # [B,2048]
```

Because AudioSet tagging is multi-label, the probabilities are independent sigmoid outputs. Do not
replace them with a softmax unless you are deliberately changing the task semantics outside the
accepted wrapper contract.

`compute_embedding()` returns an {class}`torch_dae.core.EmbeddingOutput` for the verified default
representation:

```text
embedding_id = panns-official-post-fc1-embedding
layout       = B,D
dimension    = 2048
granularity  = clipwise
```

Continue with {doc}`../user-guide/model-execution` for the complete wrapper contract or
{doc}`../user-guide/embeddings` for embedding semantics.
