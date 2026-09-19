# Supported models

The public registry contains checkpoint-specific integrations rather than family-level placeholders.
Each card identifies one exact model variant and checkpoint and records its accepted waveform,
output, embedding, environment, device, and limitation contracts.

## Current catalog

| Model Card ID | Variant | Wrapper | SR | Input | Outputs | Default embedding | Status |
|---|---|---|---:|---|---|---|---|
| `panns-cnn14-16k-map-0438` | `Cnn14_16k` | `PannsCnn14_16kMap0438` | 16 kHz | `[B,1,T]` | logits/probabilities `[B,527]` | `[B,2048]` | `runtime_verified` |
| `panns-resnet38-map-0434` | `ResNet38` | `PannsResNet38Map0434` | 32 kHz | `[B,1,T]` | logits/probabilities `[B,527]` | `[B,2048]` | `runtime_verified` |
| `panns-wavegram-logmel-cnn14-map-0439` | `Wavegram_Logmel_Cnn14` | `PannsWavegramLogmelCnn14Map0439` | 32 kHz | `[B,1,T]` | logits/probabilities `[B,527]` | `[B,2048]` | `runtime_verified` |

All three classify the 527 AudioSet classes and expose the upstream-named post-`fc1` clipwise
embedding as the verified default representation.

## Verified execution scope

The accepted Model Cards record CPU and Apple MPS as locally tested devices. CUDA is declared by the
upstream implementation but remains locally unverified. The current managed PANNs environments are
constrained to macOS on arm64.

This distinction matters:

- **upstream-declared** means the original implementation claims or contains a path for that device;
- **locally verified** means the accepted `torch-dae` evidence exercised the checkpoint-specific
  wrapper on that device;
- **profiled** means a separate Technical Card records empirical performance for one execution
  context.

## Checkpoint payloads

The repository records and validates checkpoint identity but does not redistribute the pretrained
payloads. Use:

```bash
uv run torch-dae checkpoint ensure <card-id>
```

The resulting local path can be passed explicitly to the wrapper's `from_pretrained()` method.

## PANNs

See {doc}`panns-runtime` for the exact PANNs waveform, duration, output, embedding, and device contracts, or
follow {doc}`../tutorials/panns-inference` for an end-to-end execution example.
