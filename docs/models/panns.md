# PANNs integration

The PANNs production package represents three fixed model, variant, and checkpoint identities:

- `panns-cnn14-16k-map-0438` (`Cnn14_16k`, 16 kHz);
- `panns-resnet38-map-0434` (`ResNet38`, 32 kHz);
- `panns-wavegram-logmel-cnn14-map-0439` (`Wavegram_Logmel_Cnn14`, 32 kHz).

All three use the minimal official source substrate selected from revision
`58d0e624a2055fb1f2b5fb06369ecd726cfe64e0`. This selection does not prove exact checkpoint/source
equivalence. The upstream source uses a Hann window; the paper's Hamming-window description remains
an explicit discrepancy.

## Public waveform contract

The wrappers require floating-point waveform tensors shaped `[B,1,T]` and the identity's exact
native sample rate. They do not resample, downmix, normalize amplitude, crop, pad, or truncate.
Optional `[B]` `valid_lengths` values are accepted only when every value equals `T`; padded batches
with unequal valid lengths remain unsupported.

Random-weight boundary probes establish architecture-derived minimum lengths of 4,960 samples for
`Cnn14_16k`, 9,920 samples for `ResNet38`, and 10,236 samples for
`Wavegram_Logmel_Cnn14`. The Wavegram/log-mel branches also align only when `T % 640` is in
`0..319` or `636..639`. Inputs outside these constraints are rejected explicitly. The wrappers do
not claim arbitrary-duration support and do not silently repair incompatible lengths.

The canonical forward result exposes raw `[B,527]` logits. Native multi-label sigmoid probabilities
remain available as `probabilities`, and `predict_probability()` returns those probabilities without
softmax, thresholds, or label aggregation.

## Embedding and labels

The default embedding is the upstream-named `[B,2048]` tensor produced after `ReLU(fc1)` and the
following dropout call, immediately before `fc_audioset`. In `eval()` mode the dropout call is the
identity. Temporal, Wavegram-branch, log-mel-branch, and fused representations remain documented
candidates because their timing or extraction contracts have not been verified.

The package includes the authoritative ordered 527-row `class_labels_indices.csv` resource from the
same selected revision. Its indices are contiguous from 0 through 526 and its AudioSet MIDs are
unique.

## Runtime isolation and phase boundary

`import torch_dae` and PANNs identity/label discovery require no model runtime dependencies. Tensor
execution requires the checkpoint-specific isolated environment with Python 3.12.13,
NumPy 2.4.6, PyTorch 2.13.0, and TorchLibrosa 0.1.0.

This integration phase includes no checkpoint payload, pretrained inference, runtime-verification
report, or model card. Random-initialized synthetic checks do not establish checkpoint compatibility
or pretrained numerical behavior.
