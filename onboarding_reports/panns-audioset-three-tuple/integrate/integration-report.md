# PANNs three-tuple integration recovery

The existing PANNs integration was retained and recovered by adding the generic cross-phase
artifact-supersession contract. All three fixed identities remain integrated from upstream revision
`58d0e624a2055fb1f2b5fb06369ecd726cfe64e0` with lazy model imports and isolated runtime
dependencies.

## Outputs

Each wrapper accepts floating-point `[B,1,T]` waveform input at its exact native sample rate and
returns raw `[B,527]` logits, native sigmoid `[B,527]` probabilities, and the selected `[B,2048]`
post-fc1 embedding. Ordered AudioSet metadata contains 527 unique, contiguous classes.

No wrapper resamples, downmixes, normalizes, pads, crops, or truncates. Padded batches with unequal
valid lengths remain unsupported.

## Duration contract

- `Cnn14_16k`: `T >= 4960` samples.
- `ResNet38`: `T >= 9920` samples.
- `Wavegram_Logmel_Cnn14`: `T >= 10236` and `T % 640` is `0..319` or `636..639`.

Random-weight forwards passed for every identity at exactly 0.64, 1.0, and 2.0 seconds. Boundary
failures now produce explicit adapter errors rather than opaque pooling or branch-concatenation
errors. These observations are synthetic and establish no pretrained behavior.

## Supersession and provenance

The six environment `sources.json` and `verify_environment.py` transitions record the unchanged
resolve-environment hashes, current integrate hashes, exact prior handoff digest, and a bounded
reason. The historical handoff is not modified.

Source provenance records complete upstream file hashes, exact selected ranges, hashes for every
selected source segment and AST node, deterministic ordering, the import adaptation, logits
extension, final integrated hashes, and a byte-reproduced deterministic diff.

## Phase boundary

No checkpoint was requested, downloaded, loaded, or inspected. No pretrained inference occurred.
No model card or runtime-verification report was created. Exact checkpoint/source equivalence,
checkpoint compatibility, pretrained numerical behavior, padded batches, temporal branch
embeddings, and non-CPU platform behavior remain unresolved for later phases.

After candidate validation and atomic promotion, the recommended next mode is `verify`.
