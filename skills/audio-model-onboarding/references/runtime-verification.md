# Runtime Verification

Verification covers construction, checkpoint loading, invalid checkpoint behavior, model variant
agreement, inputs, outputs, embeddings, device movement, dtypes, NaN/Inf checks, and repeated-call
behavior.

The structured report must include environment fingerprint, checkpoint hash, source revision,
package identity, test inputs, observed outputs, embedding observations, warnings, failures,
unsupported capabilities, and runtime evidence.

Schema validity alone is insufficient for `runtime_verified`.
