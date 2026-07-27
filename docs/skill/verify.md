# Verify mode

`verify` performs controlled checkpoint acquisition, checksum validation, state loading, forward
inference, probability checks where supported, declared embedding checks, device behavior, and
gradient behavior. Results are written as explicit runtime observations.

A successful import alone is not runtime verification. Every declared public output and lifecycle
precondition must be checked.
