# Integrate mode

`integrate` adds a generic public wrapper only after model identity, checkpoint selection,
environment resolution, source strategy, preprocessing, output semantics, and embedding choices are
resolved. The wrapper accepts canonical `[B,C,T]` waveforms, a sample rate in hertz, and optional
`[B]` valid lengths.

Model-specific imports remain lazy and occur only inside controlled construction, verification, or
inference. The root package must stay importable without those dependencies.
