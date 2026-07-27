# Integration Planning

The integration plan must specify wrapper path, source strategy, checkpoint strategy, construction,
random and checkpoint initialization, preprocessing ownership, sample-rate behavior, channel
behavior, waveform scaling, padding/truncation, valid lengths, forward signature, output
normalization, task capabilities, embeddings, device movement, evaluation mode, deterministic
behavior, tests, and errors.

Production integration is permitted only in an explicitly requested `integrate` mode after analysis
review, source/variant/checkpoint/embedding decisions, environment strategy resolution, and explicit
user authorization. Integration must remain scoped to the selected model and must not start
verification, add root model dependencies, commit checkpoints, or create a Git commit.
