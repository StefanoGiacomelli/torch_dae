# Architecture And Embeddings

Cover model classes, topology, frontend, temporal/spatial processing, backbone, heads,
normalization, activations, pooling, sequence handling, candidate outputs, candidate embeddings, and
variant differences.

Embedding candidates must distinguish architectural intermediate tensors, pooled representations,
task-head inputs, pre-logit representations, post-activation outputs, sequence-level embeddings,
frame-level embeddings, and latent codes.

Declare an embedding only when origin, shape semantics, batch/time dimensions, and extraction
behavior are known. Ambiguous embeddings require a user decision. Classifier logits and task
decisions are not embeddings.
