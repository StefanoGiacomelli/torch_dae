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

When one repository exposes multiple architecture or checkpoint candidates, scope preprocessing,
runtime-interface, output, architecture, dependency, and environment claims to declared variant and
checkpoint IDs where their applicability differs. Scope embedding candidates the same way. Empty
scope means report-wide applicability; it must not create or imply a candidate identity.
