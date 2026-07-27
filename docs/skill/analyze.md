# Analyze mode

`analyze` inventories an upstream repository without importing or executing its code. It records
repository identity, scientific claims, architecture candidates, dependencies, preprocessing,
outputs, checkpoint candidates, embedding candidates, source strategies, open questions, and
confidence counts.

The result is a machine-readable analysis report plus deterministic Markdown. It does not create a
model environment, download a checkpoint, or add a wrapper.
