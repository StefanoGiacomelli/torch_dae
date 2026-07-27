# Checkpoints

A checkpoint specification identifies one concrete asset source and records its expected checksum.
Acquisition is explicit, cache-contained, hash-verified, and compatible with offline operation when
the required content is already cached.

A **checkpoint** is a byte asset plus its checkpoint-specific source, format, loader, and hash
evidence. Use metadata inspection before acquisition:

```python
from pathlib import Path
from torch_dae.core import CheckpointManager

state = CheckpointManager(Path.cwd()).info("example-card")
print(state["expected_sha256"], state["cached"])
```

```bash
uv run torch-dae checkpoint ensure <card-id>
uv run torch-dae checkpoint info <card-id>
uv run torch-dae checkpoint remove <card-id>
```

Weights are never committed or silently downloaded during import or model construction. See the
detailed [checkpoint management guide](../checkpoint-management.md). Missing assets, offline cache
misses, hash mismatches, access/authentication blockers, and acquisition failures remain distinct.
`remove` deletes only ignored cache state for the card's asset.

See {class}`torch_dae.core.CheckpointSpec`,
{class}`torch_dae.core.CheckpointManager`,
{meth}`~torch_dae.core.CheckpointManager.ensure`,
{meth}`~torch_dae.core.CheckpointManager.info`, and
{meth}`~torch_dae.core.CheckpointManager.remove` in {doc}`../api/checkpoints`.
