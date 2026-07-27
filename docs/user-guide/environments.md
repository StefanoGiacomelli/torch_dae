# Environments

Each model card refers to a committed environment specification, source manifest, lockfile, project
file, and verification script. Materialized environments live in ignored `.torch-dae/` runtime
state and are keyed by a deterministic fingerprint.

An **environment** is the ignored isolated runtime; its specification, project file, lockfile,
source manifest, and verification script remain committed.

```python
from pathlib import Path
from torch_dae.environment import EnvironmentManager

manager = EnvironmentManager(Path.cwd())
state = manager.info("example-card")
if state.specification_exists:
    spec = manager.load_specification("example-card")
    print(manager.fingerprint_for(spec))
```

```bash
uv run torch-dae env create <card-id>
uv run torch-dae env ensure <card-id>
uv run torch-dae env verify <card-id>
uv run torch-dae env info <card-id>
uv run torch-dae env run <card-id> -- <command>
uv run torch-dae env remove <card-id>
```

The root control plane remains model-agnostic. See the detailed
[environment management guide](../environment-management.md) for materialization and failure
semantics. Invalid/missing committed inputs and cross-document identity mismatches fail before
materialization; `create` rejects existing target state, while `ensure` reuses verified state or
rebuilds invalid current state. Source ambiguity remains explicit until decided.

See {class}`torch_dae.environment.EnvironmentSpecification`,
{class}`torch_dae.environment.EnvironmentManager`,
{meth}`~torch_dae.environment.EnvironmentManager.create`,
{meth}`~torch_dae.environment.EnvironmentManager.ensure`, and
{meth}`~torch_dae.environment.EnvironmentManager.verify` in
{doc}`../api/environment-specifications` and {doc}`../api/environment-lifecycle`.
