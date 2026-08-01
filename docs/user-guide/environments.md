# Environments

Each environment ID resolves a committed specification, source manifest, lockfile, project file,
and verification script without consulting a model card. Materialized environments live in ignored
`.torch-dae/` runtime state and are keyed by a deterministic fingerprint.

An **environment** is the ignored isolated runtime; its specification, project file, lockfile,
source manifest, and verification script remain committed.

```python
from pathlib import Path
from torch_dae.environment import EnvironmentManager

manager = EnvironmentManager(Path.cwd())
definition = manager.resolve_environment("example-environment")
materialized = manager.materialize_environment(definition.environment_id)
verified = manager.verify_environment(
    definition.environment_id,
    expected_fingerprint=definition.environment_fingerprint,
)
print(materialized.status, verified.environment_fingerprint)
```

Successful verification returns `verification_status = passed` and
`lifecycle_state = environment_verified` only when import and smoke observations are nonempty,
uniquely named, and all passed. Failed verification evidence remains at `materialized` with an
explicit failure classification and may preserve partial or absent observations; it cannot support
a verified model card.

```bash
uv run torch-dae env create <card-id>
uv run torch-dae env ensure <card-id>
uv run torch-dae env resolve <environment-id>
uv run torch-dae env materialize <environment-id>
uv run torch-dae env verify <environment-id>
uv run torch-dae env info <card-id>
uv run torch-dae env run <card-id> -- <command>
uv run torch-dae env remove <card-id>
```

The root control plane remains model-agnostic. See the detailed
[environment management guide](../environment-management.md) for materialization and failure
semantics. Invalid/missing committed inputs and cross-document identity mismatches fail before
materialization; direct materialization reuses only matching state and never silently replaces a
conflicting fingerprint. The card-oriented `create`, `ensure`, `run`, `info`, and `remove` methods
remain compatibility conveniences where applicable and delegate through the recommended environment
ID. Source ambiguity remains explicit until decided.

See {class}`torch_dae.environment.EnvironmentSpecification`,
{class}`torch_dae.environment.EnvironmentManager`,
{meth}`~torch_dae.environment.EnvironmentManager.resolve_environment`,
{meth}`~torch_dae.environment.EnvironmentManager.materialize_environment`,
{meth}`~torch_dae.environment.EnvironmentManager.verify_environment`,
{meth}`~torch_dae.environment.EnvironmentManager.create`,
{meth}`~torch_dae.environment.EnvironmentManager.ensure`, and
{meth}`~torch_dae.environment.EnvironmentManager.verify` in
{doc}`../api/environment-specifications` and {doc}`../api/environment-lifecycle`.
