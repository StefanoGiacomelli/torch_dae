Environment lifecycle
=====================

Committed state and ignored runtime state are deliberately separated:

.. container:: api-flow

   **environment specification + lockfile + sources** → **materialization result** →
   **passed environment verification result + matching fingerprint** → identifies **verified
   isolated runtime**

.. container:: api-flow

   **runtime verification target** → binds **model + checkpoint + environment** →
   produces **checkpoint-specific passed verification report** → final **model card** consumes
   evidence

Failed environment or runtime results remain diagnostic evidence. Their existence never promotes
an environment or model-card lifecycle state.

A passed environment result requires nonempty import and smoke evidence, every observation passed,
and unique observation names. A successful result proves both successful global status and complete
successful coverage of the evidence required by its authority contract.

The direct manager loads and cross-checks environment, source-manifest, lockfile, project file, and
verification-script identities without a card. ``create`` fails if current state already exists. ``ensure`` reuses
a verified target or removes and rebuilds invalid current state. Neither operation deletes older
fingerprints. ``verify`` checks metadata hashes, Python, the local project wheel, installed sources,
dependency consistency, and the committed verification script without repairing state.

Materialization resolves exact CPython, runs locked ``uv`` synchronization, builds a deterministic
local project wheel, installs package/Git/vendored sources, records inventories, and writes ignored
runtime reports. Commands run without shell activation and with a sanitized environment. There is no
cross-process lock, so callers must serialize concurrent mutation for the same
environment/fingerprint.

A lightweight metadata-only example needs no model or network:

.. code-block:: python

   from pathlib import Path
   from torch_dae.environment import EnvironmentManager

   manager = EnvironmentManager(Path.cwd())
   definition = manager.resolve_environment("example-environment")
   print(definition.python_version, definition.environment_fingerprint)

``info`` represents a missing specification in its return value. Other loading operations propagate
missing-file, Pydantic, path, and identity errors. ``remove`` is destructive but bounded to ignored
environment materializations for one logical environment ID (directly or through a card convenience
lookup); it does not remove checkpoints, committed inputs, or reports.

.. autoclass:: torch_dae.environment.manager.InstalledSource

.. autoclass:: torch_dae.environment.manager.ResolvedEnvironment

.. autoclass:: torch_dae.environment.manager.EnvironmentInfo

.. autoclass:: torch_dae.environment.manager.EnvironmentVerification

.. autoclass:: torch_dae.environment.results.ResolvedEnvironmentDefinition

.. autoclass:: torch_dae.environment.results.EnvironmentMaterializationResult

.. autoclass:: torch_dae.environment.results.EnvironmentVerificationResult

.. autoclass:: torch_dae.environment.manager.EnvironmentManager

   .. automethod:: from_repository_root
   .. automethod:: specification_path
   .. automethod:: load_specification
   .. automethod:: load_sources_manifest
   .. automethod:: fingerprint_for
   .. automethod:: resolve_environment
   .. automethod:: materialize_environment
   .. automethod:: verify_environment
   .. automethod:: resolved_environment
   .. automethod:: create
   .. automethod:: ensure
   .. automethod:: verify
   .. automethod:: remove
   .. automethod:: info
   .. automethod:: run
   .. automethod:: info_json
