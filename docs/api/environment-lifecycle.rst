Environment lifecycle
=====================

Committed state and ignored runtime state are deliberately separated:

.. container:: api-flow

   **model card** → references **environment specification + lockfile + sources** →
   produces **environment fingerprint** → identifies **isolated runtime**

.. container:: api-flow

   **model card** → owns **checkpoint specification** → resolves **content-addressed checkpoint** →
   supplies runtime verification evidence tied to **environment fingerprint**

The manager loads and cross-checks card, environment, source-manifest, lockfile, project file, and
verification-script identities. ``create`` fails if current state already exists. ``ensure`` reuses
a verified target or removes and rebuilds invalid current state. Neither operation deletes older
fingerprints. ``verify`` checks metadata hashes, Python, the local project wheel, installed sources,
dependency consistency, and the committed verification script without repairing state.

Materialization resolves exact CPython, runs locked ``uv`` synchronization, builds a deterministic
local project wheel, installs package/Git/vendored sources, records inventories, and writes ignored
runtime reports. Commands run without shell activation and with a sanitized environment. There is no
cross-process lock, so callers must serialize concurrent mutation for the same card/fingerprint.

A lightweight metadata-only example needs no model or network:

.. code-block:: python

   from pathlib import Path
   from torch_dae.environment import EnvironmentManager

   manager = EnvironmentManager(Path.cwd())
   state = manager.info("example-card")
   if state.specification_exists:
       spec = manager.load_specification("example-card")
       print(spec.python.constraint, spec.python.resolved_version)
       print(manager.fingerprint_for(spec))

``info`` represents a missing specification in its return value. Other loading operations propagate
missing-file, Pydantic, path, and identity errors. ``remove`` is destructive but bounded to ignored
environment materializations for one canonical card; it does not remove checkpoints, committed
inputs, or reports.

.. autoclass:: torch_dae.environment.manager.InstalledSource

.. autoclass:: torch_dae.environment.manager.ResolvedEnvironment

.. autoclass:: torch_dae.environment.manager.EnvironmentInfo

.. autoclass:: torch_dae.environment.manager.EnvironmentVerification

.. autoclass:: torch_dae.environment.manager.EnvironmentManager

   .. automethod:: from_repository_root
   .. automethod:: specification_path
   .. automethod:: load_specification
   .. automethod:: load_sources_manifest
   .. automethod:: fingerprint_for
   .. automethod:: create
   .. automethod:: ensure
   .. automethod:: verify
   .. automethod:: remove
   .. automethod:: info
   .. automethod:: run
   .. automethod:: info_json
