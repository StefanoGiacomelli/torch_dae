Checkpoint management
=====================

A checkpoint specification identifies one concrete byte asset. Source fields are mutually exclusive:

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Source value
     - Required location fields
     - Rejected location fields
   * - ``https``
     - ``url``
     - repository, package, revision, release tag, and local path
   * - ``github_release``
     - ``repository_id``, ``release_tag``, ``filename``
     - URL, package fields, and local path
   * - ``huggingface``
     - ``repository_id``, ``filename``; optional ``revision``
     - URL, package fields, release tag, and local path
   * - ``package_bundle``
     - ``package``, exact ``package_version``, ``filename``
     - URL, repository, revision, release tag, and local path
   * - ``local_path``
     - repository-relative ``local_path``
     - every remote/package field and ``filename``

``filename`` and ``local_path`` cannot be absolute or escape the repository. ``expected_sha256`` is
upstream/committed integrity evidence; ``observed_sha256`` records local observation. If both are
present, validation requires equality.

For acquired bytes :math:`b`, integrity is recorded as

.. math::

   h = \operatorname{SHA256}(b),

where :math:`h` is the lowercase hexadecimal SHA-256 digest. The relation detects content mismatch;
it does not by itself authenticate the publisher or guarantee transport security.

Acquisition streams remote content into temporary ignored state, hashes it, then installs it under
``.torch-dae/checkpoints/<checkpoint-id>/<h>`` with a materialization record. Valid
content-addressed entries are reused, including in offline mode. GitHub and Hugging Face tokens are
read only at request boundaries; report text and command output are sanitized. Failures distinguish
missing assets, offline cache misses, hash mismatches, authentication/access problems, subprocess
errors, and general acquisition errors.

``info`` is the safe metadata-only call:

.. code-block:: python

   from pathlib import Path
   from torch_dae.core import CheckpointManager

   state = CheckpointManager(Path.cwd()).info("example-card")
   print(state["source_type"], state["cached"])

It validates card metadata and local cache entries but never downloads. ``remove`` deletes only the
matching ignored checkpoint cache tree and leaves cards, source assets, and environments unchanged.

.. autoclass:: torch_dae.core.checkpoint.CheckpointSourceType

.. autoclass:: torch_dae.core.checkpoint.LicenseRecord

.. autoclass:: torch_dae.core.checkpoint.CheckpointSpec

.. autoclass:: torch_dae.core.checkpoint.ResolvedCheckpoint

.. autoclass:: torch_dae.core.checkpoint.CheckpointMaterializationRecord

.. autoclass:: torch_dae.core.checkpoint.CheckpointManager

   .. automethod:: ensure
   .. automethod:: info
   .. automethod:: remove
