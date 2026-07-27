Environment specifications
==========================

Committed environment inputs describe one isolated runtime without putting model dependencies in the
root environment.

Source strategies
-----------------

.. list-table::
   :header-rows: 1
   :widths: 18 35 47

   * - Value
     - Fields
     - Contract
   * - ``package``
     - ``source_id``, ``role``, ``package``, exact ``version``
     - Install an explicitly identified official package.
   * - ``git``
     - ``source_id``, ``role``, ``url``, immutable ``revision``, ``build="wheel"``
     - Build a wheel from an explicit pinned repository revision.
   * - ``vendored``
     - identity/role, ``upstream_url``, immutable ``upstream_revision``, ``copied_files``,
       ``adaptation_description``, ``justification``
     - Copy a documented minimal repository-relative subset.

Specification fields
--------------------

``PythonSpecification`` requires a version constraint and resolved version; the resolved version must
satisfy the constraint. ``PlatformSpecification`` keeps ``resolved_on``, ``expected_compatible``,
and ``verified`` evidence separate. ``EnvironmentVerificationSpec.script`` is repository-relative.

``EnvironmentSpecification`` requires schema version ``1.0.0``, ``environment_id``,
``model_card_id``, Python and platform specifications, ``dependency_manager="uv"``, and
repository-relative ``lockfile``, ``project_file``, ``sources_file``, and verification script.
``EnvironmentSourcesManifest`` requires matching environment identity and unique source IDs.
Cross-document identity and exact conventional paths are enforced when
:class:`~torch_dae.environment.manager.EnvironmentManager` loads the complete card/environment set.

Fingerprint implementation
--------------------------

The implementation constructs this payload:

.. code-block:: text

   payload = {
       "specification": specification JSON using aliases,
       "lockfile_sha256": SHA256(exact lockfile bytes),
       "sources_manifest": source-manifest JSON using aliases,
       "resolved_python_version": specification.python.resolved_version,
       "target_platform": canonical OS/architecture tag,
       "local_package_identity": clean Git HEAD plus content digest, or content digest only,
   }
   fingerprint = SHA256(canonical_json(payload))

Canonical JSON is UTF-8, has sorted keys and compact separators. The local content digest includes
sorted build inputs: project metadata/readme, package source files, and recognized backend inputs;
each relative path and byte sequence is length-prefixed before hashing. This is an identity
algorithm, not a claim that upstream indexes or external services are reproducible forever.

.. autoclass:: torch_dae.environment.specification.SourceInstallationType

.. autoclass:: torch_dae.environment.specification.PythonSpecification

.. autoclass:: torch_dae.environment.specification.PlatformSpecification

.. autoclass:: torch_dae.environment.specification.OfficialPackageSource

.. autoclass:: torch_dae.environment.specification.PinnedGitSource

.. autoclass:: torch_dae.environment.specification.VendoredAdaptationSource

.. autoclass:: torch_dae.environment.specification.EnvironmentVerificationSpec

.. autoclass:: torch_dae.environment.specification.EnvironmentSpecification

.. autoclass:: torch_dae.environment.specification.EnvironmentSourcesManifest
