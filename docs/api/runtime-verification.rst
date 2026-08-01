Runtime verification
====================

``RuntimeVerificationTarget`` is the strict request that binds an accepted integration, model
variant/adapter, checkpoint, environment, source manifest, future card identity, expected waveform
and output contract, device scope, acquisition policy, execution limits, and ordered required and
optional check IDs. Completeness-aware targets use schema ``2.0.0`` and require at least one unique
required check; optional checks are unique and disjoint. The target makes no success claim and can
be created before any final model card.

A target-aware verification report binds observations back to that request, declares explicit
overall ``verification_status``, and records ``model_card_id``,
``environment_id``, ``environment_fingerprint``, and ``checkpoint_sha256``. It also records creation time, platform,
device, accepted input contracts, tensor observations, embedding results, passed and unsupported
capabilities, limitations, and individual checks.

A target-aware report repeats the target's exact required and optional check contracts. Its checks
are nonempty, canonical, and unique; all required checks occur exactly once and must pass for overall
success. Undeclared checks and unsupported required checks are invalid. A declared optional check may
be passed or explicitly unsupported; unsupported optional checks require details, a matching
``unsupported_capabilities`` entry, and a recorded limitation.

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Contract
     - Fields
     - Invariants/defaults
   * - ``VerificationCheck``
     - ``name``, ``status``, ``details=None``
     - Status is ``passed``, ``failed``, or ``unsupported``.
   * - ``TensorDimension``
     - ``name``, ``size=None``, ``dynamic=False``, ``description=None``
     - Known size is nonnegative; ``dynamic`` preserves unresolved runtime extent.
   * - ``TensorObservation``
     - ``name``, ``role``, ``component_path``, ``shape``, ``rank``, ``dtype``, ``device``,
       ``lengths=None``, ``temporal_metadata=None``
     - Nonnegative rank equals the number of structured dimensions.
   * - ``VerificationReport``
     - schema/report/card/environment identity, creation/platform/device/checkpoint identity, input,
       tensor, embedding, capability, limitation, check tuples, and overall status
     - A passed target-aware report is nonempty and completely covers passed required checks; a
       failed report contains failed declared evidence. Fingerprint and checkpoint digest are
       lowercase 64-character SHA-256 values.

.. autoclass:: torch_dae.runtime_verification.RuntimeVerificationTarget

For an observed tensor :math:`\mathbf{y}` with rank :math:`N`,

.. math::

   \operatorname{rank}(\mathbf{y}) = N = |\operatorname{shape}(\mathbf{y})|,

where each shape element is a named :class:`TensorDimension`. No axis name or model-specific output
rank is assumed. Only a report with successful overall status is runtime evidence for the exact card/environment/checkpoint
identity; it does not establish unsupported capabilities or broader scientific equivalence. A card
may advance to ``runtime_verified`` only when it references such a hash-, identity-, and
target-check-contract-matching report. Unsupported checks remain representable only as bounded
optional diagnostics when the capability and limitation are explicit and the required runtime
contract passed. Legacy schema ``1.0.0`` reports remain readable conservatively: any failed check
makes them unsuccessful, and they are not silently treated as completeness evidence for a new
runtime-verified card. Failed evidence remains diagnostic and never promotes lifecycle.

.. autoclass:: torch_dae.environment.verification.VerificationCheck

.. autoclass:: torch_dae.environment.verification.TensorDimension

.. autoclass:: torch_dae.environment.verification.TensorObservation

.. autoclass:: torch_dae.environment.verification.VerificationReport
