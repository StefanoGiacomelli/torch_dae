Runtime verification
====================

A verification report binds observations to ``model_card_id``, ``environment_id``,
``environment_fingerprint``, and ``checkpoint_sha256``. It also records creation time, platform,
device, accepted input contracts, tensor observations, embedding results, passed and unsupported
capabilities, limitations, and individual checks.

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
       tensor, embedding, capability, limitation, and check tuples
     - Fingerprint and checkpoint digest are lowercase 64-character SHA-256 values.

For an observed tensor :math:`\mathbf{y}` with rank :math:`N`,

.. math::

   \operatorname{rank}(\mathbf{y}) = N = |\operatorname{shape}(\mathbf{y})|,

where each shape element is a named :class:`TensorDimension`. No axis name or model-specific output
rank is assumed. A successful report is runtime evidence for the exact card/environment/checkpoint
identity; it does not establish unsupported capabilities or broader scientific equivalence. A card
may advance to ``runtime_verified`` only when it references such a report, while the contract itself
does not read or adjudicate the report file.

.. autoclass:: torch_dae.environment.verification.VerificationCheck

.. autoclass:: torch_dae.environment.verification.TensorDimension

.. autoclass:: torch_dae.environment.verification.TensorObservation

.. autoclass:: torch_dae.environment.verification.VerificationReport
