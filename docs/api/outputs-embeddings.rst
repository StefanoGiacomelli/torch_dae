Outputs and embeddings
======================

The control plane deliberately uses a structural tensor placeholder, so importing these contracts
does not import PyTorch or another model runtime.

Output field contract
---------------------

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Contract
     - Required fields
     - Optional/defaulted fields and invariants
   * - ``TensorLike``
     - ``shape``
     - No dtype, device, gradient, or numerical behavior is implied.
   * - ``AudioModelOutput``
     - ``primary``, ``tensors``
     - ``lengths=None``, empty ``metadata``, ``native_output=None``. Named tensors and the primary
       output remain implementation-defined.
   * - ``EmbeddingOutput``
     - ``embedding_id``, ``tensor``, ``layout``
     - ``lengths=None``, ``timestamps=None``, empty ``metadata``. ``layout`` defines axes; no fixed
       model-specific rank is assumed.
   * - ``PreprocessingOutput``
     - ``model_input``, ``sample_rate``
     - ``valid_lengths=None`` and empty tensor/metadata mappings. Sample rate is in hertz.
   * - ``EmbeddingSpec``
     - schema/version identity, semantic description, location, layout, granularity, transforms,
       dtype, status, rationale
     - ``dimension`` and ``temporal_hop_seconds`` are optional positive values; ``evidence_ids``
       defaults empty. Status is ``declared``, ``verified``, or ``unsupported``.

For example, a sequence representation may use ``layout="B,T,D"`` where ``B`` is batch, ``T`` is
the implementation-declared frame axis, and ``D`` is feature dimension. A pooled representation may
instead use ``"B,D"``. Optional lengths must be interpreted in the output domain, not silently as
input samples.

.. autoclass:: torch_dae.core.outputs.TensorLike

.. autoclass:: torch_dae.core.outputs.AudioModelOutput

.. autoclass:: torch_dae.core.outputs.EmbeddingOutput

.. autoclass:: torch_dae.core.outputs.PreprocessingOutput

.. autoclass:: torch_dae.core.embeddings.EmbeddingSpec
