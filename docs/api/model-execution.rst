Model execution
===============

The public waveform contract is

.. math::

   \mathbf{x} \in \mathbb{R}^{B \times C \times T},

where :math:`\mathbf{x}` is the input waveform tensor, :math:`B` is batch size, :math:`C` is the
number of audio channels, and :math:`T` is the number of samples. ``sample_rate`` is measured in
hertz. Optional ``valid_lengths`` has shape ``[B]`` and records valid sample counts before padding.

Wrappers own resampling, padding, normalization, model-native conversion, checkpoint
deserialization, and numerical behavior. Consequently, construction and output values can be
implementation-dependent even though the public axes and return containers are stable.

.. container:: api-flow

   **waveform [B,C,T]** → **preprocess** → **model-native input** → **forward pass** →
   **task output and/or selected embedding**

``forward`` preserves access to differentiable native tensors. ``predict_probability`` is valid
only when the card declares probability support. ``available_embeddings`` enumerates declared
choices, and ``compute_embedding`` selects the requested identifier or the declared default.
Unsupported operations raise
:class:`~torch_dae.core.errors.UnsupportedCapabilityError`; they do not masquerade as dependency
errors.

.. autoclass:: torch_dae.core.model.AudioModelProtocol

   .. automethod:: from_random
   .. automethod:: from_pretrained
   .. automethod:: load_checkpoint
   .. automethod:: preprocess
   .. automethod:: forward
   .. automethod:: predict_probability
   .. automethod:: available_embeddings
   .. automethod:: compute_embedding

.. autoclass:: torch_dae.core.preprocessing.WaveformInputContract
