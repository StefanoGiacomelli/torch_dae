Capabilities and public errors
==============================

Capabilities separate discoverable API shape from checkpoint-specific support. Integrators declare
support for random initialization, checkpoint loading, probabilities, and embeddings. A false
capability may include a reason; calling :meth:`torch_dae.core.capabilities.Capability.require`
raises :class:`UnsupportedCapabilityError`.

``UnsupportedCapabilityError`` means the selected integration declares an operation unsupported.
``FeatureNotAvailableError`` means a visible generic operation is intentionally unavailable. Neither
means that an optional runtime dependency merely failed to import.

.. autoclass:: torch_dae.core.capabilities.Capability

   .. automethod:: require

.. autoclass:: torch_dae.core.capabilities.ModelCapabilities

.. autoexception:: torch_dae.core.errors.FeatureNotAvailableError

.. autoexception:: torch_dae.core.errors.UnsupportedCapabilityError
