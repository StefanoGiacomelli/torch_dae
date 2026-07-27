Registry
========

The registry is the user-facing boundary between committed model-card JSON and optional wrapper
code. Every call discovers ``model_cards/**/*.json`` in sorted path order and validates every card.
Duplicate identifiers fail discovery. Card listing, lookup, and path resolution remain
model-agnostic; wrapper code and its optional dependencies are imported only by
:meth:`~torch_dae.ModelCardRegistry.get_model_class`.

The ordinary offline workflow is:

.. code-block:: python

   from pathlib import Path
   from torch_dae import ModelCardRegistry

   registry = ModelCardRegistry(Path.cwd())
   for card in registry.list_cards():
       print(card.card_id, card.card_status)
   card = registry.get_card("example-card")
   path = registry.get_card_path(card.card_id)

An empty repository returns an empty tuple. Unknown identifiers raise ``KeyError`` for card and path
lookups. Wrapper resolution additionally propagates import failures, raises ``AttributeError`` for a
missing attribute, and raises ``TypeError`` when the entry point is not a class.

.. autoclass:: torch_dae.ModelCardRegistry

   .. automethod:: list_cards
   .. automethod:: get_card
   .. automethod:: get_card_path
   .. automethod:: get_model_class
