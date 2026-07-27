"""Lazy model-card registry."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from torch_dae.cards.models import ModelCard
from torch_dae.cards.validation import validate_model_card_path
from torch_dae.core.errors import DuplicateModelCardError


class ModelCardRegistry:
    """Discover validated model cards while deferring wrapper imports.

    Parameters
    ----------
    repository_root
        Repository containing ``model_cards`` and ``schemas/model-card.schema.json``.

    Attributes
    ----------
    repository_root
        Repository root supplied by the caller.
    model_cards_root
        Directory recursively searched for JSON model cards.

    Notes
    -----
    Discovery is lazy and repeated for every public operation. Listing cards, reading a card, and
    resolving its path validate JSON but never import wrapper code. Only
    :meth:`get_model_class` imports the card's ``module:Class`` entry point.

    See Also
    --------
    torch_dae.cards.models.ModelCard
        Validated checkpoint-specific metadata returned by the registry.
    """

    def __init__(self, repository_root: Path) -> None:
        self.repository_root = repository_root
        self.model_cards_root = repository_root / "model_cards"

    def _discover(self) -> dict[str, tuple[Path, ModelCard]]:
        discovered: dict[str, tuple[Path, ModelCard]] = {}
        for path in sorted(self.model_cards_root.glob("**/*.json")):
            card = validate_model_card_path(
                path, self.repository_root / "schemas/model-card.schema.json"
            )
            if card.card_id in discovered:
                raise DuplicateModelCardError(f"duplicate model card id: {card.card_id}")
            discovered[card.card_id] = (path, card)
        return discovered

    def list_cards(self) -> tuple[ModelCard, ...]:
        """Return every discovered card in deterministic path order.

        Returns
        -------
        tuple of ModelCard
            Freshly validated cards sorted by their repository paths.

        Raises
        ------
        pydantic.ValidationError
            If a discovered card violates its model contract.
        torch_dae.core.errors.DuplicateModelCardError
            If two files declare the same ``card_id``.

        Notes
        -----
        The method reads committed JSON files but creates no files and imports no wrapper modules.
        """

        return tuple(item[1] for item in self._discover().values())

    def get_card(self, card_id: str) -> ModelCard:
        """Return the validated card identified by ``card_id``.

        Parameters
        ----------
        card_id
            Exact canonical identifier stored in a model-card JSON file.

        Returns
        -------
        ModelCard
            The validated checkpoint-specific card.

        Raises
        ------
        KeyError
            If discovery finds no matching identifier.
        pydantic.ValidationError
            If any discovered card is invalid.

        Notes
        -----
        Lookup performs a fresh lazy discovery and does not import the selected wrapper.
        """

        discovered = self._discover()
        if card_id not in discovered:
            raise KeyError(card_id)
        return discovered[card_id][1]

    def get_card_path(self, card_id: str) -> Path:
        """Return the repository path of a discovered card.

        Parameters
        ----------
        card_id
            Exact canonical model-card identifier.

        Returns
        -------
        pathlib.Path
            Path yielded by the deterministic ``model_cards/**/*.json`` discovery.

        Raises
        ------
        KeyError
            If no card has ``card_id``.
        pydantic.ValidationError
            If card discovery encounters invalid JSON.
        """

        discovered = self._discover()
        if card_id not in discovered:
            raise KeyError(card_id)
        return discovered[card_id][0]

    def get_model_class(self, card_id: str) -> type[Any]:
        """Import and return the wrapper class declared by a card.

        Parameters
        ----------
        card_id
            Exact canonical model-card identifier.

        Returns
        -------
        type
            Class resolved from the card's ``identity.wrapper_entry_point``.

        Raises
        ------
        KeyError
            If no card has ``card_id``.
        ValueError
            If the entry point is not in ``module:attribute`` form.
        ImportError
            If the wrapper module or one of its dependencies cannot be imported.
        AttributeError
            If the module does not expose the declared attribute.
        TypeError
            If the resolved attribute is not a class.

        Notes
        -----
        This is the registry's only operation that imports model-specific code. Import behavior
        after resolution is implementation-dependent and may require an isolated model environment.
        """

        card = self.get_card(card_id)
        module_name, _, attribute = card.identity.wrapper_entry_point.partition(":")
        if not module_name or not attribute:
            raise ValueError("wrapper_entry_point must use module:attribute")
        module = importlib.import_module(module_name)
        value = getattr(module, attribute)
        if not isinstance(value, type):
            raise TypeError(f"{card.identity.wrapper_entry_point} did not resolve to a class")
        return value
