"""Lazy model-card registry."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from torch_dae.cards.models import ModelCard
from torch_dae.cards.validation import validate_model_card_path
from torch_dae.core.errors import DuplicateModelCardError


class ModelCardRegistry:
    """Registry derived from validated `model_cards/**/*.json` files."""

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
        """List cards without importing wrapper code."""

        return tuple(item[1] for item in self._discover().values())

    def get_card(self, card_id: str) -> ModelCard:
        """Retrieve a validated card by id without importing wrapper code."""

        discovered = self._discover()
        if card_id not in discovered:
            raise KeyError(card_id)
        return discovered[card_id][1]

    def get_card_path(self, card_id: str) -> Path:
        """Retrieve the exact discovered JSON path for a card id."""

        discovered = self._discover()
        if card_id not in discovered:
            raise KeyError(card_id)
        return discovered[card_id][0]

    def get_model_class(self, card_id: str) -> type[Any]:
        """Lazily resolve the future wrapper entry point."""

        card = self.get_card(card_id)
        module_name, _, attribute = card.identity.wrapper_entry_point.partition(":")
        if not module_name or not attribute:
            raise ValueError("wrapper_entry_point must use module:attribute")
        module = importlib.import_module(module_name)
        value = getattr(module, attribute)
        if not isinstance(value, type):
            raise TypeError(f"{card.identity.wrapper_entry_point} did not resolve to a class")
        return value
