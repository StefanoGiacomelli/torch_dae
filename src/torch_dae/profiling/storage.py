"""Canonical `technical_cards/<model-id>/` repository storage layout helpers (Section 28).

Read-only: listing and validation never write to this tree. Nothing here promotes candidate
evidence into it; promotion is a deliberately separate, later step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TECHNICAL_CARDS_DIRECTORY = "technical_cards"


@dataclass(frozen=True)
class TechnicalCardEntry:
    """One discovered Technical Card JSON/`.npz` pair."""

    model_id: str
    technical_card_id: str
    json_path: Path
    npz_path: Path


def technical_cards_root(repository_root: Path) -> Path:
    return repository_root / TECHNICAL_CARDS_DIRECTORY


def list_technical_cards(repository_root: Path) -> tuple[TechnicalCardEntry, ...]:
    """List every `<model-id>/<technical-card-id>.json` pair under the official tree.

    Returns an empty tuple when no official Technical Cards exist yet; that is a valid repository
    state and must never be reported as an error (Section 28).
    """

    root = technical_cards_root(repository_root)
    if not root.is_dir():
        return ()
    entries: list[TechnicalCardEntry] = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for json_path in sorted(model_dir.glob("*.json")):
            entries.append(
                TechnicalCardEntry(
                    model_id=model_dir.name,
                    technical_card_id=json_path.stem,
                    json_path=json_path,
                    npz_path=json_path.with_suffix(".npz"),
                )
            )
    return tuple(entries)
