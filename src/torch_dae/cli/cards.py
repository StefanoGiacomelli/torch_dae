"""Model-card CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer

from torch_dae.cards.validation import validate_model_card_path
from torch_dae.core.registry import ModelCardRegistry
from torch_dae.environment.manager import discover_repository_root

app = typer.Typer(
    no_args_is_help=True,
    help="Inspect and validate accepted Model Cards in the repository registry.",
)


@app.command("list")
def list_cards() -> None:
    """List accepted Model Card identifiers in deterministic order."""

    registry = ModelCardRegistry(discover_repository_root())
    for card in registry.list_cards():
        typer.echo(card.card_id)


@app.command("show")
def show(card_id: str) -> None:
    """Print one accepted Model Card as normalized JSON."""

    registry = ModelCardRegistry(discover_repository_root())
    typer.echo(registry.get_card(card_id).model_dump_json(indent=2))


@app.command("validate")
def validate(card_id_or_path: str) -> None:
    """Validate a Model Card by accepted id or explicit JSON path."""

    root = discover_repository_root()
    candidate = Path(card_id_or_path)
    if candidate.exists():
        path = candidate
    else:
        registry = ModelCardRegistry(root)
        path = registry.get_card_path(card_id_or_path)
    validate_model_card_path(path, root / "schemas/model-card.schema.json")
    typer.echo(f"valid: {path}")
