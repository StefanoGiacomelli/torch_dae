"""Checkpoint CLI commands."""

from __future__ import annotations

import typer

from torch_dae.core.checkpoint import CheckpointManager
from torch_dae.core.errors import NotImplementedInPhaseError
from torch_dae.environment.manager import discover_repository_root

app = typer.Typer(no_args_is_help=True, help="Checkpoint commands.")


def _manager() -> CheckpointManager:
    return CheckpointManager(discover_repository_root())


@app.command("ensure")
def ensure(card_id: str) -> None:
    try:
        _manager().ensure(card_id)
    except NotImplementedInPhaseError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


@app.command("info")
def info(card_id: str) -> None:
    import json

    typer.echo(json.dumps(_manager().info(card_id), indent=2, sort_keys=True))


@app.command("remove")
def remove(card_id: str) -> None:
    try:
        _manager().remove(card_id)
    except NotImplementedInPhaseError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
