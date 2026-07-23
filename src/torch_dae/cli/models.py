"""Future model CLI commands."""

from __future__ import annotations

import typer

from torch_dae.core.errors import NotImplementedInPhaseError

app = typer.Typer(no_args_is_help=True, help="Model commands.")


def _deferred(card_id: str, operation: str) -> None:
    exc = NotImplementedInPhaseError(f"model {operation} for {card_id!r} belongs to Phase 03+")
    typer.echo(str(exc), err=True)
    raise typer.Exit(2) from exc


@app.command("inspect")
def inspect(card_id: str) -> None:
    _deferred(card_id, "inspect")


@app.command("verify")
def verify(card_id: str) -> None:
    _deferred(card_id, "verify")
