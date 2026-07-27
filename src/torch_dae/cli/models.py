"""Unavailable model CLI command placeholders."""

from __future__ import annotations

import typer

from torch_dae.core.errors import FeatureNotAvailableError

app = typer.Typer(no_args_is_help=True, help="Model commands.")


def _deferred(card_id: str, operation: str) -> None:
    exc = FeatureNotAvailableError(
        f"model {operation} for {card_id!r} is not available in the control-plane CLI"
    )
    typer.echo(str(exc), err=True)
    raise typer.Exit(2) from exc


@app.command("inspect")
def inspect(card_id: str) -> None:
    _deferred(card_id, "inspect")


@app.command("verify")
def verify(card_id: str) -> None:
    _deferred(card_id, "verify")
