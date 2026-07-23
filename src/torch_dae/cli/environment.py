"""Environment CLI commands."""

from __future__ import annotations

import typer

from torch_dae.core.errors import NotImplementedInPhaseError
from torch_dae.environment.manager import EnvironmentManager

app = typer.Typer(no_args_is_help=True, help="Environment commands.")


def _manager() -> EnvironmentManager:
    return EnvironmentManager.from_repository_root()


def _fail_deferred(exc: NotImplementedInPhaseError) -> None:
    typer.echo(str(exc), err=True)
    raise typer.Exit(2) from exc


@app.command("create")
def create(card_id: str) -> None:
    try:
        _manager().create(card_id)
    except NotImplementedInPhaseError as exc:
        _fail_deferred(exc)


@app.command("ensure")
def ensure(card_id: str) -> None:
    try:
        _manager().ensure(card_id)
    except NotImplementedInPhaseError as exc:
        _fail_deferred(exc)


@app.command("verify")
def verify(card_id: str) -> None:
    try:
        _manager().verify(card_id)
    except NotImplementedInPhaseError as exc:
        _fail_deferred(exc)


@app.command("remove")
def remove(card_id: str) -> None:
    try:
        _manager().remove(card_id)
    except NotImplementedInPhaseError as exc:
        _fail_deferred(exc)


@app.command("info")
def info(card_id: str) -> None:
    typer.echo(_manager().info_json(card_id))


@app.command("run", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(ctx: typer.Context, card_id: str) -> None:
    command = list(ctx.args)
    try:
        _manager().run(card_id, command)
    except NotImplementedInPhaseError as exc:
        _fail_deferred(exc)
