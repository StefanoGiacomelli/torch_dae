"""Checkpoint CLI commands."""

from __future__ import annotations

import typer

from torch_dae.core.checkpoint import CheckpointManager
from torch_dae.core.errors import (
    CheckpointAcquisitionError,
    CheckpointHashMismatchError,
    CheckpointNotFoundError,
    OfflineResourceUnavailableError,
    TorchDaeError,
)
from torch_dae.environment.manager import discover_repository_root
from torch_dae.environment.policy import ExecutionPolicy

app = typer.Typer(no_args_is_help=True, help="Checkpoint commands.")


def _manager(offline: bool = False) -> CheckpointManager:
    return CheckpointManager(discover_repository_root(), policy=ExecutionPolicy(offline=offline))


def _exit_for_error(exc: TorchDaeError) -> None:
    typer.echo(str(exc), err=True)
    if isinstance(exc, (OfflineResourceUnavailableError, CheckpointNotFoundError)):
        raise typer.Exit(3) from exc
    if isinstance(exc, (CheckpointAcquisitionError, CheckpointHashMismatchError)):
        raise typer.Exit(4) from exc
    raise typer.Exit(2) from exc


@app.command("ensure")
def ensure(
    card_id: str,
    offline: bool = typer.Option(False, "--offline"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    try:
        resolved = _manager(offline).ensure(card_id)
        if json_output:
            typer.echo(resolved.model_dump_json(indent=2))
        else:
            typer.echo(f"checkpoint ready: {resolved.checkpoint_id} {resolved.sha256}")
    except TorchDaeError as exc:
        _exit_for_error(exc)


@app.command("info")
def info(card_id: str, json_output: bool = typer.Option(False, "--json")) -> None:
    import json

    try:
        data = _manager().info(card_id)
        if json_output:
            typer.echo(json.dumps(data, indent=2, sort_keys=True))
        else:
            typer.echo(f"{data['checkpoint_id']}: {data['source_type']}")
    except TorchDaeError as exc:
        _exit_for_error(exc)


@app.command("remove")
def remove(card_id: str, json_output: bool = typer.Option(False, "--json")) -> None:
    try:
        _manager().remove(card_id)
        if json_output:
            import json

            typer.echo(json.dumps({"card_id": card_id, "removed": True}, indent=2, sort_keys=True))
        else:
            typer.echo(f"removed: {card_id}")
    except TorchDaeError as exc:
        _exit_for_error(exc)
