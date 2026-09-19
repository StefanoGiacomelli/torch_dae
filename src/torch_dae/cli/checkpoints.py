"""Checkpoint CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from torch_dae.core.checkpoint import (
    CheckpointAcquisitionPolicy,
    CheckpointManager,
    CheckpointSpec,
)
from torch_dae.core.errors import (
    CheckpointAcquisitionError,
    CheckpointHashMismatchError,
    CheckpointNotFoundError,
    OfflineResourceUnavailableError,
    TorchDaeError,
)
from torch_dae.environment.manager import discover_repository_root
from torch_dae.environment.policy import ExecutionPolicy

app = typer.Typer(
    no_args_is_help=True,
    help="Resolve, acquire, inspect, and remove managed checkpoint payloads.",
)


def _manager(offline: bool = False) -> CheckpointManager:
    return CheckpointManager(discover_repository_root(), policy=ExecutionPolicy(offline=offline))


def _exit_for_error(exc: TorchDaeError) -> None:
    typer.echo(str(exc), err=True)
    if isinstance(exc, (OfflineResourceUnavailableError, CheckpointNotFoundError)):
        raise typer.Exit(3) from exc
    if isinstance(exc, (CheckpointAcquisitionError, CheckpointHashMismatchError)):
        raise typer.Exit(4) from exc
    raise typer.Exit(2) from exc


def _load_spec(path: Path) -> CheckpointSpec:
    try:
        return CheckpointSpec.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CheckpointAcquisitionError(f"invalid checkpoint specification: {exc}") from exc


@app.command("ensure")
def ensure(
    card_id: Annotated[str, typer.Argument(help="Accepted Model Card id.")],
    offline: bool = typer.Option(False, "--offline", help="Forbid network access."),
    json_output: bool = typer.Option(False, "--json", help="Emit structured JSON."),
) -> None:
    """Ensure the checkpoint referenced by an accepted Model Card is present and verified."""

    try:
        resolved = _manager(offline).ensure(card_id)
        if json_output:
            typer.echo(resolved.model_dump_json(indent=2))
        else:
            typer.echo(f"checkpoint ready: {resolved.checkpoint_id} {resolved.sha256}")
    except TorchDaeError as exc:
        _exit_for_error(exc)


@app.command("info")
def info(
    card_id: Annotated[str, typer.Argument(help="Accepted Model Card id.")],
    json_output: bool = typer.Option(False, "--json", help="Emit structured JSON."),
) -> None:
    """Inspect checkpoint authority and cache state for an accepted Model Card."""

    import json

    try:
        data = _manager().info(card_id)
        if json_output:
            typer.echo(json.dumps(data, indent=2, sort_keys=True))
        else:
            typer.echo(f"{data['checkpoint_id']}: {data['source_type']}")
    except TorchDaeError as exc:
        _exit_for_error(exc)


@app.command("resolve")
def resolve_spec(
    spec_path: Annotated[
        Path,
        typer.Option(
            "--spec",
            exists=True,
            dir_okay=False,
            help="Path to a CheckpointSpec JSON file.",
        ),
    ],
    offline: bool = typer.Option(False, "--offline", help="Forbid network access."),
    json_output: bool = typer.Option(False, "--json", help="Emit structured JSON."),
) -> None:
    """Resolve authoritative metadata for an explicit spec without acquiring payload bytes."""

    try:
        resolved = _manager(offline).resolve_checkpoint_authority(_load_spec(spec_path))
        if json_output:
            typer.echo(resolved.model_dump_json(indent=2))
        else:
            typer.echo(
                f"authority resolved: {resolved.provider}:{resolved.record_id}:"
                f"{resolved.resolved_filename}"
            )
    except TorchDaeError as exc:
        _exit_for_error(exc)


@app.command("ensure-spec")
def ensure_spec(
    spec_path: Annotated[
        Path,
        typer.Option(
            "--spec",
            exists=True,
            dir_okay=False,
            help="Path to a CheckpointSpec JSON file.",
        ),
    ],
    offline: bool = typer.Option(False, "--offline", help="Forbid network access."),
    maximum_bytes: int | None = typer.Option(
        None,
        "--maximum-bytes",
        min=1,
        help="Optional upper bound on acquired checkpoint bytes.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit structured JSON."),
) -> None:
    """Acquire an explicit spec through the canonical checkpoint manager."""

    try:
        spec = _load_spec(spec_path)
        policy = CheckpointAcquisitionPolicy(
            allow_network=not offline,
            allow_authentication=True,
            maximum_bytes=maximum_bytes,
            require_authority=spec.authority is not None,
            require_exact_size=spec.authority is not None,
            require_published_checksums=spec.authority is not None,
            require_observed_sha256=spec.authority is not None,
        )
        resolved = _manager(offline).ensure_checkpoint(spec, acquisition_policy=policy)
        if json_output:
            typer.echo(resolved.model_dump_json(indent=2))
        else:
            typer.echo(f"checkpoint ready: {resolved.checkpoint_id} {resolved.sha256}")
    except TorchDaeError as exc:
        _exit_for_error(exc)


@app.command("info-spec")
def info_spec(
    spec_path: Annotated[
        Path,
        typer.Option(
            "--spec",
            exists=True,
            dir_okay=False,
            help="Path to a CheckpointSpec JSON file.",
        ),
    ],
    json_output: bool = typer.Option(False, "--json", help="Emit structured JSON."),
) -> None:
    """Inspect an explicit spec and managed cache without network access."""

    import json

    try:
        data = _manager(True).info_checkpoint(_load_spec(spec_path))
        if json_output:
            typer.echo(json.dumps(data, indent=2, sort_keys=True))
        else:
            typer.echo(f"{data['checkpoint_id']}: {data['source_type']}")
    except TorchDaeError as exc:
        _exit_for_error(exc)


@app.command("remove")
def remove(
    card_id: Annotated[str, typer.Argument(help="Accepted Model Card id.")],
    json_output: bool = typer.Option(False, "--json", help="Emit structured JSON."),
) -> None:
    """Remove the managed checkpoint cache entry associated with a Model Card."""

    try:
        _manager().remove(card_id)
        if json_output:
            import json

            typer.echo(json.dumps({"card_id": card_id, "removed": True}, indent=2, sort_keys=True))
        else:
            typer.echo(f"removed: {card_id}")
    except TorchDaeError as exc:
        _exit_for_error(exc)
