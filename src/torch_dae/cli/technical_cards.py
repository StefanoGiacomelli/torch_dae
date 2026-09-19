"""`torch-dae technical-card` commands (Section 29)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, help="Technical Card commands.")


@app.command("validate")
def validate(
    card: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Validate one candidate or official Technical Card JSON file."""

    from torch_dae.onboarding.handoff import discover_repository_root
    from torch_dae.profiling.validation import validate_technical_card_file

    result = validate_technical_card_file(card, repository_root=discover_repository_root())
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "technical_card_id": result.technical_card_id,
                    "valid": result.valid,
                    "errors": result.errors,
                },
                indent=2,
            )
        )
    else:
        typer.echo(f"{'valid' if result.valid else 'invalid'}: {result.technical_card_id}")
        for error in result.errors:
            typer.echo(f"  - {error}")
    if not result.valid:
        raise typer.Exit(1)


@app.command("inspect")
def inspect(
    card: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Print the parsed content of one Technical Card JSON file."""

    from torch_dae.profiling.contracts import TechnicalCard

    parsed = TechnicalCard.model_validate_json(card.read_text())
    typer.echo(parsed.model_dump_json(indent=2))


@app.command("list")
def list_cards(json_output: Annotated[bool, typer.Option("--json")] = False) -> None:
    """List every Technical Card under the canonical `technical_cards/` repository tree."""

    from torch_dae.onboarding.handoff import discover_repository_root
    from torch_dae.profiling.storage import list_technical_cards

    entries = list_technical_cards(discover_repository_root())
    if json_output:
        typer.echo(
            json.dumps(
                [
                    {
                        "model_id": entry.model_id,
                        "technical_card_id": entry.technical_card_id,
                        "json_path": str(entry.json_path),
                    }
                    for entry in entries
                ],
                indent=2,
            )
        )
        return
    if not entries:
        typer.echo("no official Technical Cards exist yet")
        return
    for entry in entries:
        typer.echo(f"{entry.model_id}/{entry.technical_card_id}")
