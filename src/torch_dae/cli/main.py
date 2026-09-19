"""Root Typer CLI for repository-backed torch-dae workflows."""

from __future__ import annotations

from typing import Annotated

import typer

from torch_dae import __version__
from torch_dae.cli.cards import app as card_app
from torch_dae.cli.checkpoints import app as checkpoint_app
from torch_dae.cli.environment import app as env_app
from torch_dae.cli.models import app as model_app
from torch_dae.cli.technical_cards import app as technical_card_app

app = typer.Typer(
    no_args_is_help=True,
    help=(
        "Reproducible audio-model integration, runtime verification, checkpoint management, "
        "and profiling."
    ),
)
app.add_typer(card_app, name="card")
app.add_typer(env_app, name="env")
app.add_typer(checkpoint_app, name="checkpoint")
app.add_typer(model_app, name="model")
app.add_typer(technical_card_app, name="technical-card")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"torch-dae {__version__}")
        raise typer.Exit


@app.callback()
def root(
    version_requested: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the installed torch-dae version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Expose the top-level control-plane command group."""

    del version_requested


if __name__ == "__main__":
    app()
