"""Root Typer CLI."""

from __future__ import annotations

import typer

from torch_dae.cli.cards import app as card_app
from torch_dae.cli.checkpoints import app as checkpoint_app
from torch_dae.cli.environment import app as env_app
from torch_dae.cli.models import app as model_app

app = typer.Typer(no_args_is_help=True, help="torch-dae control-plane CLI.")
app.add_typer(card_app, name="card")
app.add_typer(env_app, name="env")
app.add_typer(checkpoint_app, name="checkpoint")
app.add_typer(model_app, name="model")


if __name__ == "__main__":
    app()
