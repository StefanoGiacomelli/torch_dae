from __future__ import annotations

from click import unstyle
from typer.testing import CliRunner

from torch_dae import __version__
from torch_dae.cli.main import app


def test_root_and_group_help() -> None:
    runner = CliRunner()
    root = runner.invoke(app, ["--help"])
    assert root.exit_code == 0
    root_output = unstyle(root.output)
    assert "--version" in root_output
    assert "runtime verification" in root_output
    assert runner.invoke(app, ["card", "--help"]).exit_code == 0
    assert runner.invoke(app, ["env", "--help"]).exit_code == 0
    assert runner.invoke(app, ["checkpoint", "--help"]).exit_code == 0
    assert runner.invoke(app, ["model", "--help"]).exit_code == 0
    assert runner.invoke(app, ["technical-card", "--help"]).exit_code == 0


def test_root_version_matches_installed_distribution() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output == f"torch-dae {__version__}\n"


def test_card_list_reflects_committed_cards(repo_root: object) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["card", "list"])
    assert result.exit_code == 0
    listed = [line for line in result.output.splitlines() if line]
    assert listed == [
        "panns-cnn14-16k-map-0438",
        "panns-resnet38-map-0434",
        "panns-wavegram-logmel-cnn14-map-0439",
    ]
    assert listed == sorted(listed)


def test_model_verification_requires_an_explicit_target() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["env", "create", "synthetic"])
    assert result.exit_code == 4
    assert "model card not found" in result.output
    result = runner.invoke(app, ["model", "verify", "synthetic"])
    assert result.exit_code == 2
    assert "--target" in unstyle(result.output)
    help_result = runner.invoke(app, ["model", "verify", "--help"])
    assert help_result.exit_code == 0
    help_output = unstyle(help_result.output)
    assert "--target" in help_output
    assert "--offline" in help_output


def test_env_info_absent() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["env", "info", "missing-card", "--json"])
    assert result.exit_code == 0
    assert '"specification_exists": false' in result.output
