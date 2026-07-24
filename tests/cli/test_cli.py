from __future__ import annotations

from typer.testing import CliRunner

from torch_dae.cli.main import app


def test_root_and_group_help() -> None:
    runner = CliRunner()
    assert runner.invoke(app, ["--help"]).exit_code == 0
    assert runner.invoke(app, ["card", "--help"]).exit_code == 0
    assert runner.invoke(app, ["env", "--help"]).exit_code == 0


def test_card_list_empty(repo_root: object) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["card", "list"])
    assert result.exit_code == 0
    assert result.output == ""


def test_deferred_model_commands_fail_truthfully() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["env", "create", "synthetic"])
    assert result.exit_code == 4
    assert "model card not found" in result.output
    result = runner.invoke(app, ["model", "verify", "synthetic"])
    assert result.exit_code == 2
    assert "belongs to Phase 03+" in result.output


def test_env_info_absent() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["env", "info", "missing-card", "--json"])
    assert result.exit_code == 0
    assert '"specification_exists": false' in result.output
