from __future__ import annotations

from typer.testing import CliRunner

from tests.environment.test_phase01_materialization import write_phase01_repo
from torch_dae.cli import cards as card_cli
from torch_dae.cli.main import app


def test_card_cli_with_synthetic_card(tmp_path, repo_root, valid_fixture_dir, monkeypatch) -> None:
    card_id = write_phase01_repo(tmp_path, repo_root, valid_fixture_dir)
    monkeypatch.setattr(card_cli, "discover_repository_root", lambda: tmp_path)
    runner = CliRunner()
    assert card_id in runner.invoke(app, ["card", "list"]).output
    assert f'"card_id": "{card_id}"' in runner.invoke(app, ["card", "show", card_id]).output
    result = runner.invoke(app, ["card", "validate", card_id])
    assert result.exit_code == 0
    assert "valid:" in result.output
