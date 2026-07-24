from __future__ import annotations

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from torch_dae.cli import checkpoints as checkpoint_cli
from torch_dae.cli import environment as env_cli
from torch_dae.cli.main import app
from torch_dae.core.checkpoint import ResolvedCheckpoint
from torch_dae.core.errors import (
    CheckpointAcquisitionError,
    CheckpointHashMismatchError,
    EnvironmentAlreadyExistsError,
    GitUnavailableError,
    OfflineResourceUnavailableError,
    TorchDaeError,
    UvUnavailableError,
)
from torch_dae.environment.manager import (
    EnvironmentInfo,
    EnvironmentVerification,
    ResolvedEnvironment,
)
from torch_dae.environment.subprocess import ManagedProcessResult


class FakeEnvironmentManager:
    def __init__(self) -> None:
        self.resolved = ResolvedEnvironment(
            environment_id="card",
            model_card_id="card",
            root=Path("/tmp/env"),
            python_executable=Path("/tmp/env/bin/python"),
            fingerprint="a" * 64,
            python_version="3.12.0",
            platform="synthetic",
            installed_packages={"torch-dae": "0.1.0"},
            installed_sources=(),
            valid=True,
        )

    def create(self, card_id: str) -> ResolvedEnvironment:
        return self.resolved

    def ensure(self, card_id: str) -> ResolvedEnvironment:
        return self.resolved

    def verify(self, card_id: str) -> EnvironmentVerification:
        return EnvironmentVerification(card_id, True, "valid", "valid")

    def remove(self, card_id: str) -> None:
        return None

    def info(self, card_id: str) -> EnvironmentInfo:
        return EnvironmentInfo(
            card_id,
            Path("/repo/environments/card/environment.json"),
            True,
            "a" * 64,
            Path("/tmp/env"),
            True,
            "valid",
        )

    def info_json(self, card_id: str) -> str:
        return '{"status": "valid", "model_card_id": "card"}'

    def run(self, card_id: str, command: list[str]) -> ManagedProcessResult:
        return ManagedProcessResult(
            command=tuple(command),
            cwd="/repo",
            started_at="2026-01-01T00:00:00+00:00",
            ended_at="2026-01-01T00:00:01+00:00",
            duration_seconds=1.0,
            returncode=0,
            stdout="ran\n",
            stderr="",
        )


class FakeCheckpointManager:
    def ensure(self, card_id: str) -> ResolvedCheckpoint:
        return ResolvedCheckpoint(
            checkpoint_id="checkpoint",
            sha256="b" * 64,
            path=Path("/tmp/checkpoint.bin"),
            immutable=True,
        )

    def info(self, card_id: str) -> dict[str, object]:
        return {"checkpoint_id": "checkpoint", "source_type": "local_path", "cached": []}

    def remove(self, card_id: str) -> None:
        return None


class ErrorCheckpointManager(FakeCheckpointManager):
    def __init__(self, error: TorchDaeError) -> None:
        self.error = error

    def ensure(self, card_id: str) -> ResolvedCheckpoint:
        raise self.error


class ErrorEnvironmentManager(FakeEnvironmentManager):
    def __init__(self, error: TorchDaeError) -> None:
        super().__init__()
        self.error = error

    def ensure(self, card_id: str) -> ResolvedEnvironment:
        raise self.error


def test_phase01_environment_cli_success_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeEnvironmentManager()
    monkeypatch.setattr(env_cli, "_manager", lambda offline=False, no_python_downloads=False: fake)
    runner = CliRunner()
    assert runner.invoke(app, ["env", "create", "card"]).exit_code == 0
    assert '"fingerprint"' in runner.invoke(app, ["env", "ensure", "card", "--json"]).output
    assert '"passed": true' in runner.invoke(app, ["env", "verify", "card", "--json"]).output
    assert "card: valid" in runner.invoke(app, ["env", "info", "card"]).output
    assert '"status": "valid"' in runner.invoke(app, ["env", "info", "card", "--json"]).output
    assert '"removed": true' in runner.invoke(app, ["env", "remove", "card", "--json"]).output
    run = runner.invoke(app, ["env", "run", "card", "--", "python", "-V"])
    assert run.exit_code == 0
    assert run.output == "ran\n"


def test_phase01_checkpoint_cli_success_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeCheckpointManager()
    monkeypatch.setattr(checkpoint_cli, "_manager", lambda offline=False: fake)
    runner = CliRunner()
    assert '"sha256"' in runner.invoke(app, ["checkpoint", "ensure", "card", "--json"]).output
    assert "checkpoint: local_path" in runner.invoke(app, ["checkpoint", "info", "card"]).output
    assert '"cached": []' in runner.invoke(app, ["checkpoint", "info", "card", "--json"]).output
    assert (
        '"removed": true' in runner.invoke(app, ["checkpoint", "remove", "card", "--json"]).output
    )


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (OfflineResourceUnavailableError("offline"), 3),
        (EnvironmentAlreadyExistsError("exists"), 4),
        (TorchDaeError("other"), 2),
    ],
)
def test_phase01_environment_cli_error_mapping(error: TorchDaeError, code: int) -> None:
    with pytest.raises(typer.Exit) as exc_info:
        env_cli._exit_for_error(error)
    assert exc_info.value.exit_code == code


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (OfflineResourceUnavailableError("offline"), 3),
        (CheckpointAcquisitionError("failed"), 4),
        (TorchDaeError("other"), 2),
    ],
)
def test_phase01_checkpoint_cli_error_mapping(error: TorchDaeError, code: int) -> None:
    with pytest.raises(typer.Exit) as exc_info:
        checkpoint_cli._exit_for_error(error)
    assert exc_info.value.exit_code == code


@pytest.mark.parametrize(
    "command",
    [
        ["checkpoint", "ensure", "missing-card"],
        ["checkpoint", "info", "missing-card"],
        ["checkpoint", "remove", "missing-card"],
    ],
)
def test_phase01_checkpoint_cli_missing_card_uses_clean_exit(command: list[str]) -> None:
    result = CliRunner().invoke(app, command)
    assert result.exit_code == 3
    assert "model card not found" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("label", "error", "exit_code", "expected"),
    [
        (
            "interrupted remote read",
            CheckpointAcquisitionError("checkpoint download stream failed: read failed"),
            4,
            "download stream failed",
        ),
        (
            "transport-open OSError",
            CheckpointAcquisitionError("checkpoint transport failed: network down"),
            4,
            "transport failed",
        ),
        (
            "hash mismatch",
            CheckpointHashMismatchError("expected checkpoint hash mismatch"),
            4,
            "hash mismatch",
        ),
        (
            "local copy failure",
            CheckpointAcquisitionError("checkpoint copy failed: copy denied"),
            4,
            "copy failed",
        ),
        (
            "metadata-write failure",
            CheckpointAcquisitionError("checkpoint metadata write failed: metadata denied"),
            4,
            "metadata write failed",
        ),
        (
            "offline remote cache miss",
            OfflineResourceUnavailableError("checkpoint cache miss for remote-checkpoint"),
            3,
            "cache miss",
        ),
    ],
)
def test_phase01_checkpoint_cli_expected_acquisition_failures_are_concise(
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    error: TorchDaeError,
    exit_code: int,
    expected: str,
) -> None:
    monkeypatch.setattr(
        checkpoint_cli,
        "_manager",
        lambda offline=False: ErrorCheckpointManager(error),
    )
    result = CliRunner().invoke(app, ["checkpoint", "ensure", "card"])
    assert result.exit_code == exit_code, label
    assert expected in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "command",
    [
        ["env", "info", "../bad"],
        ["env", "remove", "../bad"],
    ],
)
def test_phase01_environment_cli_invalid_id_uses_clean_exit(command: list[str]) -> None:
    result = CliRunner().invoke(app, command)
    assert result.exit_code == 4
    assert "invalid environment ID" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (UvUnavailableError("uv executable is unavailable"), "uv executable is unavailable"),
        (GitUnavailableError("git executable is unavailable"), "git executable is unavailable"),
    ],
)
def test_phase01_environment_cli_missing_tool_uses_clean_exit(
    monkeypatch: pytest.MonkeyPatch,
    error: TorchDaeError,
    expected: str,
) -> None:
    monkeypatch.setattr(
        env_cli,
        "_manager",
        lambda offline=False, no_python_downloads=False: ErrorEnvironmentManager(error),
    )
    result = CliRunner().invoke(app, ["env", "ensure", "card"])
    assert result.exit_code == 4
    assert expected in result.output
    assert "Traceback" not in result.output
