from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import torch_dae.environment as environment_package
import torch_dae.environment.subprocess as subprocess_module
from torch_dae.core.errors import (
    ExternalCommandError,
    GitUnavailableError,
    UvUnavailableError,
)
from torch_dae.environment.manager import EnvironmentManager
from torch_dae.environment.runtime import RuntimeReportSink, result_to_verification
from torch_dae.environment.subprocess import CommandExecutor


class Completed:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class FailingRunner:
    def run(
        self,
        command: tuple[str, ...],
        *,
        cwd: Path | None,
        env: dict[str, str],
        timeout: float | None,
    ) -> Completed:
        return Completed(stderr="Authorization: Bearer secret", returncode=5)


class EnvCaptureRunner:
    def __init__(self) -> None:
        self.env: dict[str, str] = {}

    def run(
        self,
        command: tuple[str, ...],
        *,
        cwd: Path | None,
        env: dict[str, str],
        timeout: float | None,
    ) -> Completed:
        self.env = dict(env)
        return Completed()


def test_environment_package_lazy_export() -> None:
    assert environment_package.EnvironmentManager is EnvironmentManager
    missing = "missing"
    with pytest.raises(AttributeError):
        getattr(environment_package, missing)


def test_result_to_verification_without_process() -> None:
    record = result_to_verification("missing", "not found")
    assert record.status == "missing"
    assert record.returncode is None


def test_command_executor_failure_redacts_secrets() -> None:
    with pytest.raises(ExternalCommandError, match="redacted secret"):
        CommandExecutor(FailingRunner()).run(["tool"], check=True)


def test_command_executor_removes_host_python_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/host/path")
    monkeypatch.setenv("PYTHONHOME", "/host/home")
    runner = EnvCaptureRunner()
    CommandExecutor(runner).run(["tool"], env_remove=("PYTHONPATH", "PYTHONHOME"))
    assert "PYTHONPATH" not in runner.env
    assert "PYTHONHOME" not in runner.env


def test_command_executor_missing_uv_and_git_are_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError()

    monkeypatch.setattr(subprocess_module.subprocess, "run", missing)
    with pytest.raises(UvUnavailableError):
        CommandExecutor().run(["uv", "--version"])
    with pytest.raises(GitUnavailableError):
        CommandExecutor().run(["git", "--version"])


def test_command_executor_timeout_and_oserror_are_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout(*args: object, **kwargs: object) -> object:
        raise subprocess.TimeoutExpired(cmd=("tool",), timeout=1, stderr=b"token=secret")

    monkeypatch.setattr(subprocess_module.subprocess, "run", timeout)
    with pytest.raises(ExternalCommandError, match="redacted secret"):
        CommandExecutor().run(["tool"])

    def oserror(*args: object, **kwargs: object) -> object:
        raise OSError("boom")

    monkeypatch.setattr(subprocess_module.subprocess, "run", oserror)
    with pytest.raises(ExternalCommandError, match="could not execute"):
        CommandExecutor().run(["tool"])


def test_command_executor_report_sink_records_success_failure_and_exceptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = RuntimeReportSink(tmp_path / ".torch-dae", "reports", "commands")
    CommandExecutor(EnvCaptureRunner(), report_sink=sink).run(["tool"], operation="success")
    with pytest.raises(ExternalCommandError):
        CommandExecutor(FailingRunner(), report_sink=sink).run(
            ["tool", "token=secret"],
            operation="checked-failure",
            check=True,
        )

    def missing(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError()

    monkeypatch.setattr(subprocess_module.subprocess, "run", missing)
    with pytest.raises(UvUnavailableError):
        CommandExecutor(report_sink=sink).run(["uv"], operation="missing")

    def timeout(*args: object, **kwargs: object) -> object:
        raise subprocess.TimeoutExpired(cmd=("tool",), timeout=1, stderr=b"bearer secret")

    monkeypatch.setattr(subprocess_module.subprocess, "run", timeout)
    with pytest.raises(ExternalCommandError):
        CommandExecutor(report_sink=sink).run(["tool"], operation="timeout")

    def oserror(*args: object, **kwargs: object) -> object:
        raise OSError("token=secret")

    monkeypatch.setattr(subprocess_module.subprocess, "run", oserror)
    with pytest.raises(ExternalCommandError):
        CommandExecutor(report_sink=sink).run(["tool"], operation="oserror")

    reports = [json.loads((tmp_path / ".torch-dae" / ref).read_text()) for ref in sink.references]
    statuses = {report["operation"]: report["status"] for report in reports}
    assert statuses == {
        "success": "success",
        "checked-failure": "failed",
        "missing": "unavailable",
        "timeout": "timeout",
        "oserror": "unavailable",
    }
    payload = "\n".join(json.dumps(report) for report in reports)
    assert "token=secret" not in payload
    assert "bearer secret" not in payload.lower()
