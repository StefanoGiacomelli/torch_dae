"""Typed no-shell subprocess execution."""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from torch_dae.core.errors import ExternalCommandError, GitUnavailableError, UvUnavailableError
from torch_dae.environment.runtime import RuntimeReportSink, sanitize_text


@dataclass(frozen=True)
class ManagedProcessResult:
    """Result of a command executed by the control plane."""

    command: tuple[str, ...]
    cwd: str | None
    started_at: str
    ended_at: str
    duration_seconds: float
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class CommandLog:
    """Serializable command log entry suitable for ignored runtime reports."""

    result: ManagedProcessResult
    environment_overrides: Mapping[str, str] = field(default_factory=dict)


class CommandCompleted(Protocol):
    """Minimum completed-process shape returned by fake runners."""

    returncode: int
    stdout: str
    stderr: str


class CommandRunner(Protocol):
    """Protocol for deterministic fake command runners."""

    def run(
        self,
        command: tuple[str, ...],
        *,
        cwd: Path | None,
        env: Mapping[str, str],
        timeout: float | None,
    ) -> CommandCompleted:
        """Run a command and return an object with returncode/stdout/stderr."""


class CommandExecutor:
    """Run external commands as argument vectors with deterministic test injection."""

    def __init__(
        self,
        runner: CommandRunner | None = None,
        *,
        report_sink: RuntimeReportSink | None = None,
    ) -> None:
        self._runner = runner
        self._report_sink = report_sink

    def with_report_sink(self, report_sink: RuntimeReportSink) -> CommandExecutor:
        """Return an executor sharing the same runner with reporting enabled."""

        return CommandExecutor(self._runner, report_sink=report_sink)

    def run(
        self,
        command: Sequence[str],
        *,
        operation: str | None = None,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        env_remove: Sequence[str] = (),
        timeout: float | None = None,
        check: bool = False,
    ) -> ManagedProcessResult:
        """Execute `command` without a shell and return captured UTF-8 output."""

        if not command:
            raise ValueError("command must not be empty")
        argv = tuple(str(part) for part in command)
        started = datetime.now(UTC)
        started_monotonic = time.monotonic()
        full_env = os.environ.copy()
        for name in env_remove:
            full_env.pop(name, None)
        if env:
            full_env.update(env)
        if self._runner is None:
            try:
                completed = subprocess.run(
                    argv,
                    cwd=cwd,
                    env=full_env,
                    timeout=timeout,
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    shell=False,
                )
            except FileNotFoundError as exc:
                ended = datetime.now(UTC)
                self._record_exception(
                    operation or default_operation(argv),
                    argv,
                    cwd,
                    started,
                    started_monotonic,
                    ended,
                    "unavailable",
                    missing_executable_error(argv[0]).args[0],
                )
                raise missing_executable_error(argv[0]) from exc
            except subprocess.TimeoutExpired as exc:
                stdout = decode_timeout_output(exc.stdout)
                stderr = decode_timeout_output(exc.stderr)
                ended = datetime.now(UTC)
                self._record_exception(
                    operation or default_operation(argv),
                    argv,
                    cwd,
                    started,
                    started_monotonic,
                    ended,
                    "timeout",
                    stderr or stdout,
                )
                raise ExternalCommandError(
                    f"command timed out after {timeout} seconds: {argv[0]}: "
                    f"{redact_secrets(stderr or stdout)}"
                ) from exc
            except OSError as exc:
                ended = datetime.now(UTC)
                self._record_exception(
                    operation or default_operation(argv),
                    argv,
                    cwd,
                    started,
                    started_monotonic,
                    ended,
                    "unavailable",
                    str(exc),
                )
                raise ExternalCommandError(f"could not execute {argv[0]}: {exc}") from exc
            returncode = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        else:
            fake_completed = self._runner.run(argv, cwd=cwd, env=full_env, timeout=timeout)
            returncode = int(fake_completed.returncode)
            stdout = str(fake_completed.stdout)
            stderr = str(fake_completed.stderr)
        ended = datetime.now(UTC)
        result = ManagedProcessResult(
            command=argv,
            cwd=str(cwd) if cwd is not None else None,
            started_at=started.isoformat(),
            ended_at=ended.isoformat(),
            duration_seconds=time.monotonic() - started_monotonic,
            returncode=returncode,
            stdout=redact_secrets(stdout),
            stderr=redact_secrets(stderr),
        )
        self._record_result(operation or default_operation(argv), result)
        if check and result.returncode != 0:
            details = (result.stderr or result.stdout).strip()
            suffix = f": {details}" if details else ""
            raise ExternalCommandError(
                f"command failed with exit code {result.returncode}: {argv[0]}{suffix}",
                returncode=result.returncode,
            )
        return result

    def _record_result(self, operation: str, result: ManagedProcessResult) -> None:
        if self._report_sink is None:
            return
        self._report_sink.record_command(
            operation=operation,
            arguments=result.command,
            working_directory=result.cwd,
            started_at=result.started_at,
            completed_at=result.ended_at,
            duration_seconds=result.duration_seconds,
            return_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            status="success" if result.returncode == 0 else "failed",
        )

    def _record_exception(
        self,
        operation: str,
        argv: tuple[str, ...],
        cwd: Path | None,
        started: datetime,
        started_monotonic: float,
        ended: datetime,
        status: str,
        message: str,
    ) -> None:
        if self._report_sink is None:
            return
        self._report_sink.record_command(
            operation=operation,
            arguments=argv,
            working_directory=str(cwd) if cwd is not None else None,
            started_at=started.isoformat(),
            completed_at=ended.isoformat(),
            duration_seconds=time.monotonic() - started_monotonic,
            return_code=None,
            stdout="",
            stderr=sanitize_text(message),
            status=status,  # type: ignore[arg-type]
        )


def redact_secrets(value: str) -> str:
    """Redact obvious token-bearing lines from command output."""

    return sanitize_text(value)


def default_operation(command: tuple[str, ...]) -> str:
    """Return a generic operation name for unlabelled commands."""

    return Path(command[0]).name if command else "command"


def missing_executable_error(command: str) -> ExternalCommandError:
    """Return the precise typed error for a missing executable."""

    if command == "uv":
        return UvUnavailableError("uv executable is unavailable")
    if command == "git":
        return GitUnavailableError("git executable is unavailable")
    return ExternalCommandError(f"executable is unavailable: {command}")


def decode_timeout_output(value: str | bytes | None) -> str:
    """Decode timeout output values produced by subprocess."""

    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
