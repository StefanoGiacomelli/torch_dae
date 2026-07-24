"""Execution policies for materialization and acquisition operations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionPolicy:
    """Network and subprocess policy for Phase 01 operations."""

    offline: bool = False
    allow_python_downloads: bool = True
    command_timeout_seconds: float = 300.0
    download_timeout_seconds: float = 300.0

    def uv_flags(self) -> tuple[str, ...]:
        """Return policy flags forwarded to `uv` commands."""

        flags: list[str] = []
        if self.offline:
            flags.append("--offline")
        if self.offline or not self.allow_python_downloads:
            flags.append("--no-python-downloads")
        return tuple(flags)
