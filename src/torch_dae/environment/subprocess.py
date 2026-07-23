"""Managed process result contract for model environments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ManagedProcessResult:
    """Result of a future command executed inside a model environment."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
