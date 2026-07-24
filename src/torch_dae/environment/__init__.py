"""Environment management interfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch_dae.environment.specification import EnvironmentSpecification

if TYPE_CHECKING:
    from torch_dae.environment.manager import EnvironmentManager

__all__ = ["EnvironmentManager", "EnvironmentSpecification"]


def __getattr__(name: str) -> object:
    """Lazily expose the environment manager to avoid schema import cycles."""

    if name == "EnvironmentManager":
        from torch_dae.environment.manager import EnvironmentManager

        return EnvironmentManager
    raise AttributeError(name)
