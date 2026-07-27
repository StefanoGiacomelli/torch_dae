"""Stable environment specification, lifecycle, and verification interfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch_dae.environment.specification import (
    EnvironmentSourcesManifest,
    EnvironmentSpecification,
    EnvironmentVerificationSpec,
    OfficialPackageSource,
    PinnedGitSource,
    PlatformSpecification,
    PythonSpecification,
    SourceInstallationType,
    VendoredAdaptationSource,
)
from torch_dae.environment.verification import (
    TensorDimension,
    TensorObservation,
    VerificationCheck,
    VerificationReport,
)

if TYPE_CHECKING:
    from torch_dae.environment.manager import (
        EnvironmentInfo,
        EnvironmentManager,
        EnvironmentVerification,
        InstalledSource,
        ResolvedEnvironment,
    )

__all__ = [
    "EnvironmentInfo",
    "EnvironmentManager",
    "EnvironmentSourcesManifest",
    "EnvironmentSpecification",
    "EnvironmentVerification",
    "EnvironmentVerificationSpec",
    "InstalledSource",
    "OfficialPackageSource",
    "PinnedGitSource",
    "PlatformSpecification",
    "PythonSpecification",
    "ResolvedEnvironment",
    "SourceInstallationType",
    "TensorDimension",
    "TensorObservation",
    "VendoredAdaptationSource",
    "VerificationCheck",
    "VerificationReport",
]


def __getattr__(name: str) -> object:
    """Lazily expose the environment manager to avoid schema import cycles."""

    if name in {
        "EnvironmentInfo",
        "EnvironmentManager",
        "EnvironmentVerification",
        "InstalledSource",
        "ResolvedEnvironment",
    }:
        from torch_dae.environment import manager

        return getattr(manager, name)
    raise AttributeError(name)
