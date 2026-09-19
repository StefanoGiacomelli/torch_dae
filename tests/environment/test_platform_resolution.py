from __future__ import annotations

from pathlib import Path

import pytest

from torch_dae.core.errors import EnvironmentMaterializationError
from torch_dae.environment.manager import EnvironmentManager

ROOT = Path(__file__).resolve().parents[2]

PANN_ENVIRONMENT = "panns-cnn14-16k-map-0438"


def test_structural_environment_resolution_is_host_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "torch_dae.environment.manager.canonical_platform_tag",
        lambda: "linux-x86_64",
    )

    manager = EnvironmentManager(ROOT)

    definition = manager.resolve_environment(PANN_ENVIRONMENT)

    assert definition.environment_id == PANN_ENVIRONMENT
    assert definition.platform == "linux-x86_64"


def test_explicit_resolution_can_enforce_platform_constraints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "torch_dae.environment.manager.canonical_platform_tag",
        lambda: "linux-x86_64",
    )

    manager = EnvironmentManager(ROOT)

    with pytest.raises(
        EnvironmentMaterializationError,
        match="platform linux-x86_64 is outside declared environment constraints",
    ):
        manager.resolve_environment(
            PANN_ENVIRONMENT,
            enforce_platform=True,
        )


def test_materialization_still_enforces_platform_constraints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "torch_dae.environment.manager.canonical_platform_tag",
        lambda: "linux-x86_64",
    )

    manager = EnvironmentManager(ROOT)

    with pytest.raises(
        EnvironmentMaterializationError,
        match="platform linux-x86_64 is outside declared environment constraints",
    ):
        manager.materialize_environment(PANN_ENVIRONMENT)
