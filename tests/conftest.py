from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def valid_fixture_dir(repo_root: Path) -> Path:
    return repo_root / "tests/fixtures/valid"


@pytest.fixture(scope="session")
def invalid_fixture_dir(repo_root: Path) -> Path:
    return repo_root / "tests/fixtures/invalid"
