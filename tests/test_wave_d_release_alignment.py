from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

RELEASE_VERSION = "0.2.0"
RELEASE_DATE = "2026-09-19"
CONCEPT_DOI = "10.5281/zenodo.21641390"


def test_release_metadata_is_aligned(repo_root: Path) -> None:
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text())
    assert pyproject["project"]["version"] == RELEASE_VERSION
    assert "Development Status :: 3 - Alpha" in pyproject["project"]["classifiers"]

    lock = (repo_root / "uv.lock").read_text()
    pattern = (
        r"\[\[package\]\]\n"
        r'name = "torch-deepaudioembedding"\n'
        rf'version = "{re.escape(RELEASE_VERSION)}"\n'
        r'source = \{ editable = "\." \}'
    )
    assert re.search(pattern, lock) is not None

    citation = yaml.safe_load((repo_root / "CITATION.cff").read_text())
    assert citation["version"] == RELEASE_VERSION
    assert citation["date-released"] == RELEASE_DATE
    assert "doi" not in citation
    assert citation["identifiers"] == [
        {
            "type": "doi",
            "value": CONCEPT_DOI,
            "description": "Concept DOI representing all releases of torch-dae.",
        }
    ]


def test_docs_version_follows_pyproject(repo_root: Path) -> None:
    conf = (repo_root / "docs/conf.py").read_text()
    assert 'ROOT / "pyproject.toml"' in conf
    assert 'version = str(_project_metadata["version"])' in conf
    assert 'version = "0.1.0"' not in conf
    assert 'version = "0.2.0"' not in conf


def test_release_notes_and_changelog_are_frozen(repo_root: Path) -> None:
    changelog = (repo_root / "CHANGELOG.md").read_text()
    assert f"## {RELEASE_VERSION} - {RELEASE_DATE}" in changelog
    assert "### Accepted PANNs integrations" in changelog
    assert "### Profiling v1 and Technical Cards" in changelog
    assert "### Documentation and public usability" in changelog

    notes = (repo_root / "docs/releases/0.2.0.md").read_text()
    for value in (
        "Three accepted PANNs / AudioSet models",
        "Profiling v1 and canonical Technical Cards",
        "Nine reviewed Technical Cards",
        "pip install torch-deepaudioembedding==0.2.0",
    ):
        assert value in notes


def test_release_guide_matches_production_configuration(repo_root: Path) -> None:
    guide = (repo_root / "docs/development/releasing.md").read_text()
    for value in (
        "GitHub Release",
        "publish.yml",
        "manual approval of environment pypi",
        "PyPI Trusted Publishing",
        "v0.2.0",
        "Read the Docs",
        "Zenodo",
        "10.5281/zenodo.21641390",
        "must never be invented",
    ):
        assert value in guide


def test_readme_uses_evergreen_citation_guidance(repo_root: Path) -> None:
    readme = (repo_root / "README.md").read_text()
    assert CONCEPT_DOI in readme
    assert "currently published `0.1.0`" not in readme
    assert "10.5281/zenodo.21641391" not in readme
