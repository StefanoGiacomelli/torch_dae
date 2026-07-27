from __future__ import annotations

import re
from pathlib import Path

import yaml


def load_workflow(path: Path) -> dict[str, object]:
    return yaml.load(path.read_text(), Loader=yaml.BaseLoader)


def test_ci_python_and_artifact_contract(repo_root: Path) -> None:
    text = (repo_root / ".github/workflows/ci.yml").read_text()
    assert 'python-version: ["3.11", "3.12"]' in text
    assert "python-version: ${{ matrix.python-version }}" in text
    assert text.count("python-version: ${{ matrix.python-version }}") == 2
    assert "assert sys.version_info[:2] == expected" in text
    job_boundaries = {
        "quality": "  tests:",
        "coverage": "  documentation:",
    }
    for job, next_job in job_boundaries.items():
        block = text.split(f"  {job}:", 1)[1].split(next_job, 1)[0]
        assert "actions/setup-python@v7" in block
        assert "astral-sh/setup-uv@v9" in block
        assert block.count('python-version: "3.12"') == 2
    documentation = text.split("  documentation:", 1)[1].split("\n  distribution:", 1)[0]
    assert "sphinx-build -W --keep-going -b html docs docs/_build/html" in documentation
    assert "name: documentation-html" in documentation
    assert "retention-days: 7" in documentation
    distribution = text.split("  distribution:", 1)[1]
    assert "python -m build" in distribution
    assert "python -m twine check dist/*" in distribution
    assert "name: python-package-distributions" in distribution
    assert "retention-days: 7" in distribution
    coverage = text.split("  coverage:", 1)[1].split("\n  documentation:", 1)[0]
    assert "fetch-depth: 0" in coverage
    assert "use_oidc: true" in coverage
    assert "fail_ci_if_error: true" in coverage
    assert "CODECOV_TOKEN" not in coverage


def test_production_trigger_is_only_release_published(repo_root: Path) -> None:
    workflow = load_workflow(repo_root / ".github/workflows/publish.yml")
    assert workflow["on"] == {"release": {"types": ["published"]}}


def test_testpypi_trigger_is_only_manual(repo_root: Path) -> None:
    workflow = load_workflow(repo_root / ".github/workflows/test-publish.yml")
    assert workflow["on"] == {"workflow_dispatch": ""}


def test_production_build_once_and_artifact_reuse(repo_root: Path) -> None:
    text = (repo_root / ".github/workflows/publish.yml").read_text()
    assert text.count("uv run python -m build") == 1
    assert text.count("actions/upload-artifact@v5") == 1
    assert text.count("actions/download-artifact@v6") == 2
    assert text.count("name: python-package-distributions") == 3
    assert 'scripts/check_release_version.py "$RELEASE_TAG"' in text
    assert "environment:\n      name: pypi" in text
    assert "id-token: write" in text
    assert "https://pypi.org/p/torch-deepaudioembedding" in text
    assert "metadata.version('torch-deepaudioembedding')" in text
    assert "test.pypi.org" not in text
    assert "dist/*.whl dist/*.tar.gz" in text
    assert "gh release upload" in text
    assert "steps.project-version.outputs.value" in text
    assert 'test "$RELEASE_TAG" = "v$PROJECT_VERSION"' in text
    assert "needs.validate-and-build.outputs.project-version" in text


def test_testpypi_is_manual_oidc_and_has_distinct_endpoint(repo_root: Path) -> None:
    text = (repo_root / ".github/workflows/test-publish.yml").read_text()
    assert text.count("uv run python -m build") == 1
    assert "scripts/check_release_version.py --current" in text
    assert "environment:\n      name: testpypi" in text
    assert "id-token: write" in text
    assert "repository-url: https://test.pypi.org/legacy/" in text
    assert "https://test.pypi.org/p/torch-deepaudioembedding" in text
    assert "metadata.version('torch-deepaudioembedding')" in text
    assert "https://pypi.org/p/torch-deepaudioembedding" not in text
    assert "steps.project-version.outputs.value" in text


def test_release_workflow_version_checks_are_dynamic(repo_root: Path) -> None:
    pyproject = (repo_root / "pyproject.toml").read_text()
    current = re.search(r'(?m)^version = "([^"]+)"$', pyproject)
    assert current is not None
    workflow_text = "\n".join(
        (repo_root / ".github/workflows" / name).read_text()
        for name in ("publish.yml", "test-publish.yml")
    )
    version = re.escape(current.group(1))
    assert (
        re.search(rf"metadata\.version\([^)]*\)\s*==\s*['\"]{version}['\"]", workflow_text) is None
    )
    assert "tomllib.loads" in workflow_text
    assert workflow_text.count("PROJECT_VERSION") >= 8
    assert "all(version in name for name in names)" in workflow_text


def test_release_workflows_use_no_package_index_credentials(repo_root: Path) -> None:
    text = "\n".join(
        (repo_root / ".github/workflows" / name).read_text()
        for name in ("publish.yml", "test-publish.yml")
    )
    for forbidden in (
        "PYPI_API_TOKEN",
        "TEST_PYPI_API_TOKEN",
        "TWINE_PASSWORD",
        "username:",
        "password:",
    ):
        assert forbidden not in text
    assert text.count("pypa/gh-action-pypi-publish@release/v1") == 2
