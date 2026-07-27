from __future__ import annotations

import ast
import importlib
import inspect
import os
import re
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

MODEL_DEPENDENCIES = {
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
    "tensorflow",
    "jax",
    "librosa",
}
MANIFEST_METADATA = {"schema_version", "description"}
DIRECTIVE_PATTERN = re.compile(
    r"^\.\. (?:autoclass|autofunction|autoexception):: ([A-Za-z0-9_.]+)$",
    re.MULTILINE,
)


def load_manifest(repo_root: Path) -> tuple[dict[str, Any], ...]:
    data = tomllib.loads((repo_root / "docs/api/public-api.toml").read_text())
    symbols: list[dict[str, Any]] = []
    for section, value in data.items():
        if section in MANIFEST_METADATA:
            continue
        assert isinstance(value, dict), section
        section_symbols = value.get("symbols")
        assert isinstance(section_symbols, list) and section_symbols, section
        symbols.extend(section_symbols)
    return tuple(symbols)


def import_symbol(path: str) -> object:
    module_name, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module_name), name)


def test_readthedocs_configuration(repo_root: Path) -> None:
    config = yaml.safe_load((repo_root / ".readthedocs.yaml").read_text())
    assert config["version"] == 2
    assert config["build"]["os"] == "ubuntu-24.04"
    assert config["build"]["tools"]["python"] == "3.12"
    assert config["sphinx"] == {
        "configuration": "docs/conf.py",
        "builder": "html",
        "fail_on_warning": True,
    }
    assert config["python"]["install"] == [
        {"method": "pip", "path": ".", "extra_requirements": ["docs"]}
    ]


def test_sphinx_configuration_uses_required_extensions_and_theme(repo_root: Path) -> None:
    text = (repo_root / "docs/conf.py").read_text()
    for extension in (
        '"myst_parser"',
        '"sphinx.ext.autodoc"',
        '"sphinx.ext.autosummary"',
        '"sphinx.ext.napoleon"',
        '"sphinx.ext.intersphinx"',
        '"sphinx.ext.viewcode"',
    ):
        assert extension in text
    assert 'html_theme = "furo"' in text
    assert "napoleon_numpy_docstring = True" in text
    assert "autosummary_generate = False" in text
    assert 'html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")' in text
    assert 'html_css_files = ["custom.css"]' in text
    assert (repo_root / "docs/_static/custom.css").is_file()


def test_distribution_install_and_documentation_names(repo_root: Path) -> None:
    installation = (repo_root / "docs/getting-started/installation.md").read_text()
    release = (repo_root / "docs/development/releasing.md").read_text()
    readme = (repo_root / "README.md").read_text()
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text())
    assert "pip install torch-deepaudioembedding" in installation
    assert "import torch_dae" in installation
    assert "torch-dae --help" in installation
    assert "- PyPI project name: `torch-deepaudioembedding`" in release
    assert "- TestPyPI project name: `torch-deepaudioembedding`" in release
    assert "- GitHub repository: `torch_dae`" in release
    assert "documentation slug remain `torch-dae`" in release
    assert (
        pyproject["project"]["urls"]["Documentation"]
        == "https://torch-dae.readthedocs.io/en/latest/"
    )
    assert "https://readthedocs.org/projects/torch-dae/badge/?version=latest" in readme
    assert "https://torch-dae.readthedocs.io/en/latest/" in readme


def test_manifest_is_unique_importable_and_complete(repo_root: Path) -> None:
    symbols = load_manifest(repo_root)
    paths = [entry["path"] for entry in symbols]
    assert len(paths) == len(set(paths))
    assert {entry["audience"] for entry in symbols} == {"user", "integrator", "developer"}
    for entry in symbols:
        assert set(entry) == {
            "path",
            "display_name",
            "category",
            "audience",
            "methods",
            "page",
        }
        target = import_symbol(entry["path"])
        assert entry["display_name"] == target.__name__
        assert (repo_root / "docs" / entry["page"]).is_file()
        for method in entry["methods"]:
            assert hasattr(target, method), f"{entry['path']}.{method}"


def test_manifest_is_the_only_curated_directive_list(repo_root: Path) -> None:
    symbols = load_manifest(repo_root)
    pages = sorted((repo_root / "docs/api").glob("*.rst"))
    api_text = "\n".join(path.read_text() for path in pages)
    documented = DIRECTIVE_PATTERN.findall(api_text)
    expected = [entry["path"] for entry in symbols]
    assert Counter(documented) == Counter(expected)
    for entry in symbols:
        page = repo_root / "docs" / entry["page"]
        page_text = page.read_text()
        assert (
            len(
                re.findall(
                    rf"(?m)^\.\. (?:autoclass|autofunction|autoexception):: "
                    rf"{re.escape(entry['path'])}\s*$",
                    page_text,
                )
            )
            == 1
        )
        for method in entry["methods"]:
            assert (
                len(
                    re.findall(
                        rf"(?m)^\s*\.\. automethod:: {re.escape(method)}\s*$",
                        page_text,
                    )
                )
                == 1
            )
    assert ":members:" not in api_text
    assert ".. automodule::" not in api_text
    assert "torch_dae.models" not in api_text
    assert "src/torch_dae/models" not in api_text


def test_api_docstrings_are_meaningful_and_method_aware(repo_root: Path) -> None:
    forbidden = re.compile(r"\b(?:TODO|TBD|placeholder documentation)\b", re.IGNORECASE)
    for entry in load_manifest(repo_root):
        target = import_symbol(entry["path"])
        doc = inspect.getdoc(target) or ""
        summary = doc.splitlines()[0] if doc else ""
        assert len(summary.split()) >= 2, entry["path"]
        assert forbidden.search(doc) is None, entry["path"]
        if inspect.isfunction(target):
            signature = inspect.signature(target)
            public_parameters = [name for name in signature.parameters if not name.startswith("_")]
            if public_parameters:
                assert "Parameters\n----------" in doc
                for name in public_parameters:
                    assert re.search(rf"(?m)^\**{re.escape(name)}(?:\s|$)", doc), (
                        entry["path"],
                        name,
                    )
            if signature.return_annotation not in {None, type(None), "None"}:
                assert "Returns\n-------" in doc
        for method_name in entry["methods"]:
            method = getattr(target, method_name)
            method_doc = inspect.getdoc(method) or ""
            signature = inspect.signature(method)
            public_parameters = [
                name
                for name in signature.parameters
                if name not in {"self", "cls"} and not name.startswith("_")
            ]
            assert len(method_doc.splitlines()[0].split()) >= 3
            assert forbidden.search(method_doc) is None
            if public_parameters:
                assert "Parameters\n----------" in method_doc
                for name in public_parameters:
                    assert re.search(rf"(?m)^\**{re.escape(name)}(?:\s|$)", method_doc), (
                        entry["path"],
                        method_name,
                        name,
                    )
            if signature.return_annotation not in {None, type(None), "None"}:
                assert "Returns\n-------" in method_doc


def test_api_structure_and_native_visuals(repo_root: Path) -> None:
    index = (repo_root / "docs/api/index.md").read_text()
    pages = {
        "registry",
        "model-execution",
        "outputs-embeddings",
        "capabilities",
        "checkpoints",
        "model-cards",
        "environment-specifications",
        "environment-lifecycle",
        "runtime-verification",
        "onboarding-contracts",
        "onboarding-inspection",
        "report-rendering",
    }
    for page in pages:
        assert page in index
    all_docs = "\n".join(
        path.read_text()
        for path in (repo_root / "docs").rglob("*")
        if path.suffix in {".md", ".rst"} and "_build" not in path.parts
    )
    assert all_docs.count(".. math::") >= 3
    assert all_docs.count("api-flow") >= 3
    assert "mermaid" not in all_docs.lower()
    assert "no recursive" in index.lower()
    assert "model-specific wrappers are opt-in" in index.lower()


def test_complete_documentation_builds_and_contains_api_anchors(
    repo_root: Path, tmp_path: Path
) -> None:
    output = tmp_path / "html"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-W",
            "--keep-going",
            "-b",
            "html",
            str(repo_root / "docs"),
            str(output),
        ],
        cwd=repo_root,
        check=True,
    )
    expected_pages = {
        "index.html",
        "api/registry.html",
        "api/model-execution.html",
        "api/checkpoints.html",
        "api/model-cards.html",
        "api/environment-lifecycle.html",
        "api/onboarding-contracts.html",
        "api/onboarding-inspection.html",
        "api/report-rendering.html",
    }
    for path in expected_pages:
        assert (output / path).is_file(), path
    smoke = "\n".join(path.read_text() for path in (output / "api").glob("*.html"))
    for name in (
        "ModelCardRegistry",
        "AudioModelProtocol",
        "compute_embedding",
        "CheckpointManager.ensure",
        "EnvironmentManager.create",
        "EnvironmentManager.verify",
        "ModelCard",
        "AnalysisReport",
        "inspect_scenario_repository",
        "render_analysis_markdown",
    ):
        assert name in smoke
    assert (output / "_modules").is_dir()


def test_stable_namespaces_match_curated_exports(repo_root: Path) -> None:
    intended = {
        "torch_dae": {"ModelCardRegistry"},
        "torch_dae.cards": {"ModelCard"},
        "torch_dae.core": {"AudioModelProtocol", "CheckpointManager", "EmbeddingOutput"},
        "torch_dae.environment": {
            "EnvironmentManager",
            "EnvironmentSpecification",
            "VerificationReport",
        },
        "torch_dae.onboarding": {
            "AnalysisReport",
            "EnvironmentCandidate",
            "inspect_repository",
            "render_analysis_markdown",
        },
    }
    for module_name, names in intended.items():
        module = importlib.import_module(module_name)
        assert names <= set(module.__all__)
        for name in names:
            assert getattr(module, name) is not None


def test_public_modules_have_no_eager_model_dependency_imports(repo_root: Path) -> None:
    public_files = [
        repo_root / "src/torch_dae/__init__.py",
        repo_root / "src/torch_dae/cards/__init__.py",
        repo_root / "src/torch_dae/core/__init__.py",
        repo_root / "src/torch_dae/environment/__init__.py",
        repo_root / "src/torch_dae/onboarding/__init__.py",
    ]
    eager_imports: set[str] = set()
    for path in public_files:
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.Import):
                eager_imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                eager_imports.add(node.module.split(".", 1)[0])
    assert eager_imports.isdisjoint(MODEL_DEPENDENCIES)

    dependencies = sorted(MODEL_DEPENDENCIES)
    code = f"""
import json
import sys
import torch_dae
import torch_dae.cards
import torch_dae.core
import torch_dae.environment
import torch_dae.onboarding
print(json.dumps(sorted(set(sys.modules) & set({dependencies!r}))))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "[]"
