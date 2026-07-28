from __future__ import annotations

import hashlib
import re
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest
import yaml

from scripts.validate_repository import MODEL_DEPS, numbered_stage_errors


def git_blob_hash(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def markdown_link_errors(root: Path, paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text()):
            clean = target.split("#", 1)[0]
            if (
                not clean
                or clean == "repository-relative/path"
                or clean.startswith(("http://", "https://", "mailto:"))
            ):
                continue
            if not (path.parent / clean).resolve().exists():
                errors.append(f"{path.relative_to(root)} -> {clean}")
    return errors


def test_project_spec_is_byte_preserved(repo_root: Path) -> None:
    assert (
        git_blob_hash(repo_root / "project_spec.md") == "e915c388190e64aeebaae2efead051ef98fe8a18"
    )


def test_tracked_paths_and_text_are_free_of_numbered_stage_labels(repo_root: Path) -> None:
    assert numbered_stage_errors(repo_root) == []


def test_public_package_metadata(repo_root: Path) -> None:
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text())
    project = pyproject["project"]
    assert project["name"] == "torch-deepaudioembedding"
    assert project["version"] == "0.1.0"
    assert project["description"] == (
        "torch-dae: an AI skill-based framework for Audio Embedding Models"
    )
    assert project["requires-python"] == ">=3.11,<3.13"
    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE", "NOTICE"]
    assert project["authors"] == [
        {
            "name": "Stefano Giacomelli",
            "email": "stefano.giacomelli@graduate.univaq.it",
        }
    ]
    assert set(project["keywords"]) >= {"audio", "pytorch", "model onboarding"}
    assert project["urls"]["Repository"] == "https://github.com/StefanoGiacomelli/torch_dae"
    assert project["scripts"] == {"torch-dae": "torch_dae.cli.main:app"}
    assert pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["src/torch_dae"]
    assert (repo_root / "src/torch_dae/__init__.py").is_file()
    assert project["name"] != "torch-dae"
    assert project["name"].replace("-", "_") != "torch_dae"
    dependency_names = {
        re.split(r"[^A-Za-z0-9_.-]", item, maxsplit=1)[0].lower()
        for item in project["dependencies"]
    }
    assert not dependency_names & MODEL_DEPS


def test_license_citation_and_contribution_files(repo_root: Path) -> None:
    license_text = (repo_root / "LICENSE").read_text()
    assert license_text.lstrip().startswith("Apache License")
    assert "Version 2.0, January 2004" in license_text
    for section in range(1, 10):
        assert f"   {section}." in license_text
    assert "END OF TERMS AND CONDITIONS" in license_text
    assert "APPENDIX: How to apply the Apache License to your work." in license_text
    assert not license_text.startswith("Copyright")
    assert (repo_root / "NOTICE").read_text() == ("torch-dae\nCopyright 2026 Stefano Giacomelli\n")

    citation_text = (repo_root / "CITATION.cff").read_text()
    citation = yaml.safe_load(citation_text)
    assert citation["cff-version"] == "1.2.0"
    assert citation["title"] == "torch-dae: an AI skill-based framework for Audio Embedding Models"
    assert citation["type"] == "software"
    assert citation["version"] == "0.1.0"
    assert citation["date-released"] == "2026-07-28"
    assert citation["doi"] == "10.5281/zenodo.21641391"

    identifiers = {(item["type"], item["value"]) for item in citation["identifiers"]}
    assert identifiers == {
        ("doi", "10.5281/zenodo.21641390"),
        ("doi", "10.5281/zenodo.21641391"),
    }

    assert citation["license"] == "Apache-2.0"
    assert citation["authors"][0]["orcid"] == "https://orcid.org/0009-0009-0438-1748"
    assert citation["authors"][0]["affiliation"] == (
        "Department of Information Engineering, Computer Science and Mathematics (DISIM), "
        "University of L'Aquila"
    )
    assert "DM) 118/2023" in citation["abstract"]
    assert "CUP: E11I23000100001" in citation["abstract"]
    assert "preferred-citation" not in citation
    assert citation["references"][0]["type"] == "conference-paper"
    assert citation["references"][0]["doi"] == "10.1109/ISCC65549.2025.11326439"

    contribution = (repo_root / "CONTRIBUTING.md").read_text()
    for value in (
        "uv sync --all-groups",
        "ruff format --check",
        "mypy src scripts",
        "pytest",
        "generate_schemas.py",
        "validate_repository.py",
        "validate_skill_artifacts.py",
        "model-specific runtime dependencies",
        "checkpoint binaries",
        "NumPy-style docstrings",
        "sphinx-build -W --keep-going",
        "Trusted Publishing",
        "Pull requests",
    ):
        assert value in contribution


def test_readme_badges_sections_and_public_status(repo_root: Path) -> None:
    text = (repo_root / "README.md").read_text()
    for badge in (
        "actions/workflows/ci.yml/badge.svg",
        "codecov.io",
        "3.11%20%7C%203.12",
        "Apache--2.0",
        "readthedocs.org/projects/torch-dae/badge/?version=stable",
        "zenodo.org/badge/DOI/10.5281/zenodo.21641390.svg",
        "status-release",
        "version-v0.1.0",
    ):
        assert badge in text
    sections = [
        "Project status",
        "Overview",
        "Current capabilities",
        "Not available yet",
        "Architecture",
        "Installation",
        "Quick start",
        "Illustrative model-wrapper usage",
        "Available CLI commands",
        "Audio-model-onboarding skill",
        "Copy-paste agent request",
        "Expected agent response",
        "Repository layout",
        "Development and validation",
        "Roadmap",
        "Funding",
        "Citations",
        "License",
        "Contact",
    ]
    positions = [text.index(f"## {section}") for section in sections]
    assert positions == sorted(positions)
    assert text.startswith("# torch-dae: an AI skill-based framework for Audio Embedding Models")
    assert "https://torch-dae.readthedocs.io/en/stable/" in text
    assert 'src="graphics/embedding_pipeline.png"' in text
    assert "raw.githubusercontent.com" not in text
    assert "from torch_dae import ModelCardRegistry" in text
    assert "from torch_dae.core import ModelCardRegistry" not in text
    assert 'registry.get_model_class("model_name")' in text
    assert "model.compute_embedding(" in text
    assert "model.predict_probability(" in text
    assert "torch-dae is pre-release research software" not in text

    graphic = repo_root / "graphics/embedding_pipeline.png"
    assert graphic.is_file()
    assert graphic.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    assert "pip install torch-deepaudioembedding" in text

    normalized_text = " ".join(text.split())
    assert (
        "No model-specific integrations are distributed in the current release." in normalized_text
    )
    assert "uv run torch-dae card list" in text
    assert "uv run torch-dae env create" in text
    assert "uv run torch-dae checkpoint ensure" in text
    for value in (
        "0009-0009-0438-1748",
        "10.5281/zenodo.21641390",
        "10.5281/zenodo.21641391",
        "phdict.disim.univaq.it/wp-content/uploads/2024/06/logo-univaq-disim-2-2-768x283.png",
        "stefano.giacomelli@graduate.univaq.it",
        "DM 118/2023",
        "Mission 4",
        "Component 1",
        "Investment 4.1",
        "PNRR Research",
        "CUP: E11I23000100001",
        "@software{giacomelli2026torch_dae",
        "10.1109/ISCC65549.2025.11326439",
    ):
        assert value in text


def test_agent_templates_have_exact_contract(repo_root: Path) -> None:
    request = (repo_root / "skills/audio-model-onboarding/templates/agent-request.md").read_text()
    for placeholder in (
        "MODE: <analyze | resolve-environment | integrate | verify | card>",
        "MODEL_NAME: <MODEL_NAME>",
        "UPSTREAM_REPOSITORY: <GITHUB_REPOSITORY_URL>",
        "PAPER_OR_TECHNICAL_REFERENCE: <PAPER_URL_OR_NONE>",
        "TARGET_VARIANT: <VARIANT_NAME_OR_AUTO_DISCOVER>",
        "TARGET_CHECKPOINT: <CHECKPOINT_NAME_OR_AUTO_DISCOVER>",
        "PREFERRED_EMBEDDING: <EMBEDDING_NAME_OR_UNRESOLVED>",
        "<OPTIONAL_PROJECT_SPECIFIC_CONDITIONING_OR_NONE>",
    ):
        assert placeholder in request
    response = (repo_root / "skills/audio-model-onboarding/templates/agent-response.md").read_text()
    for heading in (
        "## Summary",
        "## Work completed",
        "## Problems and resolutions",
        "## Open questions",
        "## Files",
        "## Validation",
    ):
        assert heading in response
    assert "None." in response


def test_empty_markers_and_typed_package_marker(repo_root: Path) -> None:
    for relative in (
        "environments/.gitkeep",
        "model_cards/.gitkeep",
        "verification_reports/.gitkeep",
        "src/torch_dae/py.typed",
    ):
        assert (repo_root / relative).read_bytes() == b""


def test_ci_and_codecov_contract(repo_root: Path) -> None:
    workflow = (repo_root / ".github/workflows/ci.yml").read_text()
    for action in (
        "actions/checkout@v7",
        "actions/setup-python@v7",
        "astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9 # v9.0.0",
        "actions/upload-artifact@v7",
        "codecov/codecov-action@v7",
    ):
        assert action in workflow
    assert 'python-version: ["3.11", "3.12"]' in workflow
    for command in (
        "uv lock --check",
        "uv sync --all-groups --frozen",
        "uv run ruff format --check",
        "uv run ruff check",
        "uv run mypy src scripts",
        "scripts/generate_schemas.py --check",
        "scripts/validate_repository.py",
        "validate_skill_artifacts.py . --json",
        "uv run python -m build",
        "uv run python -m twine check dist/*",
        "documentation-html",
        "python-package-distributions",
        "retention-days: 7",
        "--cov-branch",
        "--min-line 85",
        "--min-branch 70",
        "use_oidc: true",
        "head.repo.fork == false",
    ):
        assert command in workflow
    codecov = (repo_root / ".codecov.yml").read_text()
    for value in (
        "require_ci_to_pass: true",
        "target: 85%",
        "threshold: 1%",
        "target: 80%",
        "comment: false",
    ):
        assert value in codecov


def test_documentation_relative_links_resolve(repo_root: Path) -> None:
    paths = [repo_root / "README.md"]
    paths.extend(sorted((repo_root / "docs").rglob("*.md")))
    paths.extend(sorted((repo_root / "skills/audio-model-onboarding").glob("*.md")))
    paths.extend(sorted((repo_root / "skills/audio-model-onboarding/references").glob("*.md")))
    paths.extend(sorted((repo_root / "skills/audio-model-onboarding/templates").glob("*.md")))
    assert markdown_link_errors(repo_root, paths) == []


@pytest.fixture(scope="module")
def built_distributions(repo_root: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("distributions")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--outdir",
            str(output),
        ],
        cwd=repo_root,
        check=True,
    )
    return output


def test_license_files_in_distribution_metadata_and_archives(
    built_distributions: Path,
) -> None:
    wheel = next(built_distributions.glob("*.whl"))
    source = next(built_distributions.glob("*.tar.gz"))
    assert wheel.name == "torch_deepaudioembedding-0.1.0-py3-none-any.whl"
    assert source.name == "torch_deepaudioembedding-0.1.0.tar.gz"
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = archive.read(metadata_name).decode()
        assert "Name: torch-deepaudioembedding" in metadata
        assert "License-File: LICENSE" in metadata
        assert "License-File: NOTICE" in metadata
        assert any(name.endswith(".dist-info/licenses/LICENSE") for name in names)
        assert any(name.endswith(".dist-info/licenses/NOTICE") for name in names)
        assert any(name.endswith("torch_dae/py.typed") for name in names)
    with tarfile.open(source, "r:gz") as archive:
        names = archive.getnames()
        assert any(name.endswith("/LICENSE") for name in names)
        assert any(name.endswith("/NOTICE") for name in names)
        assert any(name.endswith("/src/torch_dae/py.typed") for name in names)


def test_no_future_integration_roadmap_terminology(repo_root: Path) -> None:
    prohibited_terms = (
        "pi" + "lot",
        "pi" + "lots",
        "pa" + "nns",
        "cn" + "n14",
        "byol" + "-a",
        "en" + "codec",
        "audio" + "clip",
        "hu" + "bert",
    )
    prohibited = re.compile(rf"\b(?:{'|'.join(prohibited_terms)})\b", re.IGNORECASE)
    public_paths = [
        repo_root / "README.md",
        repo_root / "CHANGELOG.md",
        repo_root / "CONTRIBUTING.md",
        repo_root / "CITATION.cff",
        repo_root / "pyproject.toml",
    ]
    public_paths.extend((repo_root / "docs").rglob("*.md"))
    public_paths.extend((repo_root / ".github/workflows").glob("*.yml"))
    assert {
        path.relative_to(repo_root).as_posix(): prohibited.findall(path.read_text())
        for path in public_paths
        if prohibited.search(path.read_text())
    } == {}


def test_no_tracked_checkpoint_binary(repo_root: Path) -> None:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    tracked = [item.decode() for item in result.stdout.split(b"\0") if item]
    for relative in tracked:
        path = repo_root / relative
        if path.suffix.lower() not in {".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".onnx"}:
            continue
        data = path.read_bytes()
        assert b"\0" not in data
        data.decode("utf-8")
