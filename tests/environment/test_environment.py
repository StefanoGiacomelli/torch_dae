from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from torch_dae.core.errors import NotImplementedInPhaseError
from torch_dae.environment.fingerprint import (
    FingerprintInputs,
    calculate_environment_fingerprint,
    local_package_identity,
)
from torch_dae.environment.manager import EnvironmentManager
from torch_dae.environment.materialization import materialization_path
from torch_dae.environment.specification import EnvironmentSourcesManifest, EnvironmentSpecification


def test_environment_specification_and_fingerprint(valid_fixture_dir: Path) -> None:
    spec = EnvironmentSpecification.model_validate_json(
        (valid_fixture_dir / "environment.synthetic.json").read_text()
    )
    sources = EnvironmentSourcesManifest.model_validate_json(
        (valid_fixture_dir / "environment-sources.synthetic.json").read_text()
    )
    inputs = FingerprintInputs(spec, b"lock", sources, "macos-arm64", "identity")
    assert calculate_environment_fingerprint(inputs) == calculate_environment_fingerprint(inputs)
    changed = FingerprintInputs(spec, b"lock changed", sources, "macos-arm64", "identity")
    assert calculate_environment_fingerprint(inputs) != calculate_environment_fingerprint(changed)


def test_fingerprint_changes_for_each_material_input(valid_fixture_dir: Path) -> None:
    import json

    spec_data = json.loads((valid_fixture_dir / "environment.synthetic.json").read_text())
    sources_data = json.loads(
        (valid_fixture_dir / "environment-sources.synthetic.json").read_text()
    )
    spec = EnvironmentSpecification.model_validate(spec_data)
    sources = EnvironmentSourcesManifest.model_validate(sources_data)
    baseline = FingerprintInputs(spec, b"lock", sources, "macos-arm64", "identity")
    baseline_hash = calculate_environment_fingerprint(baseline)

    variants = []
    changed_spec_data = json.loads(json.dumps(spec_data))
    changed_spec_data["project_file"] = "environments/synthetic-family-variant-checkpoint/alt.toml"
    variants.append(
        FingerprintInputs(
            EnvironmentSpecification.model_validate(changed_spec_data),
            b"lock",
            sources,
            "macos-arm64",
            "identity",
        )
    )
    changed_python_data = json.loads(json.dumps(spec_data))
    changed_python_data["python"]["resolved_version"] = "3.11.9"
    changed_python_data["python"]["constraint"] = ">=3.11,<3.13"
    variants.append(
        FingerprintInputs(
            EnvironmentSpecification.model_validate(changed_python_data),
            b"lock",
            sources,
            "macos-arm64",
            "identity",
        )
    )
    variants.append(FingerprintInputs(spec, b"lock2", sources, "macos-arm64", "identity"))
    changed_sources_data = json.loads(json.dumps(sources_data))
    changed_sources_data["sources"][0]["version"] = "0.0.1"
    variants.append(
        FingerprintInputs(
            spec,
            b"lock",
            EnvironmentSourcesManifest.model_validate(changed_sources_data),
            "macos-arm64",
            "identity",
        )
    )
    changed_sources_data = json.loads(json.dumps(sources_data))
    changed_sources_data["sources"][1]["revision"] = "e" * 40
    variants.append(
        FingerprintInputs(
            spec,
            b"lock",
            EnvironmentSourcesManifest.model_validate(changed_sources_data),
            "macos-arm64",
            "identity",
        )
    )
    changed_sources_data = json.loads(json.dumps(sources_data))
    changed_sources_data["sources"][0] = json.loads(json.dumps(changed_sources_data["sources"][1]))
    changed_sources_data["sources"][0]["source_id"] = "changed-strategy"
    variants.append(
        FingerprintInputs(
            spec,
            b"lock",
            EnvironmentSourcesManifest.model_validate(changed_sources_data),
            "macos-arm64",
            "identity",
        )
    )
    variants.append(FingerprintInputs(spec, b"lock", sources, "linux-arm64", "identity"))
    variants.append(FingerprintInputs(spec, b"lock", sources, "macos-x86_64", "identity"))
    variants.append(FingerprintInputs(spec, b"lock", sources, "macos-arm64", "identity2"))

    assert all(calculate_environment_fingerprint(item) != baseline_hash for item in variants)


def test_materialization_path() -> None:
    assert (
        materialization_path(Path(".torch-dae"), "card", "a" * 64)
        == Path(".torch-dae/environments/card/" + "a" * 64).resolve()
    )


@pytest.mark.parametrize("card_id", ["../escape", "a/b", "a\\b", "has space"])
def test_materialization_path_rejects_escaping_ids(card_id: str) -> None:
    with pytest.raises(ValueError):
        materialization_path(Path(".torch-dae"), card_id, "a" * 64)


@pytest.mark.parametrize("card_id", ["../escape", "a/b", "a\\b", "has space"])
def test_environment_specification_path_rejects_escaping_ids(repo_root: Path, card_id: str) -> None:
    with pytest.raises(ValueError):
        EnvironmentManager(repo_root).specification_path(card_id)


def test_environment_manager_absent_state(repo_root: Path) -> None:
    info = EnvironmentManager(repo_root).info("missing-card")
    assert not info.specification_exists
    assert not info.materialized


def test_environment_manager_deferred_operations(repo_root: Path) -> None:
    manager = EnvironmentManager(repo_root)
    with pytest.raises(NotImplementedInPhaseError):
        manager.create("card")
    with pytest.raises(NotImplementedInPhaseError):
        manager.ensure("card")
    with pytest.raises(NotImplementedInPhaseError):
        manager.verify("card")
    with pytest.raises(NotImplementedInPhaseError):
        manager.remove("card")
    with pytest.raises(NotImplementedInPhaseError):
        manager.run("card", ["python", "--version"])


def test_invalid_environment_git_revision(invalid_fixture_dir: Path) -> None:
    with pytest.raises(ValidationError):
        EnvironmentSpecification.model_validate_json(
            (invalid_fixture_dir / "environment.invalid-git-revision.json").read_text()
        )


def write_environment_files(root: Path, model_card_id: str = "card-one") -> None:
    env_dir = root / "environments/card-one"
    env_dir.mkdir(parents=True)
    (env_dir / "uv.lock").write_text("lock")
    (env_dir / "sources.json").write_text(
        """
{
  "schema_version": "1.0.0",
  "environment_id": "environment-one",
  "sources": [
    {
      "source_id": "source-one",
      "role": "model_implementation",
      "installation": "package",
      "package": "synthetic",
      "version": "0.0.0"
    }
  ]
}
""".strip()
    )
    (env_dir / "environment.json").write_text(
        f"""
{{
  "schema_version": "1.0.0",
  "environment_id": "environment-one",
  "model_card_id": "{model_card_id}",
  "python": {{"constraint": ">=3.11,<3.13", "resolved_version": "3.12.13"}},
  "platforms": {{"resolved_on": ["macos-arm64"], "expected_compatible": [], "verified": []}},
  "dependency_manager": "uv",
  "lockfile": "environments/card-one/uv.lock",
  "project_file": "environments/card-one/pyproject.toml",
  "sources_file": "environments/card-one/sources.json",
  "verification": {{"script": "environments/card-one/verify_environment.py"}}
}}
""".strip()
    )


def test_environment_identity_coherence_valid(tmp_path: Path) -> None:
    (tmp_path / "project_spec.md").write_text("spec")
    write_environment_files(tmp_path)
    manager = EnvironmentManager(tmp_path)
    spec = manager.load_specification("card-one")
    assert spec.model_card_id == "card-one"
    assert manager.load_sources_manifest(spec).environment_id == spec.environment_id


def test_environment_load_rejects_requested_card_mismatch(tmp_path: Path) -> None:
    (tmp_path / "project_spec.md").write_text("spec")
    write_environment_files(tmp_path, model_card_id="other-card")
    with pytest.raises(ValueError, match="model_card_id"):
        EnvironmentManager(tmp_path).load_specification("card-one")


def test_environment_source_manifest_rejects_environment_mismatch(tmp_path: Path) -> None:
    (tmp_path / "project_spec.md").write_text("spec")
    write_environment_files(tmp_path)
    sources = tmp_path / "environments/card-one/sources.json"
    sources.write_text(sources.read_text().replace("environment-one", "environment-two"))
    manager = EnvironmentManager(tmp_path)
    spec = manager.load_specification("card-one")
    with pytest.raises(ValueError, match="environment_id"):
        manager.load_sources_manifest(spec)


def run_git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)


def commit_all(root: Path) -> str:
    run_git(root, "add", ".")
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test User",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "snapshot",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def make_package_repo(root: Path) -> str:
    (root / "src/torch_dae").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='0.0.0'\n")
    (root / "src/torch_dae/__init__.py").write_text("__version__ = '0.0.0'\n")
    run_git(root, "init")
    return commit_all(root)


def test_local_package_identity_clean_and_dirty_states(tmp_path: Path) -> None:
    head = make_package_repo(tmp_path)
    clean = local_package_identity(tmp_path)
    assert clean == f"git:{head}"
    assert local_package_identity(tmp_path) == clean

    source = tmp_path / "src/torch_dae/__init__.py"
    source.write_text("__version__ = '0.0.1'\n")
    unstaged = local_package_identity(tmp_path)
    assert unstaged.startswith("content:")
    assert unstaged != clean
    assert local_package_identity(tmp_path) == unstaged

    run_git(tmp_path, "add", "src/torch_dae/__init__.py")
    staged = local_package_identity(tmp_path)
    assert staged.startswith("content:")
    assert staged == unstaged

    commit_all(tmp_path)
    (tmp_path / "src/torch_dae/new_module.py").write_text("VALUE = 1\n")
    untracked = local_package_identity(tmp_path)
    assert untracked.startswith("content:")
    assert untracked != clean

    run_git(tmp_path, "add", "src/torch_dae/new_module.py")
    commit_all(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='0.0.1'\n")
    pyproject_dirty = local_package_identity(tmp_path)
    assert pyproject_dirty.startswith("content:")
    assert pyproject_dirty != untracked
