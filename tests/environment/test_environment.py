from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from torch_dae.core.errors import EnvironmentMaterializationError
from torch_dae.environment.fingerprint import (
    FingerprintInputs,
    calculate_environment_fingerprint,
    local_package_identity,
    local_package_provenance,
)
from torch_dae.environment.manager import EnvironmentManager, local_wheel_cache_key
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


def test_direct_environment_resolution_rejects_missing_and_invalid_specifications(
    tmp_path: Path,
) -> None:
    (tmp_path / "project_spec.md").write_text("synthetic\n")
    manager = EnvironmentManager(tmp_path)
    with pytest.raises(EnvironmentMaterializationError, match="not found"):
        manager.resolve_environment("missing-environment")

    invalid = tmp_path / "environments/invalid-environment"
    invalid.mkdir(parents=True)
    (invalid / "environment.json").write_text("{invalid json")
    with pytest.raises(EnvironmentMaterializationError, match="invalid environment specification"):
        manager.resolve_environment("invalid-environment")


def test_environment_manager_missing_card_operations(repo_root: Path) -> None:
    manager = EnvironmentManager(repo_root)
    with pytest.raises(EnvironmentMaterializationError):
        manager.create("card")
    with pytest.raises(EnvironmentMaterializationError):
        manager.ensure("card")
    with pytest.raises(EnvironmentMaterializationError):
        manager.verify("card")
    manager.remove("card")
    with pytest.raises(EnvironmentMaterializationError):
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
    (root / "pyproject.toml").write_text(
        "[project]\nname='fixture'\nversion='0.0.0'\nreadme='README.md'\n"
    )
    (root / "README.md").write_text("fixture readme\n")
    (root / "src/torch_dae/__init__.py").write_text("__version__ = '0.0.0'\n")
    (root / "src/torch_dae/package-data.json").write_text('{"value": 1}\n')
    (root / "src/torch_dae/vendor").mkdir()
    (root / "src/torch_dae/vendor/source.txt").write_text("vendored\n")
    (root / "onboarding_reports/workflow/integrate").mkdir(parents=True)
    (root / "onboarding_reports/workflow/integrate/report.json").write_text("{}\n")
    (root / "model_cards/family").mkdir(parents=True)
    (root / "model_cards/family/card.json").write_text("{}\n")
    (root / "verification_reports/family").mkdir(parents=True)
    (root / "verification_reports/family/report.json").write_text("{}\n")
    run_git(root, "init")
    return commit_all(root)


def test_local_package_identity_is_equal_in_clean_and_dirty_repository(tmp_path: Path) -> None:
    make_package_repo(tmp_path)
    clean = local_package_identity(tmp_path)
    assert clean.startswith("content-sha256:")
    report = tmp_path / "onboarding_reports/workflow/integrate/report.json"
    report.write_text('{"changed": true}\n')
    assert local_package_identity(tmp_path) == clean


def test_committing_staged_package_bytes_does_not_change_identity(tmp_path: Path) -> None:
    make_package_repo(tmp_path)
    source = tmp_path / "src/torch_dae/__init__.py"
    source.write_text("__version__ = '0.0.1'\n")
    dirty_identity = local_package_identity(tmp_path)
    run_git(tmp_path, "add", "src/torch_dae/__init__.py")
    assert local_package_identity(tmp_path) == dirty_identity
    commit_all(tmp_path)
    assert local_package_identity(tmp_path) == dirty_identity


def test_changing_only_head_does_not_change_package_identity(tmp_path: Path) -> None:
    first_head = make_package_repo(tmp_path)
    identity = local_package_identity(tmp_path)
    report = tmp_path / "onboarding_reports/workflow/integrate/report.json"
    report.write_text('{"revision": 2}\n')
    second_head = commit_all(tmp_path)
    assert second_head != first_head
    assert local_package_identity(tmp_path) == identity


@pytest.mark.parametrize(
    "relative",
    [
        "onboarding_reports/workflow/integrate/report.json",
        "model_cards/family/card.json",
        "verification_reports/family/report.json",
    ],
)
def test_non_wheel_evidence_changes_do_not_change_package_identity(
    tmp_path: Path,
    relative: str,
) -> None:
    make_package_repo(tmp_path)
    identity = local_package_identity(tmp_path)
    (tmp_path / relative).write_text('{"changed": true}\n')
    assert local_package_identity(tmp_path) == identity


def test_packaged_source_change_changes_package_identity(tmp_path: Path) -> None:
    make_package_repo(tmp_path)
    identity = local_package_identity(tmp_path)
    (tmp_path / "src/torch_dae/__init__.py").write_text("VALUE = 'changed'\n")
    assert local_package_identity(tmp_path) != identity


def test_active_package_metadata_change_changes_package_identity(tmp_path: Path) -> None:
    make_package_repo(tmp_path)
    identity = local_package_identity(tmp_path)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(pyproject.read_text().replace("version='0.0.0'", "version='0.0.1'"))
    assert local_package_identity(tmp_path) != identity


def test_packaged_readme_change_changes_package_identity(tmp_path: Path) -> None:
    make_package_repo(tmp_path)
    identity = local_package_identity(tmp_path)
    (tmp_path / "README.md").write_text("changed wheel metadata description\n")
    assert local_package_identity(tmp_path) != identity


def test_git_provenance_is_informational_and_separate(tmp_path: Path) -> None:
    head = make_package_repo(tmp_path)
    clean = local_package_provenance(tmp_path)
    assert clean.repository_head == head
    assert clean.repository_dirty is False
    assert local_package_identity(tmp_path) == f"content-sha256:{clean.package_content_sha256}"
    report = tmp_path / "onboarding_reports/workflow/integrate/report.json"
    report.write_text('{"dirty": true}\n')
    dirty = local_package_provenance(tmp_path)
    assert dirty.repository_head == head
    assert dirty.repository_dirty is True
    assert dirty.package_content_sha256 == clean.package_content_sha256


def test_environment_fingerprint_is_stable_across_dirty_to_clean_transition(
    tmp_path: Path,
    valid_fixture_dir: Path,
) -> None:
    make_package_repo(tmp_path)
    spec = EnvironmentSpecification.model_validate_json(
        (valid_fixture_dir / "environment.synthetic.json").read_text()
    )
    sources = EnvironmentSourcesManifest.model_validate_json(
        (valid_fixture_dir / "environment-sources.synthetic.json").read_text()
    )
    report = tmp_path / "onboarding_reports/workflow/integrate/report.json"
    report.write_text('{"dirty": true}\n')
    dirty_identity = local_package_identity(tmp_path)
    dirty_fingerprint = calculate_environment_fingerprint(
        FingerprintInputs(spec, b"lock", sources, "macos-arm64", dirty_identity)
    )
    commit_all(tmp_path)
    clean_identity = local_package_identity(tmp_path)
    clean_fingerprint = calculate_environment_fingerprint(
        FingerprintInputs(spec, b"lock", sources, "macos-arm64", clean_identity)
    )
    assert clean_identity == dirty_identity
    assert clean_fingerprint == dirty_fingerprint


def test_wheel_cache_key_is_stable_across_dirty_to_clean_transition(tmp_path: Path) -> None:
    make_package_repo(tmp_path)
    source = tmp_path / "src/torch_dae/__init__.py"
    source.write_text("VALUE = 'staged bytes'\n")
    dirty_key = local_wheel_cache_key(local_package_identity(tmp_path))
    commit_all(tmp_path)
    assert local_wheel_cache_key(local_package_identity(tmp_path)) == dirty_key


def test_local_package_identity_does_not_require_git(tmp_path: Path) -> None:
    (tmp_path / "src/torch_dae").mkdir(parents=True)
    (tmp_path / "src/torch_dae/__init__.py").write_text("VALUE = 1\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='0.0.0'\n")
    assert local_package_identity(tmp_path).startswith("content-sha256:")
