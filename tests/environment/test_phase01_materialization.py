from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from torch_dae.core.errors import (
    EnvironmentIdentityMismatchError,
    EnvironmentVerificationError,
    ExternalCommandError,
)
from torch_dae.environment.fingerprint import local_package_identity
from torch_dae.environment.manager import EnvironmentManager
from torch_dae.environment.policy import ExecutionPolicy


def write_phase01_repo(root: Path, repo_root: Path, valid_fixture_dir: Path) -> str:
    card_id = "phase01-synthetic-card"
    environment_id = "phase01-synthetic-shared-environment"
    version = ".".join(str(part) for part in sys.version_info[:3])
    (root / "project_spec.md").write_text("synthetic Phase 01 test repository\n")
    shutil.copy2(repo_root / "pyproject.toml", root / "pyproject.toml")
    shutil.copy2(repo_root / "README.md", root / "README.md")
    shutil.copy2(repo_root / "uv.lock", root / "uv.lock")
    shutil.copytree(repo_root / "src/torch_dae", root / "src/torch_dae")
    (root / "schemas").mkdir()
    (root / "schemas/model-card.schema.json").write_text(
        (repo_root / "schemas/model-card.schema.json").read_text()
    )
    (root / "model_cards/synthetic").mkdir(parents=True)
    checkpoint = root / "tests/fixtures/phase01/checkpoint.bin"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic checkpoint bytes\n")
    card = json.loads((valid_fixture_dir / "model-card.analyzed.json").read_text())
    card["card_id"] = card_id
    card["usage"]["recommended_environment"] = {
        "environment_id": environment_id,
        "specification": f"environments/{card_id}/environment.json",
        "lockfile": f"environments/{card_id}/uv.lock",
        "verified": True,
    }
    card["checkpoint"] = {
        "schema_version": "1.0.0",
        "checkpoint_id": "phase01-synthetic-checkpoint",
        "source_type": "local_path",
        "local_path": "tests/fixtures/phase01/checkpoint.bin",
        "format": "binary",
        "loader": "manual",
        "license": {"status": "not_applicable"},
    }
    (root / "model_cards/synthetic/card.json").write_text(json.dumps(card, indent=2))
    env_dir = root / f"environments/{card_id}"
    env_dir.mkdir(parents=True)
    shutil.copy2(root / "pyproject.toml", env_dir / "pyproject.toml")
    shutil.copy2(root / "uv.lock", env_dir / "uv.lock")
    (env_dir / "sources.json").write_text(
        json.dumps({"schema_version": "1.0.0", "environment_id": environment_id, "sources": []})
    )
    (env_dir / "environment.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "environment_id": environment_id,
                "model_card_id": card_id,
                "python": {"constraint": f"=={version}", "resolved_version": version},
                "platforms": {
                    "resolved_on": ["synthetic-local"],
                    "expected_compatible": [],
                    "verified": ["synthetic-local"],
                },
                "dependency_manager": "uv",
                "lockfile": f"environments/{card_id}/uv.lock",
                "project_file": f"environments/{card_id}/pyproject.toml",
                "sources_file": f"environments/{card_id}/sources.json",
                "verification": {"script": f"environments/{card_id}/verify_environment.py"},
            },
            indent=2,
        )
    )
    (env_dir / "verify_environment.py").write_text(
        """
from __future__ import annotations

import importlib.metadata as metadata
import os
import pathlib
import sysconfig

import torch_dae
import torch_dae.cards.models
import torch_dae.environment

assert "PYTHONPATH" not in os.environ
assert "PYTHONHOME" not in os.environ
assert metadata.version("torch-dae") == "0.1.0"
scripts = pathlib.Path(sysconfig.get_path("scripts"))
assert (scripts / "torch-dae").exists()
pyvenv = pathlib.Path(os.environ["TORCH_DAE_ENVIRONMENT_ROOT"]) / "pyvenv.cfg"
assert "include-system-site-packages = false" in pyvenv.read_text()
assert pathlib.Path(torch_dae.__file__).is_relative_to(
    pathlib.Path(os.environ["TORCH_DAE_ENVIRONMENT_ROOT"]).resolve()
)
print("synthetic verification passed")
""".lstrip()
    )
    return card_id


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("card-lock", "lockfile mismatch"),
        ("specification-other-card", "specification path"),
        ("spec-lock-other-card", "lockfile path"),
        ("spec-project-other-card", "project_file path"),
        ("spec-sources-other-card", "sources_file path"),
        ("spec-verify-other-card", "verification script path"),
    ],
)
def test_phase01_environment_load_rejects_cross_document_path_mismatch(
    tmp_path: Path,
    repo_root: Path,
    valid_fixture_dir: Path,
    mutation: str,
    message: str,
) -> None:
    card_id = write_phase01_repo(tmp_path, repo_root, valid_fixture_dir)
    card_path = tmp_path / "model_cards/synthetic/card.json"
    environment_path = tmp_path / f"environments/{card_id}/environment.json"
    card = json.loads(card_path.read_text())
    environment = json.loads(environment_path.read_text())
    if mutation == "card-lock":
        card["usage"]["recommended_environment"]["lockfile"] = f"environments/{card_id}/other.lock"
    elif mutation == "specification-other-card":
        card["usage"]["recommended_environment"]["specification"] = (
            "environments/other-card/environment.json"
        )
    elif mutation == "spec-lock-other-card":
        environment["lockfile"] = "environments/other-card/uv.lock"
        card["usage"]["recommended_environment"]["lockfile"] = "environments/other-card/uv.lock"
    elif mutation == "spec-project-other-card":
        environment["project_file"] = "environments/other-card/pyproject.toml"
    elif mutation == "spec-sources-other-card":
        environment["sources_file"] = "environments/other-card/sources.json"
    else:
        environment["verification"]["script"] = "environments/other-card/verify_environment.py"
    card_path.write_text(json.dumps(card, indent=2))
    environment_path.write_text(json.dumps(environment, indent=2))

    with pytest.raises(EnvironmentIdentityMismatchError, match=message):
        EnvironmentManager(tmp_path).info(card_id)


@pytest.mark.integration
def test_phase01_environment_lifecycle(
    tmp_path: Path,
    repo_root: Path,
    valid_fixture_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path.parent / f"{tmp_path.name}-uv-cache"))
    hostile = tmp_path / "hostile"
    (hostile / "torch_dae").mkdir(parents=True)
    (hostile / "torch_dae/__init__.py").write_text("raise RuntimeError('host leak')\n")
    monkeypatch.setenv("PYTHONPATH", str(hostile))
    monkeypatch.setenv("PYTHONHOME", "/definitely/not/pythonhome")
    card_id = write_phase01_repo(tmp_path, repo_root, valid_fixture_dir)
    manager = EnvironmentManager(
        tmp_path,
        policy=ExecutionPolicy(command_timeout_seconds=120, download_timeout_seconds=120),
    )
    resolved = manager.ensure(card_id)
    assert resolved.environment_id == "phase01-synthetic-shared-environment"
    assert resolved.model_card_id == card_id
    assert resolved.root == tmp_path / ".torch-dae/environments" / card_id / resolved.fingerprint
    assert resolved.python_executable.exists()
    assert not (tmp_path / f"environments/{card_id}/.venv").exists()
    assert resolved.installed_packages["torch-dae"] == "0.1.0"
    metadata = resolved.root / "lib/python{}.{}".format(*sys.version_info[:2])
    assert metadata.exists()
    materialization = json.loads((resolved.root / "torch-dae-materialization.json").read_text())
    command_refs = materialization["command_log_references"]
    assert command_refs
    assert (tmp_path / ".torch-dae" / command_refs[0]).is_file()
    operations = operations_for_refs(tmp_path, command_refs)
    assert {
        "python-resolution",
        "uv-venv",
        "uv-sync",
        "local-wheel-build",
        "local-wheel-install",
        "dependency-check",
        "verification-script",
    }.issubset(operations)

    verification = manager.verify(card_id)
    assert verification.passed
    run = manager.run(
        card_id,
        [
            "python",
            "-c",
            (
                "import importlib.metadata as m, os, pathlib, torch_dae, "
                "torch_dae.cards.models, torch_dae.environment; "
                "assert 'PYTHONPATH' not in os.environ; "
                "assert 'PYTHONHOME' not in os.environ; "
                "assert m.version('torch-dae') == '0.1.0'; "
                "assert pathlib.Path(torch_dae.__file__).is_relative_to("
                "pathlib.Path(os.environ['VIRTUAL_ENV']).resolve()); "
                "print('inside')"
            ),
        ],
    )
    assert run.returncode == 0
    assert run.stdout.strip() == "inside"
    reused = manager.ensure(card_id)
    assert reused.root == resolved.root
    info = manager.info(card_id)
    assert info.status == "valid"

    (resolved.root / ".torch-dae-complete").unlink()
    with pytest.raises(EnvironmentVerificationError, match="incomplete"):
        manager.verify(card_id)
    rebuilt = manager.ensure(card_id)
    assert rebuilt.root == resolved.root
    assert (rebuilt.root / ".torch-dae-complete").exists()
    manager.remove(card_id)
    assert not (tmp_path / ".torch-dae/environments" / card_id).exists()


@pytest.mark.integration
def test_phase01_offline_environment_reuse(
    tmp_path: Path,
    repo_root: Path,
    valid_fixture_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path.parent / f"{tmp_path.name}-uv-cache"))
    card_id = write_phase01_repo(tmp_path, repo_root, valid_fixture_dir)
    EnvironmentManager(tmp_path).ensure(card_id)
    offline = EnvironmentManager(tmp_path, policy=ExecutionPolicy(offline=True))
    assert offline.ensure(card_id).valid
    result = offline.run(card_id, ["python", "-c", "print('offline')"])
    assert result.stdout.strip() == "offline"


@pytest.mark.integration
@pytest.mark.parametrize(
    "drift",
    [
        "modify-package-file",
        "remove-dist-info",
        "replace-wheel",
        "remove-wheel-json",
        "malformed-wheel-json",
    ],
)
def test_phase01_environment_verify_detects_installed_integrity_drift(
    tmp_path: Path,
    repo_root: Path,
    valid_fixture_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path.parent / f"{tmp_path.name}-uv-cache"))
    card_id = write_phase01_repo(tmp_path, repo_root, valid_fixture_dir)
    manager = EnvironmentManager(tmp_path, policy=ExecutionPolicy(command_timeout_seconds=120))
    resolved = manager.ensure(card_id)
    if drift == "modify-package-file":
        location = manager.run(
            card_id,
            ["python", "-c", "import torch_dae; print(torch_dae.__file__)"],
        ).stdout.strip()
        Path(location).write_text("# modified after materialization\n")
        expected = "installed wheel file drifted|local torch-dae import check failed"
    elif drift == "remove-dist-info":
        site_packages = Path(
            manager.run(
                card_id,
                [
                    "python",
                    "-c",
                    "import importlib.metadata as m; print(m.distribution('torch-dae')._path)",
                ],
            ).stdout.strip()
        )
        shutil.rmtree(site_packages)
        expected = "torch-dae is not installed"
    else:
        wheel_cache_dir = next((tmp_path / ".torch-dae/source-builds/torch-dae").glob("*"))
        if drift == "replace-wheel":
            next(wheel_cache_dir.glob("*.whl")).write_bytes(b"not a wheel")
        elif drift == "remove-wheel-json":
            (wheel_cache_dir / "wheel.json").unlink()
        else:
            (wheel_cache_dir / "wheel.json").write_text("{bad json")
        expected = "local torch-dae wheel cache metadata is missing or invalid"
    with pytest.raises(EnvironmentVerificationError, match=expected):
        manager.verify(card_id)
    assert resolved.root.exists()


@pytest.mark.integration
def test_phase01_environment_materialization_logs_failed_command_with_redaction(
    tmp_path: Path,
    repo_root: Path,
    valid_fixture_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path.parent / f"{tmp_path.name}-uv-cache"))
    card_id = write_phase01_repo(tmp_path, repo_root, valid_fixture_dir)
    (tmp_path / f"environments/{card_id}/verify_environment.py").write_text(
        """
from __future__ import annotations

import sys

print("Authorization: Bearer secret-token")
raise SystemExit(7)
""".lstrip()
    )
    manager = EnvironmentManager(tmp_path, policy=ExecutionPolicy(command_timeout_seconds=120))
    with pytest.raises(EnvironmentVerificationError):
        manager.ensure(card_id)
    failed_metadata = failed_materialization_metadata(tmp_path, card_id)
    assert failed_metadata["status"] == "failed"
    logs = [tmp_path / ".torch-dae" / ref for ref in failed_metadata["command_log_references"]]
    assert logs
    assert any(json.loads(path.read_text())["operation"] == "verification-script" for path in logs)
    payload = "\n".join(path.read_text() for path in logs)
    assert "secret-token" not in payload
    assert "redacted secret" in payload


@pytest.mark.integration
def test_phase01_failed_uv_sync_metadata_references_reports(
    tmp_path: Path,
    repo_root: Path,
    valid_fixture_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path.parent / f"{tmp_path.name}-uv-cache"))
    card_id = write_phase01_repo(tmp_path, repo_root, valid_fixture_dir)
    (tmp_path / f"environments/{card_id}/uv.lock").write_text("not a uv lock\n")

    with pytest.raises(ExternalCommandError):
        EnvironmentManager(tmp_path, policy=ExecutionPolicy(command_timeout_seconds=120)).ensure(
            card_id
        )

    metadata = failed_materialization_metadata(tmp_path, card_id)
    assert metadata["status"] == "failed"
    assert "uv-sync" in operations_for_refs(tmp_path, metadata["command_log_references"])


@pytest.mark.integration
def test_phase01_failed_git_clone_metadata_references_reports(
    tmp_path: Path,
    repo_root: Path,
    valid_fixture_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path.parent / f"{tmp_path.name}-uv-cache"))
    card_id = write_phase01_repo(tmp_path, repo_root, valid_fixture_dir)
    write_git_source_manifest(tmp_path, card_id, str(tmp_path / "missing-upstream"), "b" * 40)

    with pytest.raises(ExternalCommandError):
        EnvironmentManager(tmp_path, policy=ExecutionPolicy(command_timeout_seconds=120)).ensure(
            card_id
        )

    metadata = failed_materialization_metadata(tmp_path, card_id)
    assert metadata["status"] == "failed"
    assert "git-clone" in operations_for_refs(tmp_path, metadata["command_log_references"])


@pytest.mark.integration
def test_phase01_failed_git_wheel_build_metadata_references_reports(
    tmp_path: Path,
    repo_root: Path,
    valid_fixture_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path.parent / f"{tmp_path.name}-uv-cache"))
    card_id = write_phase01_repo(tmp_path, repo_root, valid_fixture_dir)
    upstream = tmp_path / "bad-upstream"
    upstream.mkdir()
    (upstream / "pyproject.toml").write_text(
        """
[build-system]
requires = []
build-backend = "missing_backend"

[project]
name = "bad-git-package"
version = "0.1.0"
""".lstrip()
    )
    (upstream / "bad_git_package.py").write_text("VALUE = 1\n")
    run_git(upstream, "init")
    run_git(upstream, "add", ".")
    run_git(
        upstream,
        "-c",
        "user.name=Test User",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "initial",
    )
    revision = git_text(upstream, "rev-parse", "HEAD")
    write_git_source_manifest(tmp_path, card_id, str(upstream), revision)

    with pytest.raises(ExternalCommandError):
        EnvironmentManager(tmp_path, policy=ExecutionPolicy(command_timeout_seconds=120)).ensure(
            card_id
        )

    metadata = failed_materialization_metadata(tmp_path, card_id)
    assert metadata["status"] == "failed"
    assert "git-wheel-build" in operations_for_refs(tmp_path, metadata["command_log_references"])


@pytest.mark.integration
def test_phase01_local_wheel_backend_build_is_reproducible(
    tmp_path: Path,
    repo_root: Path,
    valid_fixture_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path.parent / f"{tmp_path.name}-uv-cache"))
    roots = (tmp_path / "first", tmp_path / "second")
    shas: list[str] = []
    for root in roots:
        root.mkdir()
        write_phase01_repo(root, repo_root, valid_fixture_dir)
        manager = EnvironmentManager(root, policy=ExecutionPolicy(command_timeout_seconds=120))
        wheel, sha = manager._build_local_wheel(local_package_identity(root))
        assert wheel.is_file()
        assert sha
        shas.append(sha)
    assert shas[0] == shas[1]


def operations_for_refs(root: Path, references: list[str]) -> set[str]:
    return {json.loads((root / ".torch-dae" / ref).read_text())["operation"] for ref in references}


def failed_materialization_metadata(root: Path, card_id: str) -> dict[str, object]:
    metadata_path = next(
        (root / ".torch-dae/environments" / card_id / ".failed").glob(
            "*/torch-dae-materialization.json"
        )
    )
    return json.loads(metadata_path.read_text())


def write_git_source_manifest(root: Path, card_id: str, url: str, revision: str) -> None:
    env_dir = root / "environments" / card_id
    sources = {
        "schema_version": "1.0.0",
        "environment_id": "phase01-synthetic-shared-environment",
        "sources": [
            {
                "source_id": "git-source",
                "role": "model_implementation",
                "installation": "git",
                "url": url,
                "revision": revision,
                "build": "wheel",
            }
        ],
    }
    (env_dir / "sources.json").write_text(json.dumps(sources, indent=2))


def run_git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)


def git_text(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
