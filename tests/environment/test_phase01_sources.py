from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from torch_dae.core.errors import (
    ExternalCommandError,
    OfflineResourceUnavailableError,
    SourceMaterializationError,
)
from torch_dae.environment.policy import ExecutionPolicy
from torch_dae.environment.sources import SourceContext, SourceManager
from torch_dae.environment.specification import EnvironmentSourcesManifest
from torch_dae.environment.subprocess import CommandExecutor


class Completed:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class InventoryRunner:
    def run(
        self,
        command: tuple[str, ...],
        *,
        cwd: Path | None,
        env: dict[str, str],
        timeout: float | None,
    ) -> Completed:
        rows = [{"name": "synthetic-package", "version": "1.2.3", "location": "/tmp/pkg"}]
        return Completed(json.dumps(rows))


class GitSourceRunner:
    def __init__(
        self,
        *,
        url: str,
        revision: str,
        initial_url: str | None = None,
        initial_revision: str | None = None,
        initial_status: str = "",
    ) -> None:
        self.url = url
        self.revision = revision
        self.initial_url = initial_url
        self.initial_revision = initial_revision
        self.initial_status = initial_status
        self.commands: list[tuple[str, ...]] = []
        self.builds = 0
        self.cloned = False

    def run(
        self,
        command: tuple[str, ...],
        *,
        cwd: Path | None,
        env: dict[str, str],
        timeout: float | None,
    ) -> Completed:
        self.commands.append(command)
        if command[:2] == ("git", "clone"):
            Path(command[-1]).mkdir(parents=True, exist_ok=True)
            self.cloned = True
            return Completed()
        if command[:3] == ("git", "checkout", "--detach"):
            return Completed()
        if command[:3] == ("git", "rev-parse", "HEAD"):
            if not self.cloned and self.initial_revision is not None:
                return Completed(self.initial_revision + "\n")
            return Completed(self.revision + "\n")
        if command[:4] == ("git", "remote", "get-url", "origin"):
            if not self.cloned and self.initial_url is not None:
                return Completed(self.initial_url + "\n")
            return Completed(self.url + "\n")
        if command[:3] == ("git", "status", "--porcelain"):
            if not self.cloned and self.initial_status:
                return Completed(self.initial_status)
            return Completed()
        if command[:2] == ("git", "archive"):
            output = Path(command[command.index("--output") + 1])
            output.write_bytes(b"synthetic tar")
            return Completed()
        if command[:2] == ("tar", "-xf"):
            return Completed()
        if command[:3] == ("uv", "build", "--wheel"):
            self.builds += 1
            out_dir = Path(command[command.index("--out-dir") + 1])
            make_distribution_wheel(out_dir / "fixture_pkg-0.0.1-py3-none-any.whl")
            return Completed()
        if command[:3] == ("uv", "pip", "install"):
            return Completed()
        if command[1:3] == ("-c", INSTALLED_DISTRIBUTIONS_SENTINEL):
            rows = [{"name": "fixture-pkg", "version": "0.0.1", "location": "/tmp/fixture"}]
            return Completed(json.dumps(rows))
        if command[1] == "-c" and "importlib.metadata as metadata" in command[2]:
            rows = [{"name": "fixture-pkg", "version": "0.0.1", "location": "/tmp/fixture"}]
            return Completed(json.dumps(rows))
        raise ExternalCommandError(f"unexpected command: {command}")


INSTALLED_DISTRIBUTIONS_SENTINEL = "import importlib.metadata as metadata"


def source_context(root: Path) -> SourceContext:
    lock = root / "uv.lock"
    lock.write_text(
        """
version = 1
revision = 3

[[package]]
name = "synthetic-package"
version = "1.2.3"
""".lstrip()
    )
    wheel = root / "local-wheel/torch_dae-0.1.0-py3-none-any.whl"
    wheel.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("torch_dae/vendor/synthetic.py", b"VALUE = 1\n")
    return SourceContext(
        repository_root=root,
        runtime_root=root / ".torch-dae",
        environment_root=root / ".torch-dae/environments/card/hash",
        python_executable=Path("/usr/bin/python3"),
        lockfile_path=lock,
        lockfile_sha256="a" * 64,
        python_version="3.12.0",
        platform="synthetic-platform",
        local_package_wheel_path=wheel,
        local_package_wheel_sha256=sha256_path(wheel),
        local_package_identity="synthetic-local-package",
    )


def test_phase01_package_source_verifies_lock_and_installed_distribution(tmp_path: Path) -> None:
    manifest = EnvironmentSourcesManifest.model_validate(
        {
            "schema_version": "1.0.0",
            "environment_id": "card",
            "sources": [
                {
                    "source_id": "package-source",
                    "role": "model_implementation",
                    "installation": "package",
                    "package": "synthetic-package",
                    "version": "1.2.3",
                }
            ],
        }
    )
    records = SourceManager(executor=CommandExecutor(InventoryRunner())).materialize(
        manifest,
        source_context(tmp_path),
    )
    assert records[0].source_id == "package-source"
    assert records[0].version == "1.2.3"


def test_phase01_vendored_source_hashes_declared_files(tmp_path: Path) -> None:
    vendored = tmp_path / "src/torch_dae/vendor/synthetic.py"
    vendored.parent.mkdir(parents=True)
    vendored.write_text("VALUE = 1\n")
    manifest = EnvironmentSourcesManifest.model_validate(
        {
            "schema_version": "1.0.0",
            "environment_id": "card",
            "sources": [
                {
                    "source_id": "vendored-source",
                    "role": "model_implementation",
                    "installation": "vendored",
                    "upstream_url": "https://example.invalid/repo",
                    "upstream_revision": "a" * 40,
                    "copied_files": ["src/torch_dae/vendor/synthetic.py"],
                    "adaptation_description": "Synthetic vendored fixture.",
                    "justification": "Synthetic test only.",
                }
            ],
        }
    )
    records = SourceManager().materialize(manifest, source_context(tmp_path))
    assert records[0].file_hashes["src/torch_dae/vendor/synthetic.py"]


def test_phase01_git_source_offline_cache_miss_fails(tmp_path: Path) -> None:
    manifest = EnvironmentSourcesManifest.model_validate(
        {
            "schema_version": "1.0.0",
            "environment_id": "card",
            "sources": [
                {
                    "source_id": "git-source",
                    "role": "model_implementation",
                    "installation": "git",
                    "url": str(tmp_path / "missing.git"),
                    "revision": "b" * 40,
                    "build": "wheel",
                }
            ],
        }
    )
    manager = SourceManager(policy=ExecutionPolicy(offline=True))
    with pytest.raises(OfflineResourceUnavailableError):
        manager.materialize(manifest, source_context(tmp_path))


def test_phase01_git_source_build_metadata_workspace_and_offline_reuse(
    tmp_path: Path,
) -> None:
    url = "file:///tmp/synthetic-git-source"
    revision = "b" * 40
    manifest = EnvironmentSourcesManifest.model_validate(
        {
            "schema_version": "1.0.0",
            "environment_id": "card",
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
    )
    context = source_context(tmp_path)
    runner = GitSourceRunner(url=url, revision=revision)
    records = SourceManager(executor=CommandExecutor(runner)).materialize(manifest, context)
    assert records[0].revision == revision
    assert runner.builds == 1
    build_root = tmp_path / ".torch-dae/source-builds/git-source"
    metadata = next(build_root.glob("*/source-wheel.json"))
    data = json.loads(metadata.read_text())
    assert data["source_url"] == url
    assert data["source_revision"] == revision
    assert data["wheel_sha256"] == records[0].wheel_sha256
    assert not (metadata.parent / "workspace").exists()
    assert any(command[:2] == ("git", "archive") for command in runner.commands)
    clone_commands = [command for command in runner.commands if command[:2] == ("git", "clone")]
    assert clone_commands
    assert ".clone-" in clone_commands[0][-1]
    assert clone_commands[0][-1] != str(tmp_path / ".torch-dae/repositories/git-source" / revision)

    offline_runner = GitSourceRunner(url=url, revision=revision)
    offline = SourceManager(
        executor=CommandExecutor(offline_runner),
        policy=ExecutionPolicy(offline=True),
    )
    assert offline.materialize(manifest, context)[0].wheel_sha256 == records[0].wheel_sha256
    assert offline_runner.builds == 0

    metadata.write_text("{bad json")
    with pytest.raises(OfflineResourceUnavailableError):
        offline.materialize(manifest, context)
    recovery_runner = GitSourceRunner(url=url, revision=revision)
    SourceManager(executor=CommandExecutor(recovery_runner)).materialize(manifest, context)
    assert recovery_runner.builds == 1


@pytest.mark.parametrize(
    "runner_kwargs",
    [
        {"initial_status": " M pyproject.toml\n"},
        {"initial_revision": "c" * 40},
        {"initial_url": "file:///tmp/wrong-origin"},
    ],
)
def test_phase01_git_source_online_recovers_invalid_checkout(
    tmp_path: Path,
    runner_kwargs: dict[str, str],
) -> None:
    url = "file:///tmp/synthetic-git-source"
    revision = "b" * 40
    checkout = tmp_path / ".torch-dae/repositories/git-source" / revision
    checkout.mkdir(parents=True)
    (checkout / "dirty.txt").write_text("invalid cached checkout\n")
    manifest = EnvironmentSourcesManifest.model_validate(
        {
            "schema_version": "1.0.0",
            "environment_id": "card",
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
    )
    runner = GitSourceRunner(url=url, revision=revision, **runner_kwargs)
    records = SourceManager(executor=CommandExecutor(runner)).materialize(
        manifest,
        source_context(tmp_path),
    )

    assert records[0].revision == revision
    assert not (checkout / "dirty.txt").exists()
    assert any(command[:2] == ("git", "clone") for command in runner.commands)


def test_phase01_git_source_offline_invalid_checkout_fails_without_mutation(tmp_path: Path) -> None:
    url = "file:///tmp/synthetic-git-source"
    revision = "b" * 40
    checkout = tmp_path / ".torch-dae/repositories/git-source" / revision
    checkout.mkdir(parents=True)
    marker = checkout / "dirty.txt"
    marker.write_text("invalid cached checkout\n")
    manifest = EnvironmentSourcesManifest.model_validate(
        {
            "schema_version": "1.0.0",
            "environment_id": "card",
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
    )
    runner = GitSourceRunner(url=url, revision=revision, initial_status=" M pyproject.toml\n")
    with pytest.raises(OfflineResourceUnavailableError):
        SourceManager(
            executor=CommandExecutor(runner),
            policy=ExecutionPolicy(offline=True),
        ).materialize(manifest, source_context(tmp_path))

    assert marker.read_text() == "invalid cached checkout\n"
    assert not any(command[:2] == ("git", "clone") for command in runner.commands)


@pytest.mark.integration
def test_phase01_git_source_real_local_repository_installs_and_reuses_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv-cache"))
    source_repo = tmp_path / "upstream"
    source_repo.mkdir()
    (source_repo / "pyproject.toml").write_text(
        """
[build-system]
requires = []
build-backend = "backend"
backend-path = ["."]

[project]
name = "synthetic-git-package"
version = "0.1.0"
""".lstrip()
    )
    (source_repo / "backend.py").write_text(MINIMAL_WHEEL_BACKEND)
    (source_repo / "synthetic_git_package.py").write_text("VALUE = 'from git wheel'\n")
    run_git_command(source_repo, "init")
    run_git_command(source_repo, "add", ".")
    run_git_command(
        source_repo,
        "-c",
        "user.name=Test User",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "initial",
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (source_repo / "synthetic_git_package.py").write_text("VALUE = 'second commit'\n")
    run_git_command(source_repo, "add", ".")
    run_git_command(
        source_repo,
        "-c",
        "user.name=Test User",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "second",
    )
    other_revision = run_git_text(source_repo, "rev-parse", "HEAD")
    env_root = tmp_path / "model-env"
    subprocess.run(
        ["uv", "venv", str(env_root), "--python", sys.executable],
        check=True,
        capture_output=True,
        text=True,
    )
    python_executable = env_root / "bin/python"
    lock = tmp_path / "uv.lock"
    lock.write_text("version = 1\nrevision = 3\n")
    local_wheel = tmp_path / "local-wheel/torch_dae-0.1.0-py3-none-any.whl"
    local_wheel.parent.mkdir(parents=True)
    make_distribution_wheel(local_wheel)
    context = SourceContext(
        repository_root=tmp_path,
        runtime_root=tmp_path / ".torch-dae",
        environment_root=env_root,
        python_executable=python_executable,
        lockfile_path=lock,
        lockfile_sha256="a" * 64,
        python_version=".".join(str(part) for part in sys.version_info[:3]),
        platform="synthetic-platform",
        local_package_wheel_path=local_wheel,
        local_package_wheel_sha256=sha256_path(local_wheel),
        local_package_identity="synthetic-local-package",
    )
    manifest = EnvironmentSourcesManifest.model_validate(
        {
            "schema_version": "1.0.0",
            "environment_id": "card",
            "sources": [
                {
                    "source_id": "git-source",
                    "role": "model_implementation",
                    "installation": "git",
                    "url": str(source_repo),
                    "revision": revision,
                    "build": "wheel",
                }
            ],
        }
    )

    record = SourceManager().materialize(manifest, context)[0]
    assert record.revision == revision
    checkout = Path(record.location)
    assert run_git_text(checkout, "rev-parse", "HEAD") == revision
    assert run_git_text(checkout, "status", "--porcelain") == ""
    result = subprocess.run(
        [
            str(python_executable),
            "-c",
            "import synthetic_git_package; print(synthetic_git_package.VALUE)",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=clean_python_env(),
    )
    assert result.stdout.strip() == "from git wheel"
    metadata = next((tmp_path / ".torch-dae/source-builds/git-source").glob("*/source-wheel.json"))
    assert json.loads(metadata.read_text())["source_revision"] == revision
    assert not (metadata.parent / "workspace").exists()

    offline_record = SourceManager(policy=ExecutionPolicy(offline=True)).materialize(
        manifest,
        context,
    )[0]
    assert offline_record.wheel_sha256 == record.wheel_sha256

    (checkout / "dirty.txt").write_text("dirty checkout\n")
    recovered_dirty = SourceManager().materialize(manifest, context)[0]
    assert recovered_dirty.revision == revision
    assert run_git_text(Path(recovered_dirty.location), "status", "--porcelain") == ""

    checkout = Path(recovered_dirty.location)
    dirty_marker = checkout / "dirty-offline.txt"
    dirty_marker.write_text("dirty offline\n")
    with pytest.raises(OfflineResourceUnavailableError):
        SourceManager(policy=ExecutionPolicy(offline=True)).materialize(manifest, context)
    assert dirty_marker.read_text() == "dirty offline\n"
    SourceManager().materialize(manifest, context)
    checkout = tmp_path / ".torch-dae/repositories/git-source" / revision

    run_git_command(checkout, "checkout", "--detach", other_revision)
    recovered_head = SourceManager().materialize(manifest, context)[0]
    assert run_git_text(Path(recovered_head.location), "rev-parse", "HEAD") == revision

    run_git_command(Path(recovered_head.location), "remote", "set-url", "origin", str(tmp_path))
    recovered_remote = SourceManager().materialize(manifest, context)[0]
    assert run_git_text(Path(recovered_remote.location), "remote", "get-url", "origin") == str(
        source_repo
    )

    metadata.write_text("{bad json")
    recovered_metadata = SourceManager().materialize(manifest, context)[0]
    assert recovered_metadata.revision == revision
    assert json.loads(metadata.read_text())["source_revision"] == revision

    metadata.write_text("{bad json")
    with pytest.raises(OfflineResourceUnavailableError):
        SourceManager(policy=ExecutionPolicy(offline=True)).materialize(manifest, context)
    assert metadata.read_text() == "{bad json"


def run_git_command(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)


def run_git_text(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def clean_python_env() -> dict[str, str]:
    import os

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return env


MINIMAL_WHEEL_BACKEND = r"""
from __future__ import annotations

import zipfile
from pathlib import Path


def get_requires_for_build_wheel(config_settings=None):
    return []


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    wheel = Path(wheel_directory) / "synthetic_git_package-0.1.0-py3-none-any.whl"
    dist_info = "synthetic_git_package-0.1.0.dist-info"
    members = {
        "synthetic_git_package.py": Path("synthetic_git_package.py").read_text(),
        f"{dist_info}/METADATA": (
            "Metadata-Version: 2.1\n"
            "Name: synthetic-git-package\n"
            "Version: 0.1.0\n"
        ),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: synthetic\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ),
    }
    record = "".join(f"{name},,\n" for name in [*members, f"{dist_info}/RECORD"])
    members[f"{dist_info}/RECORD"] = record
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return wheel.name
"""


def test_phase01_package_source_rejects_lock_mismatch(tmp_path: Path) -> None:
    manifest = EnvironmentSourcesManifest.model_validate(
        {
            "schema_version": "1.0.0",
            "environment_id": "card",
            "sources": [
                {
                    "source_id": "package-source",
                    "role": "model_implementation",
                    "installation": "package",
                    "package": "missing-package",
                    "version": "1.2.3",
                }
            ],
        }
    )
    with pytest.raises(SourceMaterializationError, match="lock file"):
        SourceManager(executor=CommandExecutor(InventoryRunner())).materialize(
            manifest,
            source_context(tmp_path),
        )


def test_phase01_package_source_rejects_installed_mismatch(tmp_path: Path) -> None:
    manifest = EnvironmentSourcesManifest.model_validate(
        {
            "schema_version": "1.0.0",
            "environment_id": "card",
            "sources": [
                {
                    "source_id": "package-source",
                    "role": "model_implementation",
                    "installation": "package",
                    "package": "synthetic-package",
                    "version": "9.9.9",
                }
            ],
        }
    )
    context = source_context(tmp_path)
    context.lockfile_path.write_text(
        context.lockfile_path.read_text().replace('version = "1.2.3"', 'version = "9.9.9"')
    )
    with pytest.raises(SourceMaterializationError, match="environment"):
        SourceManager(executor=CommandExecutor(InventoryRunner())).materialize(manifest, context)


def test_phase01_vendored_source_rejects_missing_file_and_empty_notes(tmp_path: Path) -> None:
    base = {
        "schema_version": "1.0.0",
        "environment_id": "card",
        "sources": [
            {
                "source_id": "vendored-source",
                "role": "model_implementation",
                "installation": "vendored",
                "upstream_url": "https://example.invalid/repo",
                "upstream_revision": "a" * 40,
                "copied_files": ["src/torch_dae/vendor/missing.py"],
                "adaptation_description": "Synthetic vendored fixture.",
                "justification": "Synthetic test only.",
            }
        ],
    }
    with pytest.raises(SourceMaterializationError, match="missing"):
        SourceManager().materialize(
            EnvironmentSourcesManifest.model_validate(base),
            source_context(tmp_path),
        )
    base["sources"][0]["adaptation_description"] = " "
    with pytest.raises(SourceMaterializationError, match="empty"):
        SourceManager().materialize(
            EnvironmentSourcesManifest.model_validate(base),
            source_context(tmp_path),
        )


def sha256_path(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_distribution_wheel(path: Path) -> None:
    dist_info = "fixture_pkg-0.0.1.dist-info"
    metadata = b"Metadata-Version: 2.3\nName: fixture-pkg\nVersion: 0.0.1\n"
    wheel = b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    record = f"{dist_info}/METADATA,,\n{dist_info}/WHEEL,,\n{dist_info}/RECORD,,\n".encode()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{dist_info}/METADATA", metadata)
        archive.writestr(f"{dist_info}/WHEEL", wheel)
        archive.writestr(f"{dist_info}/RECORD", record)
