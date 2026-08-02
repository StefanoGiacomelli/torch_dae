from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from queue import Empty
from typing import Any

import pytest

from torch_dae.environment.fingerprint import local_package_identity
from torch_dae.environment.locking import ManagedDirectoryLock, ManagedLockTimeoutError
from torch_dae.environment.manager import EnvironmentManager, local_wheel_cache_key
from torch_dae.environment.policy import ExecutionPolicy
from torch_dae.environment.subprocess import CommandExecutor, ManagedProcessResult


class CountingBuildExecutor(CommandExecutor):
    """Record authoritative builds and optionally fail the first one across all processes."""

    def __init__(
        self,
        counter_path: Path,
        *,
        fail_once_path: Path | None = None,
        rendezvous: bool = False,
    ) -> None:
        super().__init__()
        self.counter_path = counter_path
        self.fail_once_path = fail_once_path
        self.rendezvous = rendezvous

    def run(
        self,
        command: Sequence[str],
        *,
        operation: str | None = None,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        env_remove: Sequence[str] = (),
        timeout: float | None = None,
        check: bool = False,
    ) -> ManagedProcessResult:
        if operation == "local-wheel-build":
            with self.counter_path.open("a", encoding="utf-8") as stream:
                stream.write(f"{os.getpid()}\n")
                stream.flush()
                os.fsync(stream.fileno())
            if self.fail_once_path is not None:
                try:
                    descriptor = os.open(
                        self.fail_once_path,
                        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    )
                except FileExistsError:
                    pass
                else:
                    os.close(descriptor)
                    raise RuntimeError("deliberate first-builder failure")
            if self.rendezvous:
                deadline = time.monotonic() + 10
                while len(self.counter_path.read_text(encoding="utf-8").splitlines()) < 2:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("identity-scoped builds were unnecessarily serialized")
                    time.sleep(0.02)
        return super().run(
            command,
            operation=operation,
            cwd=cwd,
            env=env,
            env_remove=env_remove,
            timeout=timeout,
            check=check,
        )


def _wheel_worker(
    repository_root: str,
    start_event: Any,
    result_queue: Any,
    counter_path: str,
    fail_once_path: str | None = None,
    rendezvous: bool = False,
    identity_suffix: str = "",
) -> None:
    root = Path(repository_root)
    executor = CountingBuildExecutor(
        Path(counter_path),
        fail_once_path=Path(fail_once_path) if fail_once_path else None,
        rendezvous=rendezvous,
    )
    manager = EnvironmentManager(
        root,
        executor=executor,
        policy=ExecutionPolicy(command_timeout_seconds=60),
        wheel_lock_timeout_seconds=20,
        wheel_lock_stale_seconds=60,
        wheel_lock_poll_seconds=0.02,
    )
    start_event.wait(20)
    try:
        identity = local_package_identity(root) + identity_suffix
        wheel, digest = manager._build_local_wheel(identity)
        result_queue.put(
            {
                "status": "passed",
                "path": wheel.relative_to(root).as_posix(),
                "sha256": digest,
                "identity": identity,
            }
        )
    except Exception as exc:
        result_queue.put({"status": "failed", "error": str(exc)})


def _copy_package_repository(target: Path, repository_root: Path) -> None:
    (target / "src").mkdir(parents=True)
    shutil.copytree(repository_root / "src/torch_dae", target / "src/torch_dae")
    shutil.copy2(repository_root / "pyproject.toml", target / "pyproject.toml")
    shutil.copy2(repository_root / "README.md", target / "README.md")
    (target / "project_spec.md").write_text("synthetic local-wheel concurrency repository\n")


def _run_workers(
    root: Path,
    counter: Path,
    *,
    count: int = 2,
    fail_once: Path | None = None,
    rendezvous: bool = False,
    identity_suffixes: tuple[str, ...] | None = None,
) -> list[dict[str, str]]:
    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    result_queue = context.Queue()
    suffixes = identity_suffixes or tuple("" for _ in range(count))
    processes = [
        context.Process(
            target=_wheel_worker,
            args=(
                str(root),
                start_event,
                result_queue,
                str(counter),
                str(fail_once) if fail_once else None,
                rendezvous,
                suffixes[index],
            ),
        )
        for index in range(count)
    ]
    for process in processes:
        process.start()
    start_event.set()
    results: list[dict[str, str]] = []
    for _ in processes:
        try:
            results.append(result_queue.get(timeout=90))
        except Empty as exc:
            raise AssertionError("wheel worker did not return a bounded result") from exc
    for process in processes:
        process.join(timeout=10)
        assert not process.is_alive()
        assert process.exitcode == 0
    return results


@pytest.mark.integration
def test_concurrent_same_identity_builds_once_and_reuses_atomic_cache(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    _copy_package_repository(root, repo_root)
    counter = tmp_path / "build-count.txt"
    counter.write_text("")

    results = _run_workers(root, counter)

    assert [item["status"] for item in results] == ["passed", "passed"]
    assert len(counter.read_text().splitlines()) == 1
    assert len({item["sha256"] for item in results}) == 1
    assert len({item["path"] for item in results}) == 1
    wheel_path = root / results[0]["path"]
    assert wheel_path.is_file()
    cache_root = wheel_path.parent.parent
    assert not list(cache_root.glob(".*.build-*"))
    assert not list(cache_root.glob(".*.lock"))

    original_bytes = wheel_path.read_bytes()
    reuse_counter = tmp_path / "reuse-build-count.txt"
    reuse_counter.write_text("")
    reused = _run_workers(root, reuse_counter, count=1)
    assert reused[0]["status"] == "passed"
    assert reuse_counter.read_text() == ""
    assert wheel_path.read_bytes() == original_bytes


@pytest.mark.integration
def test_waiter_recovers_after_failed_builder_without_partial_publication(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    _copy_package_repository(root, repo_root)
    counter = tmp_path / "build-count.txt"
    counter.write_text("")
    fail_once = tmp_path / "fail-once"

    results = _run_workers(root, counter, fail_once=fail_once)

    assert sorted(item["status"] for item in results) == ["failed", "passed"]
    assert len(counter.read_text().splitlines()) == 2
    passed = next(item for item in results if item["status"] == "passed")
    wheel_path = root / passed["path"]
    assert wheel_path.is_file()
    cache_root = wheel_path.parent.parent
    assert not list(cache_root.glob(".*.build-*"))
    assert not list(cache_root.glob(".*.lock"))
    assert (
        json.loads((wheel_path.parent / "wheel.json").read_text())["wheel_sha256"]
        == passed["sha256"]
    )


def test_stale_lock_recovery_and_live_lock_timeout_are_bounded(tmp_path: Path) -> None:
    lock_path = tmp_path / ".identity.lock"
    lock_path.mkdir()
    (lock_path / "owner.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "resource_id": "identity",
                "owner_token": "abandoned",
                "pid": 999_999_999,
                "host_sha256": ManagedDirectoryLock(
                    tmp_path / "unused",
                    resource_id="identity",
                    timeout_seconds=1,
                    stale_after_seconds=1,
                )._host_sha256,
                "created_at_unix": 0,
            }
        )
    )
    with ManagedDirectoryLock(
        lock_path,
        resource_id="identity",
        timeout_seconds=1,
        stale_after_seconds=0.01,
        poll_interval_seconds=0.01,
    ):
        assert lock_path.is_dir()
    assert not lock_path.exists()

    holder = ManagedDirectoryLock(
        lock_path,
        resource_id="identity",
        timeout_seconds=1,
        stale_after_seconds=60,
        poll_interval_seconds=0.01,
    )
    holder.acquire()
    started = time.monotonic()
    try:
        with pytest.raises(ManagedLockTimeoutError, match="identity"):
            ManagedDirectoryLock(
                lock_path,
                resource_id="identity",
                timeout_seconds=0.1,
                stale_after_seconds=60,
                poll_interval_seconds=0.01,
            ).acquire()
    finally:
        holder.release()
    assert time.monotonic() - started < 1


@pytest.mark.integration
def test_different_wheel_identities_build_without_global_serialization(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    _copy_package_repository(root, repo_root)
    counter = tmp_path / "build-count.txt"
    counter.write_text("")

    results = _run_workers(
        root,
        counter,
        rendezvous=True,
        identity_suffixes=(":first", ":second"),
    )

    assert [item["status"] for item in results] == ["passed", "passed"]
    assert len(counter.read_text().splitlines()) == 2
    assert len({item["path"] for item in results}) == 2
    assert {Path(item["path"]).parent.name for item in results} == {
        local_wheel_cache_key(item["identity"]) for item in results
    }
