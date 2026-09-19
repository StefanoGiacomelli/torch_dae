"""Minimal cross-platform inter-process locks for managed runtime caches."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType


class ManagedLockTimeoutError(TimeoutError):
    """Raised when a managed cache lock cannot be acquired within its bound."""


@dataclass(frozen=True)
class ManagedLockOwner:
    """Sanitized ownership metadata stored inside an acquired lock directory."""

    schema_version: str
    resource_id: str
    owner_token: str
    pid: int
    host_sha256: str
    created_at_unix: float


class ManagedDirectoryLock:
    """Coordinate processes using atomic directory creation.

    The lock uses only Python and filesystem primitives available on every supported host. A stale
    lock is reclaimed only when it belongs to this host and its process is no longer live, or when
    old ownership metadata is absent or invalid. Locks attributed to another host are never broken.
    """

    def __init__(
        self,
        lock_path: Path,
        *,
        resource_id: str,
        timeout_seconds: float,
        stale_after_seconds: float,
        poll_interval_seconds: float = 0.05,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("lock timeout must be positive")
        if stale_after_seconds <= 0:
            raise ValueError("stale-lock threshold must be positive")
        if poll_interval_seconds <= 0:
            raise ValueError("lock poll interval must be positive")
        self.lock_path = lock_path
        self.resource_id = resource_id
        self.timeout_seconds = timeout_seconds
        self.stale_after_seconds = stale_after_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self._host_sha256 = hashlib.sha256(socket.gethostname().encode("utf-8")).hexdigest()
        self._owner_token: str | None = None

    def __enter__(self) -> ManagedDirectoryLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

    def acquire(self) -> None:
        """Acquire the lock or raise a bounded, path-free timeout error."""

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            token = uuid.uuid4().hex
            try:
                self.lock_path.mkdir()
            except FileExistsError:
                if self._reclaim_stale_lock():
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ManagedLockTimeoutError(
                        f"timed out waiting for managed cache lock {self.resource_id}"
                    ) from None
                time.sleep(min(self.poll_interval_seconds, remaining))
                continue
            owner = ManagedLockOwner(
                schema_version="1.0.0",
                resource_id=self.resource_id,
                owner_token=token,
                pid=os.getpid(),
                host_sha256=self._host_sha256,
                created_at_unix=time.time(),
            )
            try:
                self._write_owner(owner)
            except Exception:
                shutil.rmtree(self.lock_path, ignore_errors=True)
                raise
            self._owner_token = token
            return

    def release(self) -> None:
        """Release only the lock still carrying this instance's ownership token."""

        token = self._owner_token
        self._owner_token = None
        if token is None:
            return
        owner = self._read_owner()
        if owner is None or owner.owner_token != token:
            return
        retired = self.lock_path.with_name(f".{self.lock_path.name}.released-{uuid.uuid4().hex}")
        try:
            self.lock_path.rename(retired)
        except FileNotFoundError:
            return
        shutil.rmtree(retired, ignore_errors=True)

    def _write_owner(self, owner: ManagedLockOwner) -> None:
        metadata = self.lock_path / "owner.json"
        temporary = self.lock_path / f".owner-{uuid.uuid4().hex}.tmp"
        temporary.write_text(
            json.dumps(asdict(owner), sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, metadata)

    def _read_owner(self) -> ManagedLockOwner | None:
        try:
            data = json.loads((self.lock_path / "owner.json").read_text(encoding="utf-8"))
            owner = ManagedLockOwner(**data)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if owner.schema_version != "1.0.0" or owner.resource_id != self.resource_id:
            return None
        return owner

    def _reclaim_stale_lock(self) -> bool:
        try:
            directory_age = max(0.0, time.time() - self.lock_path.stat().st_mtime)
        except FileNotFoundError:
            return True
        owner = self._read_owner()
        if owner is None:
            stale = directory_age >= self.stale_after_seconds
        else:
            owner_age = max(0.0, time.time() - owner.created_at_unix)
            stale = (
                owner_age >= self.stale_after_seconds
                and owner.host_sha256 == self._host_sha256
                and not _process_is_live(owner.pid)
            )
        if not stale:
            return False
        retired = self.lock_path.with_name(f".{self.lock_path.name}.stale-{uuid.uuid4().hex}")
        try:
            self.lock_path.rename(retired)
        except FileNotFoundError:
            return True
        except OSError:
            return False
        shutil.rmtree(retired, ignore_errors=True)
        return True


def _process_is_live(pid: int) -> bool:
    """Return whether a same-host process appears live without acquiring new privileges."""

    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True
