"""Regression coverage for the corrective-task fix to repository-validator checkpoint evidence
resolution.

Root cause of the previously observed "materialization evidence SHA-256 mismatch" repository
validation failure: an accepted `runtime-verification.json` (or Model Card) embeds a
`checkpoint_materialization.path` pointing into the ignored, mutable `.torch-dae/` runtime cache
(`CLAUDE.md`: "Treat `.torch-dae/` as ignored runtime state"). That cache is legitimately allowed
to be refreshed (e.g. a later, unrelated checkpoint re-acquisition for the same checkpoint id/hash
rewrites its `acquired_at` timestamp and per-request log-correlation ids) without invalidating
already-accepted evidence. The validator must not let that mutable-cache churn fail validation as
long as a byte-identical, durably committed evidence copy still exists under
`onboarding_reports/**/checkpoint-materializations/`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from scripts.validate_repository import _resolve_checkpoint_materialization_evidence


@dataclass
class _StubEvidence:
    path: str
    sha256: str


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_resolves_the_literal_path_when_it_still_matches(tmp_path: Path) -> None:
    live = tmp_path / ".torch-dae/checkpoints/model-x/deadbeef/checkpoint-materialization.json"
    live.parent.mkdir(parents=True)
    live.write_bytes(b'{"acquired_at": "now"}')
    evidence = _StubEvidence(
        path=str(live.relative_to(tmp_path)), sha256=_sha256_bytes(live.read_bytes())
    )
    resolved = _resolve_checkpoint_materialization_evidence(tmp_path, evidence)
    assert resolved == live


def test_falls_back_to_committed_evidence_when_the_runtime_cache_has_drifted(
    tmp_path: Path,
) -> None:
    original_bytes = b'{"acquired_at": "2026-09-15T15:19:51Z", "observed_sha256": "abc"}'
    pinned_sha256 = _sha256_bytes(original_bytes)

    # The .torch-dae cache has since been refreshed with different (but semantically equivalent)
    # content -- e.g. a re-download of the identical payload, which rewrites the timestamp.
    live = tmp_path / ".torch-dae/checkpoints/model-x/deadbeef/checkpoint-materialization.json"
    live.parent.mkdir(parents=True)
    live.write_bytes(b'{"acquired_at": "2026-09-17T09:15:36Z", "observed_sha256": "abc"}')

    # The durable, committed onboarding-evidence copy still carries the original accepted bytes.
    committed = (
        tmp_path
        / "onboarding_reports/some-workflow/verify/checkpoint-materializations/model-x.json"
    )
    committed.parent.mkdir(parents=True)
    committed.write_bytes(original_bytes)

    evidence = _StubEvidence(path=str(live.relative_to(tmp_path)), sha256=pinned_sha256)
    resolved = _resolve_checkpoint_materialization_evidence(tmp_path, evidence)
    assert resolved == committed


def test_returns_none_when_neither_cache_nor_committed_evidence_matches(tmp_path: Path) -> None:
    live = tmp_path / ".torch-dae/checkpoints/model-x/deadbeef/checkpoint-materialization.json"
    live.parent.mkdir(parents=True)
    live.write_bytes(b"{}")
    evidence = _StubEvidence(path=str(live.relative_to(tmp_path)), sha256="f" * 64)
    assert _resolve_checkpoint_materialization_evidence(tmp_path, evidence) is None


def test_does_not_fall_back_when_evidence_path_is_outside_the_ignored_cache(
    tmp_path: Path,
) -> None:
    """Only `.torch-dae/`-rooted evidence paths get the fallback; a mismatch anywhere else
    (e.g. a tracked, supposedly-immutable path) must still fail outright."""

    tracked = tmp_path / "verification_reports/model-x/checkpoint-materialization.json"
    tracked.parent.mkdir(parents=True)
    tracked.write_bytes(b'{"acquired_at": "changed"}')

    committed = (
        tmp_path
        / "onboarding_reports/some-workflow/verify/checkpoint-materializations/model-x.json"
    )
    committed.parent.mkdir(parents=True)
    committed.write_bytes(b'{"acquired_at": "original"}')
    pinned_sha256 = _sha256_bytes(committed.read_bytes())

    evidence = _StubEvidence(path=str(tracked.relative_to(tmp_path)), sha256=pinned_sha256)
    assert _resolve_checkpoint_materialization_evidence(tmp_path, evidence) is None


def test_rejects_a_path_outside_the_repository_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-evidence.json"
    outside.write_bytes(b"{}")
    evidence = _StubEvidence(path="../outside-evidence.json", sha256=_sha256_bytes(b"{}"))
    assert _resolve_checkpoint_materialization_evidence(tmp_path, evidence) is None
