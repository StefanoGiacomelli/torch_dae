"""Regression test: profiling must reuse an already-valid cached checkpoint read-only.

`CheckpointManager.ensure_checkpoint`'s strict specification-fingerprint cache check can force a
network re-download even when the cached payload bytes are already correct, rewriting
`checkpoint-materialization.json` with fresh non-deterministic provenance and invalidating the
byte-pinned local cross-check that already-accepted `verification_reports/*.json` perform against
it (see CHANGELOG.md). `torch_dae.profiling_executor._resolve_checkpoint_read_only` must find and
reuse such a cache entry without touching it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from torch_dae.cards.models import ModelCard
from torch_dae.profiling_executor import _resolve_checkpoint_read_only


def _synthetic_model_card(
    *, repo_root: Path, checkpoint_id: str, sha256: str, filename: str
) -> ModelCard:
    """Start from the real accepted PANNs card and swap in a synthetic checkpoint identity."""

    base = json.loads((repo_root / "model_cards/panns/panns-cnn14-16k-map-0438.json").read_text())
    base["card_id"] = checkpoint_id
    base["checkpoint"]["checkpoint_id"] = checkpoint_id
    base["checkpoint"]["filename"] = filename
    base["checkpoint"]["observed_sha256"] = sha256
    base["checkpoint"]["authority"]["filename"] = filename
    return ModelCard.model_validate(base)


def test_reuses_valid_cache_entry_without_mutation(tmp_path: Path, repo_root: Path) -> None:
    checkpoint_id = "synthetic-checkpoint"
    filename = "weights.pth"
    payload_bytes = b"synthetic checkpoint payload bytes"
    sha256 = hashlib.sha256(payload_bytes).hexdigest()

    cache_dir = tmp_path / ".torch-dae" / "checkpoints" / checkpoint_id / sha256
    cache_dir.mkdir(parents=True)
    payload_path = cache_dir / filename
    payload_path.write_bytes(payload_bytes)
    materialization_path = cache_dir / "checkpoint-materialization.json"
    materialization_path.write_text('{"sentinel": "must-not-change"}')
    before_mtime = materialization_path.stat().st_mtime_ns

    model_card = _synthetic_model_card(
        repo_root=repo_root, checkpoint_id=checkpoint_id, sha256=sha256, filename=filename
    )

    resolved = _resolve_checkpoint_read_only(tmp_path, model_card, checkpoint_id)

    assert resolved.path == payload_path
    assert resolved.sha256 == sha256
    # The unrelated materialization record must remain byte-identical: no cache mutation.
    assert materialization_path.stat().st_mtime_ns == before_mtime
    assert materialization_path.read_text() == '{"sentinel": "must-not-change"}'


def test_falls_back_to_manager_when_no_local_cache_hit(tmp_path: Path, repo_root: Path) -> None:
    from unittest.mock import MagicMock, patch

    checkpoint_id = "synthetic-checkpoint-missing"
    model_card = _synthetic_model_card(
        repo_root=repo_root,
        checkpoint_id=checkpoint_id,
        sha256="0" * 64,
        filename="weights.pth",
    )

    fake_resolved = MagicMock(path=tmp_path / "fallback.pth", sha256="0" * 64)
    with patch("torch_dae.profiling_executor.CheckpointManager") as manager_cls:
        manager_cls.return_value.ensure_checkpoint.return_value = fake_resolved
        resolved = _resolve_checkpoint_read_only(tmp_path, model_card, checkpoint_id)

    manager_cls.return_value.ensure_checkpoint.assert_called_once()
    assert resolved.path == fake_resolved.path
    assert resolved.sha256 == fake_resolved.sha256
