from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from torch_dae.profiling.raw_assets import (
    build_manifest,
    read_raw_npz,
    validate_manifest_matches_file,
    write_raw_npz,
)


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    arrays = {"raw_ns__b1": np.arange(50, dtype=np.int64)}
    path = tmp_path / "evidence.npz"
    write_raw_npz(path, arrays)
    read_back = read_raw_npz(path)
    assert np.array_equal(read_back["raw_ns__b1"], arrays["raw_ns__b1"])


def test_rejects_object_dtype(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_raw_npz(tmp_path / "bad.npz", {"bad": np.array([object()], dtype=object)})


def test_manifest_matches_file(tmp_path: Path) -> None:
    arrays = {"raw_ns__b1": np.arange(50, dtype=np.int64)}
    path = tmp_path / "evidence.npz"
    write_raw_npz(path, arrays)
    manifest = build_manifest(card_relative_path="tc-abc.npz", path=path, arrays=arrays)
    validate_manifest_matches_file(manifest, path)  # must not raise


def test_manifest_detects_sha256_mismatch(tmp_path: Path) -> None:
    arrays = {"raw_ns__b1": np.arange(50, dtype=np.int64)}
    path = tmp_path / "evidence.npz"
    write_raw_npz(path, arrays)
    manifest = build_manifest(card_relative_path="tc-abc.npz", path=path, arrays=arrays)
    write_raw_npz(path, {"raw_ns__b1": np.arange(51, dtype=np.int64)})
    with pytest.raises(ValueError, match="SHA-256"):
        validate_manifest_matches_file(manifest, path)


def test_manifest_detects_missing_file(tmp_path: Path) -> None:
    arrays = {"raw_ns__b1": np.arange(50, dtype=np.int64)}
    path = tmp_path / "evidence.npz"
    write_raw_npz(path, arrays)
    manifest = build_manifest(card_relative_path="tc-abc.npz", path=path, arrays=arrays)
    path.unlink()
    with pytest.raises(ValueError, match="missing"):
        validate_manifest_matches_file(manifest, path)


def test_npz_never_uses_pickle(tmp_path: Path) -> None:
    arrays = {"raw_ns__b1": np.arange(50, dtype=np.int64)}
    path = tmp_path / "evidence.npz"
    write_raw_npz(path, arrays)
    # allow_pickle=False must succeed; a pickled object array would fail this load.
    with np.load(path, allow_pickle=False) as data:
        assert "raw_ns__b1" in data.files
