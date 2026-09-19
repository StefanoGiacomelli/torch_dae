"""Compact lossless raw measurement `.npz` assets (Section 26).

No pickle. project_spec.md Section 27.19 ("Raw evidence storage") states raw evidence "MAY
contain the 50 latency observations per successful condition, bounded host-memory samples,
bounded accelerator-memory samples, and bounded energy/power samples when exposed" -- this is
permissive, not mandatory. The current implementation writes only the 50 raw per-condition
latency observations (aggregate-only host-memory/accelerator-memory/energy evidence is carried
in the Technical Card JSON itself, not duplicated as raw sample arrays here); this is a
deliberate, spec-compliant minimal choice, not an omission. No checkpoint payloads, profiler
traces, or unbounded logs are ever written here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from torch_dae.profiling.contracts import RawArrayDescriptor, RawMeasurementManifest


def write_raw_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write bounded arrays losslessly to `.npz`; never uses pickled object arrays."""

    for name, array in arrays.items():
        if array.dtype == object:
            raise ValueError(f"raw array {name!r} must not use dtype=object (would require pickle)")
    path.parent.mkdir(parents=True, exist_ok=True)
    # numpy's stubs mis-resolve `**arrays: dict[str, np.ndarray]` against `savez`'s overloads.
    np.savez(path, **arrays)  # type: ignore[arg-type]


def read_raw_npz(path: Path) -> dict[str, np.ndarray]:
    """Read a raw `.npz` asset with pickle loading strictly disabled."""

    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 of a file's bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    *, card_relative_path: str, path: Path, arrays: dict[str, np.ndarray]
) -> RawMeasurementManifest:
    """Build the Technical Card raw-asset manifest for a written `.npz` file.

    ``card_relative_path`` is relative to the Technical Card JSON's own directory (never to the
    repository root), so a reviewed JSON+`.npz` pair can be relocated together -- e.g. from a
    candidate workspace into `technical_cards/<model-id>/` -- without rewriting the JSON.
    """

    return RawMeasurementManifest(
        path=card_relative_path,
        sha256=sha256_file(path),
        arrays=tuple(
            RawArrayDescriptor(name=name, dtype=str(array.dtype), shape=tuple(array.shape))
            for name, array in sorted(arrays.items())
        ),
    )


def validate_manifest_matches_file(manifest: RawMeasurementManifest, path: Path) -> None:
    """Raise ``ValueError`` if the manifest disagrees with the actual `.npz` file on disk."""

    if not path.is_file():
        raise ValueError(f"raw asset missing: {path}")
    observed_sha256 = sha256_file(path)
    if observed_sha256 != manifest.sha256:
        raise ValueError(
            f"raw asset SHA-256 mismatch: manifest={manifest.sha256} observed={observed_sha256}"
        )
    arrays = read_raw_npz(path)
    observed_names = set(arrays)
    manifest_names = {item.name for item in manifest.arrays}
    if observed_names != manifest_names:
        raise ValueError(
            f"raw asset array-name mismatch: manifest={sorted(manifest_names)} "
            f"observed={sorted(observed_names)}"
        )
    for item in manifest.arrays:
        array = arrays[item.name]
        if str(array.dtype) != item.dtype or tuple(array.shape) != item.shape:
            raise ValueError(f"raw asset array {item.name!r} manifest/shape-dtype mismatch")
