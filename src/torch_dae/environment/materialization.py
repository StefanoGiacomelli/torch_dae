"""Future runtime materialization path helpers."""

from pathlib import Path

from torch_dae.contracts import contained_path, ensure_canonical_id
from torch_dae.core.checkpoint import validate_sha256


def materialization_path(runtime_root: Path, environment_id: str, fingerprint: str) -> Path:
    """Return `.torch-dae/environments/<environment-id>/<fingerprint>/`."""

    ensure_canonical_id(environment_id)
    validate_sha256(fingerprint)
    return contained_path(runtime_root / "environments", environment_id, fingerprint)
