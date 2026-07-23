"""Future runtime materialization path helpers."""

from pathlib import Path

from torch_dae.contracts import contained_path, ensure_canonical_id
from torch_dae.core.checkpoint import validate_sha256


def materialization_path(runtime_root: Path, card_id: str, fingerprint: str) -> Path:
    """Return `.torch-dae/environments/<card-id>/<fingerprint>/`."""

    ensure_canonical_id(card_id)
    validate_sha256(fingerprint)
    return contained_path(runtime_root / "environments", card_id, fingerprint)
