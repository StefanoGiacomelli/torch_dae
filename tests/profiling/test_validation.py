from __future__ import annotations

from pathlib import Path

from tests.profiling.conftest import build_valid_card
from torch_dae.profiling.validation import validate_technical_card, validate_technical_card_file


def test_valid_card_passes(
    repository_root: Path, real_model_card_path: Path, technical_card_workspace: Path
) -> None:
    _card, card_path = build_valid_card(
        repository_root=repository_root,
        model_card_path=real_model_card_path,
        workspace=technical_card_workspace,
    )
    result = validate_technical_card_file(card_path, repository_root=repository_root)
    assert result.valid, result.errors


def test_invalid_model_card_reference(
    repository_root: Path, real_model_card_path: Path, technical_card_workspace: Path
) -> None:
    card, card_path = build_valid_card(
        repository_root=repository_root,
        model_card_path=real_model_card_path,
        workspace=technical_card_workspace,
    )
    bad = card.model_copy(
        update={
            "model": card.model.model_copy(
                update={"model_card_path": "model_cards/panns/does-not-exist.json"}
            )
        }
    )
    result = validate_technical_card(bad, repository_root=repository_root, card_path=card_path)
    assert not result.valid
    assert any("does not exist" in e for e in result.errors)


def test_model_card_hash_mismatch(
    repository_root: Path, real_model_card_path: Path, technical_card_workspace: Path
) -> None:
    card, card_path = build_valid_card(
        repository_root=repository_root,
        model_card_path=real_model_card_path,
        workspace=technical_card_workspace,
    )
    bad = card.model_copy(
        update={"model": card.model.model_copy(update={"model_card_sha256": "f" * 64})}
    )
    result = validate_technical_card(bad, repository_root=repository_root, card_path=card_path)
    assert not result.valid
    assert any("SHA-256 mismatch" in e for e in result.errors)


def test_model_card_status_mismatch(
    repository_root: Path, real_model_card_path: Path, technical_card_workspace: Path
) -> None:
    card, card_path = build_valid_card(
        repository_root=repository_root,
        model_card_path=real_model_card_path,
        workspace=technical_card_workspace,
    )
    bad = card.model_copy(
        update={"model": card.model.model_copy(update={"model_card_status": "checkpoint_verified"})}
    )
    result = validate_technical_card(bad, repository_root=repository_root, card_path=card_path)
    assert not result.valid
    assert any("Model Card status mismatch" in error for error in result.errors)


def test_referenced_model_card_status_must_be_profiling_eligible(
    tmp_path: Path, real_model_card_path: Path
) -> None:
    import json

    repository_root = tmp_path / "repository"
    copied_model_card = repository_root / "model_cards/panns/panns-cnn14-16k-map-0438.json"
    copied_model_card.parent.mkdir(parents=True)
    payload = json.loads(real_model_card_path.read_text())
    payload["card_status"] = "checkpoint_verified"
    copied_model_card.write_text(json.dumps(payload))
    workspace = repository_root / "candidates"
    workspace.mkdir()
    card, card_path = build_valid_card(
        repository_root=repository_root,
        model_card_path=copied_model_card,
        workspace=workspace,
    )
    card = card.model_copy(
        update={"model": card.model.model_copy(update={"model_card_status": "checkpoint_verified"})}
    )

    result = validate_technical_card(card, repository_root=repository_root, card_path=card_path)
    assert not result.valid
    assert any("not eligible for profiling" in error for error in result.errors)


def test_identity_recomputation_mismatch(
    repository_root: Path, real_model_card_path: Path, technical_card_workspace: Path
) -> None:
    card, card_path = build_valid_card(
        repository_root=repository_root,
        model_card_path=real_model_card_path,
        workspace=technical_card_workspace,
    )
    bad = card.model_copy(
        update={"identity": card.identity.model_copy(update={"identity_digest_sha256": "0" * 64})}
    )
    result = validate_technical_card(bad, repository_root=repository_root, card_path=card_path)
    assert not result.valid
    assert any("identity" in e.lower() for e in result.errors)


def test_raw_hash_mismatch(
    repository_root: Path, real_model_card_path: Path, technical_card_workspace: Path
) -> None:
    card, card_path = build_valid_card(
        repository_root=repository_root,
        model_card_path=real_model_card_path,
        workspace=technical_card_workspace,
    )
    bad = card.model_copy(
        update={"raw_measurements": card.raw_measurements.model_copy(update={"sha256": "1" * 64})}
    )
    result = validate_technical_card(bad, repository_root=repository_root, card_path=card_path)
    assert not result.valid
    assert any("SHA-256 mismatch" in e for e in result.errors)


def test_manifest_array_mismatch(
    repository_root: Path, real_model_card_path: Path, technical_card_workspace: Path
) -> None:
    from torch_dae.profiling.contracts import RawArrayDescriptor

    card, card_path = build_valid_card(
        repository_root=repository_root,
        model_card_path=real_model_card_path,
        workspace=technical_card_workspace,
    )
    bad_manifest = card.raw_measurements.model_copy(
        update={"arrays": (RawArrayDescriptor(name="nonexistent", dtype="int64", shape=(50,)),)}
    )
    bad = card.model_copy(update={"raw_measurements": bad_manifest})
    result = validate_technical_card(bad, repository_root=repository_root, card_path=card_path)
    assert not result.valid
    assert any("array-name mismatch" in e for e in result.errors)


def test_raw_vs_summary_mismatch(
    repository_root: Path, real_model_card_path: Path, technical_card_workspace: Path
) -> None:
    card, card_path = build_valid_card(
        repository_root=repository_root,
        model_card_path=real_model_card_path,
        workspace=technical_card_workspace,
    )
    tampered_condition = card.conditions[0].model_copy(
        update={"timing": card.conditions[0].timing.model_copy(update={"mean_ns": 999.0})}
    )
    bad = card.model_copy(update={"conditions": (tampered_condition,)})
    result = validate_technical_card(bad, repository_root=repository_root, card_path=card_path)
    assert not result.valid
    assert any("raw-vs-summary" in e for e in result.errors)


def test_duplicate_card_id_detected(
    repository_root: Path, real_model_card_path: Path, technical_card_workspace: Path
) -> None:
    card_a, path_a = build_valid_card(
        repository_root=repository_root,
        model_card_path=real_model_card_path,
        workspace=technical_card_workspace,
        technical_card_id_seed="dup-a",
    )
    # Force a duplicate id by writing card_a's own JSON again under a different filename.
    duplicate_path = technical_card_workspace / "duplicate-copy.json"
    duplicate_path.write_text(card_a.model_dump_json(indent=2))

    result = validate_technical_card(card_a, repository_root=repository_root, card_path=path_a)
    assert any("duplicate technical_card_id" in e for e in result.errors)


def test_supersession_reference_must_exist(
    repository_root: Path, real_model_card_path: Path, technical_card_workspace: Path
) -> None:
    card, card_path = build_valid_card(
        repository_root=repository_root,
        model_card_path=real_model_card_path,
        workspace=technical_card_workspace,
    )
    bad = card.model_copy(
        update={
            "supersession": card.supersession.model_copy(
                update={"supersedes": "tc-doesnotexist", "supersede_reason": "correction"}
            )
        }
    )
    result = validate_technical_card(bad, repository_root=repository_root, card_path=card_path)
    assert not result.valid
    assert any("supersedes references missing" in e for e in result.errors)


def test_privacy_leak_detected(
    repository_root: Path, real_model_card_path: Path, technical_card_workspace: Path
) -> None:
    card, card_path = build_valid_card(
        repository_root=repository_root,
        model_card_path=real_model_card_path,
        workspace=technical_card_workspace,
    )
    bad = card.model_copy(
        update={
            "contributor": card.contributor.model_copy(
                update={"display_name": "reachable at 192.168.1.55"}
            )
        }
    )
    result = validate_technical_card(bad, repository_root=repository_root, card_path=card_path)
    assert not result.valid
    assert any("privacy violation" in e for e in result.errors)


def test_optional_unavailable_evidence_does_not_invalidate(
    repository_root: Path, real_model_card_path: Path, technical_card_workspace: Path
) -> None:
    # The fixture already uses EnergyMeasurementKind.UNAVAILABLE and an "unavailable"
    # architecture status; the card must still validate.
    card, card_path = build_valid_card(
        repository_root=repository_root,
        model_card_path=real_model_card_path,
        workspace=technical_card_workspace,
    )
    assert card.energy.measurement_kind.value == "unavailable"
    assert card.architecture.flops_macs_status.value == "unavailable"
    result = validate_technical_card_file(card_path, repository_root=repository_root)
    assert result.valid, result.errors


def test_malformed_json_is_a_validation_failure_not_a_crash(
    tmp_path: Path, repository_root: Path
) -> None:
    bad_path = tmp_path / "not-a-card.json"
    bad_path.write_text("{not valid json")
    result = validate_technical_card_file(bad_path, repository_root=repository_root)
    assert not result.valid


def test_raw_asset_reference_is_relocation_safe(
    repository_root: Path, real_model_card_path: Path, tmp_path: Path
) -> None:
    """A reviewed JSON+`.npz` pair must validate identically before and after being relocated,
    without rewriting the JSON bytes (Section 3 of the corrective task: candidate/promotion
    byte-identical relocation)."""

    candidate_dir = tmp_path / "candidate_technical_cards"
    candidate_dir.mkdir()
    card, card_path = build_valid_card(
        repository_root=repository_root,
        model_card_path=real_model_card_path,
        workspace=candidate_dir,
    )
    assert "/" not in card.raw_measurements.path
    npz_path = candidate_dir / card.raw_measurements.path
    assert npz_path.is_file()

    result_before = validate_technical_card_file(card_path, repository_root=repository_root)
    assert result_before.valid, result_before.errors

    promoted_dir = tmp_path / "technical_cards" / card.model.model_card_id
    promoted_dir.mkdir(parents=True)
    promoted_card_path = promoted_dir / card_path.name
    promoted_npz_path = promoted_dir / npz_path.name
    original_json_bytes = card_path.read_bytes()
    original_npz_bytes = npz_path.read_bytes()
    promoted_card_path.write_bytes(original_json_bytes)
    promoted_npz_path.write_bytes(original_npz_bytes)

    assert promoted_card_path.read_bytes() == original_json_bytes  # byte-identical, no rewrite

    result_after = validate_technical_card_file(promoted_card_path, repository_root=repository_root)
    assert result_after.valid, result_after.errors


def test_raw_asset_path_rejects_absolute_path() -> None:
    import pytest

    from torch_dae.profiling.contracts import RawMeasurementManifest

    with pytest.raises(ValueError, match="string_pattern_mismatch"):
        RawMeasurementManifest(path="/etc/passwd", sha256="a" * 64, arrays=())


def test_raw_asset_path_rejects_traversal_outside_card_directory() -> None:
    import pytest

    from torch_dae.profiling.contracts import RawMeasurementManifest

    with pytest.raises(ValueError, match="string_pattern_mismatch"):
        RawMeasurementManifest(path="../escaped.npz", sha256="a" * 64, arrays=())
