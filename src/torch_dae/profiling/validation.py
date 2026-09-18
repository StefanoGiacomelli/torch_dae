"""Strict Technical Card validation (Section 27).

Runs entirely in the root control-plane environment: it never imports PyTorch or executes a
model. Optional/unavailable evidence (energy, FLOPs/MACs) never invalidates an otherwise valid
card; only structural, identity, and cross-reference defects do.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from torch_dae.cards.models import ModelCard
from torch_dae.profiling.contracts import ConditionStatus, TechnicalCard
from torch_dae.profiling.eligibility import require_profiling_eligible_model_card
from torch_dae.profiling.identity import (
    TechnicalCardIdentityInputs,
    validate_identity_recomputation,
)
from torch_dae.profiling.privacy import PrivacyViolation, validate_privacy_safe
from torch_dae.profiling.raw_assets import sha256_file, validate_manifest_matches_file
from torch_dae.profiling.timing import verify_timing_summary


def _resolve_card_relative_asset(*, card_directory: Path, relative_path: str) -> Path:
    """Resolve a manifest ``path`` relative to the Technical Card's own directory.

    Rejects absolute paths, `..` traversal, and any resolved location outside
    ``card_directory`` -- pydantic's ``ensure_repository_relative`` already rejects absolute
    paths and `..` segments at parse time, but this is re-checked here defensively since a
    validator must not trust that a `TechnicalCard` it is handed necessarily went through that
    validator.
    """

    if Path(relative_path).is_absolute():
        raise ValueError(f"raw asset path must not be absolute: {relative_path}")
    candidate = (card_directory / relative_path).resolve()
    base = card_directory.resolve()
    if candidate != base and base not in candidate.parents:
        raise ValueError(f"raw asset path escapes the Technical Card directory: {relative_path}")
    return candidate


@dataclass
class TechnicalCardValidationResult:
    """Structured pass/fail result with every collected error."""

    technical_card_id: str
    valid: bool
    errors: list[str] = field(default_factory=list)


def validate_technical_card(
    card: TechnicalCard,
    *,
    repository_root: Path,
    card_path: Path | None = None,
) -> TechnicalCardValidationResult:
    """Validate one Technical Card against its referenced Model Card and raw assets."""

    errors: list[str] = []
    root = repository_root.resolve()

    try:
        validate_privacy_safe(json.loads(card.model_dump_json()))
    except PrivacyViolation as exc:
        errors.append(f"privacy violation: {exc}")

    model_card_path = root / card.model.model_card_path
    if not model_card_path.is_file():
        errors.append(f"referenced Model Card does not exist: {card.model.model_card_path}")
    else:
        observed_sha256 = sha256_file(model_card_path)
        if observed_sha256 != card.model.model_card_sha256:
            errors.append(
                "referenced Model Card SHA-256 mismatch: "
                f"card={card.model.model_card_sha256} observed={observed_sha256}"
            )
        try:
            model_card = ModelCard.model_validate_json(model_card_path.read_text())
            if model_card.card_id != card.model.model_card_id:
                errors.append("referenced Model Card id mismatch")
            if model_card.card_status.value != card.model.model_card_status:
                errors.append(
                    "referenced Model Card status mismatch: "
                    f"card={card.model.model_card_status!r} "
                    f"observed={model_card.card_status.value!r}"
                )
            try:
                require_profiling_eligible_model_card(model_card)
            except ValueError as exc:
                errors.append(str(exc))
            if model_card.checkpoint.observed_sha256 not in (None, card.model.checkpoint_sha256):
                errors.append("referenced checkpoint SHA-256 disagrees with Model Card")
        except Exception as exc:
            errors.append(f"referenced Model Card failed to parse: {exc}")

    # `comparability.protocol_id` is typed `Literal["audio-inference-v1"]`; pydantic already
    # rejects any other value when the card is parsed, so no runtime check is needed here.

    identity_inputs = TechnicalCardIdentityInputs(
        technical_card_schema_version=card.technical_card_schema_version,
        model_card_id=card.model.model_card_id,
        model_card_sha256=card.model.model_card_sha256,
        checkpoint_sha256=card.model.checkpoint_sha256,
        profiling_protocol_id=card.comparability.protocol_id,
        profiling_protocol_version=card.comparability.protocol_version,
        torch_dae_content_identity=card.profiler.torch_dae_content_identity,
        source_revision=card.profiler.torch_dae_repository_head,
        repository_dirty=card.profiler.torch_dae_repository_dirty,
        hardware_fingerprint=card.execution_context.hardware_fingerprint,
        execution_context_fingerprint=card.execution_context.execution_context_fingerprint,
        device_backend=card.device.backend.value,
        device_index=card.device.device_index,
        nonce=card.identity.nonce,
    )
    try:
        validate_identity_recomputation(card.identity, identity_inputs)
    except ValueError as exc:
        errors.append(str(exc))

    raw_asset_path: Path | None = None
    try:
        raw_asset_path = _resolve_card_relative_asset(
            card_directory=(card_path.parent if card_path is not None else root),
            relative_path=card.raw_measurements.path,
        )
    except ValueError as exc:
        errors.append(str(exc))

    if raw_asset_path is not None:
        try:
            validate_manifest_matches_file(card.raw_measurements, raw_asset_path)
        except ValueError as exc:
            errors.append(str(exc))
        else:
            errors.extend(_validate_raw_vs_summary(card, raw_asset_path))

    for condition in card.conditions:
        if condition.status == ConditionStatus.SUCCESS and condition.raw_timing_array is None:
            errors.append(f"condition {condition.condition_id} missing raw_timing_array reference")
        if condition.status == ConditionStatus.UNSUPPORTED and not condition.unsupported_reason:
            errors.append(f"condition {condition.condition_id} unsupported without a reason")

    if card_path is not None:
        by_id: dict[str, list[Path]] = {}
        for sibling in sorted(card_path.parent.glob("*.json")):
            try:
                other = TechnicalCard.model_validate_json(sibling.read_text())
            except Exception:
                continue
            by_id.setdefault(other.identity.technical_card_id, []).append(sibling)
        for technical_card_id, paths in by_id.items():
            if technical_card_id == card.identity.technical_card_id and len(paths) > 1:
                errors.append(
                    f"duplicate technical_card_id {technical_card_id} in "
                    f"{', '.join(str(p) for p in paths)}"
                )

    if card.supersession.supersedes is not None and card_path is not None:
        target = card_path.parent / f"{card.supersession.supersedes}.json"
        if not target.is_file():
            errors.append(f"supersedes references missing Technical Card: {target}")

    return TechnicalCardValidationResult(
        technical_card_id=card.identity.technical_card_id, valid=not errors, errors=errors
    )


def _validate_raw_vs_summary(card: TechnicalCard, raw_asset_path: Path) -> list[str]:
    from torch_dae.profiling.raw_assets import read_raw_npz

    errors: list[str] = []
    arrays = read_raw_npz(raw_asset_path)
    for condition in card.conditions:
        if condition.status != ConditionStatus.SUCCESS:
            continue
        array_name = condition.raw_timing_array
        if array_name not in arrays:
            errors.append(f"raw array missing for condition {condition.condition_id}")
            continue
        raw_ns = [int(value) for value in arrays[array_name].tolist()]
        assert condition.timing is not None
        if not verify_timing_summary(
            condition.timing,
            raw_ns,
            batch_size=condition.batch_size,
            input_duration_seconds=condition.duration_seconds,
        ):
            errors.append(f"raw-vs-summary timing mismatch for condition {condition.condition_id}")
    return errors


def validate_technical_card_file(
    card_path: Path, *, repository_root: Path
) -> TechnicalCardValidationResult:
    """Load and validate a Technical Card JSON file from disk."""

    try:
        card = TechnicalCard.model_validate_json(card_path.read_text())
    except Exception as exc:
        return TechnicalCardValidationResult(
            technical_card_id=card_path.stem, valid=False, errors=[f"schema validity: {exc}"]
        )
    return validate_technical_card(card, repository_root=repository_root, card_path=card_path)
