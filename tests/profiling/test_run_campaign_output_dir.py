from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from torch_dae.profiling_executor import run_campaign


def _invoke(*, repository_root: Path, output_dir: Path) -> None:
    run_campaign(
        repository_root=repository_root,
        model_card_id="does-not-matter",
        requested_devices=("cpu",),
        energy_mode="off",
        output_dir=output_dir,
    )


def test_run_campaign_accepts_default_candidate_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="model card not found"):
        _invoke(repository_root=tmp_path, output_dir=tmp_path / "candidate_technical_cards")


def test_run_campaign_accepts_another_repository_local_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="model card not found"):
        _invoke(repository_root=tmp_path, output_dir=tmp_path / "profiling_candidates")


def test_run_campaign_rejects_hidden_leading_dot_output_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="repository-relative evidence path"):
        _invoke(repository_root=tmp_path, output_dir=tmp_path / ".torch-dae/profiling-candidates")


def test_run_campaign_rejects_canonical_technical_cards_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="canonical technical_cards/ tree"):
        _invoke(repository_root=tmp_path, output_dir=tmp_path / "technical_cards")


def test_run_campaign_rejects_descendant_of_canonical_technical_cards_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="canonical technical_cards/ tree"):
        _invoke(repository_root=tmp_path, output_dir=tmp_path / "technical_cards" / "some-model")


def test_run_campaign_rejects_repository_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="repository root itself"):
        _invoke(repository_root=tmp_path, output_dir=tmp_path)


def test_run_campaign_rejects_external_absolute_directory(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    with pytest.raises(ValueError, match="must resolve inside the repository root"):
        _invoke(repository_root=repository_root, output_dir=tmp_path / "external")


def test_run_campaign_rejects_symlink_escape(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    escape = repository_root / "escape"
    try:
        escape.symlink_to(external, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    with pytest.raises(ValueError, match="must resolve inside the repository root"):
        _invoke(repository_root=repository_root, output_dir=escape / "candidates")


def test_ineligible_model_card_is_rejected_before_profiling_side_effects(
    tmp_path: Path, real_model_card_path: Path
) -> None:
    import json

    repository_root = tmp_path / "repository"
    model_card_path = repository_root / "model_cards/panns/panns-cnn14-16k-map-0438.json"
    model_card_path.parent.mkdir(parents=True)
    payload = json.loads(real_model_card_path.read_text())
    payload["card_status"] = "checkpoint_verified"
    model_card_path.write_text(json.dumps(payload))

    with patch("torch_dae.profiling_executor.EnvironmentManager") as manager:
        with pytest.raises(ValueError, match="expected card_status='runtime_verified'"):
            run_campaign(
                repository_root=repository_root,
                model_card_id="panns-cnn14-16k-map-0438",
                requested_devices=("cpu",),
                energy_mode="off",
                output_dir=repository_root / "candidate_technical_cards",
            )
    manager.assert_not_called()


def test_runtime_verified_model_card_passes_lifecycle_gate(
    tmp_path: Path, real_model_card_path: Path
) -> None:
    repository_root = tmp_path / "repository"
    model_card_path = repository_root / "model_cards/panns/panns-cnn14-16k-map-0438.json"
    model_card_path.parent.mkdir(parents=True)
    model_card_path.write_bytes(real_model_card_path.read_bytes())

    with patch(
        "torch_dae.profiling_executor.EnvironmentManager",
        side_effect=RuntimeError("lifecycle gate passed"),
    ) as manager:
        with pytest.raises(RuntimeError, match="lifecycle gate passed"):
            run_campaign(
                repository_root=repository_root,
                model_card_id="panns-cnn14-16k-map-0438",
                requested_devices=("cpu",),
                energy_mode="off",
                output_dir=repository_root / "candidate_technical_cards",
            )
    manager.assert_called_once()


def test_legacy_profiled_model_card_is_not_eligible(real_model_card_path: Path) -> None:
    from torch_dae.cards.models import ModelCard, ModelCardLifecycle, ProfilingStatus
    from torch_dae.profiling.eligibility import require_profiling_eligible_model_card

    model_card = ModelCard.model_validate_json(real_model_card_path.read_text())
    profiled = model_card.model_copy(
        update={
            "card_status": ModelCardLifecycle.PROFILED,
            "architectural_profiling": model_card.architectural_profiling.model_copy(
                update={"status": ProfilingStatus.PROFILED}
            ),
            "inference_profiling": model_card.inference_profiling.model_copy(
                update={"status": ProfilingStatus.PROFILED}
            ),
            "energy_profiling": model_card.energy_profiling.model_copy(
                update={"status": ProfilingStatus.PROFILED}
            ),
        }
    )
    with pytest.raises(ValueError, match="got 'profiled'"):
        require_profiling_eligible_model_card(profiled)
