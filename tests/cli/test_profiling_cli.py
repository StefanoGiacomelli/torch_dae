from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from torch_dae.cli.main import app
from torch_dae.profiling.contracts import ProfilingCampaignResult


def _empty_campaign(model_card_id: str = "panns-cnn14-16k-map-0438") -> ProfilingCampaignResult:
    from datetime import UTC, datetime

    return ProfilingCampaignResult(
        campaign_id="campaign-1",
        model_card_id=model_card_id,
        requested_devices=("cpu",),
        detected_devices=("cpu",),
        attempted_devices=("cpu",),
        successful_device_runs=(),
        failed_device_diagnostics=(),
        profiling_protocol_id="audio-inference-v1",
        profiling_protocol_version="1.0.0",
        profiler_implementation_version="1.0.0",
        energy_mode="off",
        created_at=datetime.now(UTC).isoformat(),
    )


def test_model_profile_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["model", "profile", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output
    assert "--device" in result.output
    assert "--protocol" in result.output
    assert "--energy" in result.output
    assert "--output-dir" in result.output


def test_technical_card_help() -> None:
    runner = CliRunner()
    assert runner.invoke(app, ["technical-card", "--help"]).exit_code == 0
    assert runner.invoke(app, ["technical-card", "validate", "--help"]).exit_code == 0
    assert runner.invoke(app, ["technical-card", "inspect", "--help"]).exit_code == 0
    assert runner.invoke(app, ["technical-card", "list", "--help"]).exit_code == 0


def test_model_profile_requires_model_option() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["model", "profile"])
    assert result.exit_code != 0
    assert "--model" in result.output


def test_model_profile_rejects_unknown_protocol() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["model", "profile", "--model", "x", "--protocol", "bogus-v2"])
    assert result.exit_code == 2
    assert "unsupported profiling protocol" in result.output


def test_model_profile_rejects_unknown_energy_mode() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["model", "profile", "--model", "x", "--energy", "bogus"])
    assert result.exit_code == 2
    assert "--energy" in result.output


def test_model_profile_json_output_on_success() -> None:
    runner = CliRunner()
    with patch("torch_dae.profiling_executor.run_campaign", return_value=(_empty_campaign(), [])):
        result = runner.invoke(
            app, ["model", "profile", "--model", "panns-cnn14-16k-map-0438", "--json"]
        )
    assert result.exit_code == 0
    assert '"campaign_id": "campaign-1"' in result.output


def test_model_profile_text_output_reports_failed_devices() -> None:
    from torch_dae.profiling.contracts import DeviceSmokeDiagnostic

    campaign = _empty_campaign().model_copy(
        update={
            "failed_device_diagnostics": (
                DeviceSmokeDiagnostic(device_label="cuda:0", error="not available"),
            )
        }
    )
    runner = CliRunner()
    with patch("torch_dae.profiling_executor.run_campaign", return_value=(campaign, [])):
        result = runner.invoke(app, ["model", "profile", "--model", "panns-cnn14-16k-map-0438"])
    assert result.exit_code == 0
    assert "cuda:0: FAILED (not available)" in result.output


def test_model_profile_reports_error_and_exits_nonzero_on_exception() -> None:
    runner = CliRunner()
    with patch("torch_dae.profiling_executor.run_campaign", side_effect=RuntimeError("boom")):
        result = runner.invoke(app, ["model", "profile", "--model", "does-not-exist"])
    assert result.exit_code == 2
    assert "boom" in result.output


def test_model_profile_default_output_dir_is_not_canonical() -> None:
    from pathlib import Path

    runner = CliRunner()
    with patch(
        "torch_dae.profiling_executor.run_campaign", return_value=(_empty_campaign(), [])
    ) as mock_run:
        result = runner.invoke(
            app, ["model", "profile", "--model", "panns-cnn14-16k-map-0438", "--json"]
        )
    assert result.exit_code == 0
    output_dir = mock_run.call_args.kwargs["output_dir"]
    assert output_dir == Path("./candidate_technical_cards")
    assert output_dir.name != "technical_cards"


def test_technical_card_list_reports_no_official_cards_yet(repo_root: object) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["technical-card", "list"])
    assert result.exit_code == 0
    assert "no official Technical Cards exist yet" in result.output


def test_technical_card_validate_reports_missing_file() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["technical-card", "validate", "/no/such/file.json"])
    assert result.exit_code != 0
