from __future__ import annotations

from unittest.mock import patch

from click import unstyle
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
    output = unstyle(result.output)
    assert "--model" in output
    assert "--device" in output
    assert "--protocol" in output
    assert "--energy" in output
    assert "--allow-privileged-energy" in output
    assert "--output-dir" in output
    assert "runtime_verified" in output
    assert "Protocol" in output or "protocol" in output
    assert "candidate" in output


def test_technical_card_help() -> None:
    runner = CliRunner()
    group = runner.invoke(app, ["technical-card", "--help"])
    assert group.exit_code == 0
    assert "raw evidence" in unstyle(group.output)

    validate = runner.invoke(app, ["technical-card", "validate", "--help"])
    assert validate.exit_code == 0
    validate_output = unstyle(validate.output)
    assert "raw NPZ evidence" in validate_output
    assert "--json" in validate_output

    assert runner.invoke(app, ["technical-card", "inspect", "--help"]).exit_code == 0
    assert runner.invoke(app, ["technical-card", "list", "--help"]).exit_code == 0


def test_model_profile_requires_model_option() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["model", "profile"])
    assert result.exit_code != 0
    assert "--model" in unstyle(result.output)


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


def test_model_profile_propagates_privileged_energy_consent() -> None:
    runner = CliRunner()

    with patch(
        "torch_dae.profiling_executor.run_campaign",
        return_value=(_empty_campaign(), []),
    ) as mock_run:
        result = runner.invoke(
            app,
            [
                "model",
                "profile",
                "--model",
                "panns-cnn14-16k-map-0438",
                "--allow-privileged-energy",
                "--json",
            ],
        )

    assert result.exit_code == 0
    assert mock_run.call_args.kwargs["allow_privileged_energy"] is True


def test_technical_card_list_reports_canonical_cards(repo_root: object) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["technical-card", "list"])
    assert result.exit_code == 0
    listed = [line for line in result.output.splitlines() if line]
    assert listed == [
        "panns-cnn14-16k-map-0438/tc-059e5a619219e8dd",
        "panns-cnn14-16k-map-0438/tc-799b336a42338294",
        "panns-cnn14-16k-map-0438/tc-c40157b533b69cb3",
        "panns-resnet38-map-0434/tc-82c3d3220b2ae57b",
        "panns-resnet38-map-0434/tc-84fcb0e59f0cd4eb",
        "panns-resnet38-map-0434/tc-a612f635b7bd6fcc",
        "panns-wavegram-logmel-cnn14-map-0439/tc-3ce3f58a0831eb15",
        "panns-wavegram-logmel-cnn14-map-0439/tc-c8fec918f4cd7a4e",
        "panns-wavegram-logmel-cnn14-map-0439/tc-cc1e330beaab9f78",
    ]


def test_technical_card_validate_reports_missing_file() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["technical-card", "validate", "/no/such/file.json"])
    assert result.exit_code != 0
