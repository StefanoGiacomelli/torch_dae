"""Model runtime-verification and profiling commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from torch_dae.core.errors import FeatureNotAvailableError

app = typer.Typer(
    no_args_is_help=True,
    help="Verify explicit runtime targets and profile accepted runtime-verified models.",
)


@app.command("profile")
def profile(
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help="Accepted runtime_verified Model Card id to profile.",
        ),
    ],
    device: Annotated[
        list[str] | None,
        typer.Option(
            "--device",
            help=(
                "Device selector; repeat for multiple selectors. Supported values are auto, cpu, "
                "mps, cuda, and cuda:<index>. Default: auto."
            ),
        ),
    ] = None,
    protocol: Annotated[
        str,
        typer.Option(
            "--protocol",
            help="Profiling protocol id. Profiling v1 uses audio-inference-v1.",
        ),
    ] = "audio-inference-v1",
    energy: Annotated[
        str,
        typer.Option(
            "--energy",
            help=(
                "Energy policy: auto attempts supported measurement; off disables energy "
                "measurement."
            ),
        ),
    ] = "auto",
    allow_privileged_energy: Annotated[
        bool,
        typer.Option(
            "--allow-privileged-energy",
            help=(
                "Permit a platform energy backend to request local privileged hardware counters "
                "when required (for example Apple PowerMetrics)."
            ),
        ),
    ] = False,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help=(
                "Repository-local candidate directory for campaign results and Technical Card "
                "JSON/NPZ evidence. Canonical technical_cards/ paths are rejected."
            ),
        ),
    ] = Path("./candidate_technical_cards"),
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the ProfilingCampaignResult as JSON."),
    ] = False,
) -> None:
    """Run Profiling v1 for one accepted, runtime-verified Model Card.

    The command creates candidate evidence only. It never mutates the Model Card and never promotes
    output into the canonical ``technical_cards/`` tree. Protocol-controlled warmups, repetitions,
    batch sizes, durations, and CPU thread regimes are intentionally not CLI knobs.
    """

    device = device or ["auto"]
    if protocol != "audio-inference-v1":
        typer.echo(f"unsupported profiling protocol: {protocol!r}", err=True)
        raise typer.Exit(2)
    if energy not in ("auto", "off"):
        typer.echo("--energy must be 'auto' or 'off'", err=True)
        raise typer.Exit(2)

    from torch_dae.onboarding.handoff import discover_repository_root
    from torch_dae.profiling_executor import run_campaign

    try:
        campaign, _outcomes = run_campaign(
            repository_root=discover_repository_root(),
            model_card_id=model,
            requested_devices=tuple(device),
            energy_mode=energy,
            output_dir=output_dir,
            allow_privileged_energy=allow_privileged_energy,
        )
    except Exception as exc:
        payload = {"status": "error", "error": str(exc)}
        typer.echo(json.dumps(payload) if json_output else str(exc), err=True)
        raise typer.Exit(2) from exc

    if json_output:
        typer.echo(campaign.model_dump_json(indent=2))
    else:
        typer.echo(f"campaign {campaign.campaign_id}: {model}")
        typer.echo(f"  requested devices: {', '.join(campaign.requested_devices)}")
        typer.echo(f"  detected devices:  {', '.join(campaign.detected_devices)}")
        for run in campaign.successful_device_runs:
            status = "valid" if run.validation_passed else "INVALID"
            regime = f"/{run.thread_regime.value}" if run.thread_regime else ""
            typer.echo(f"  {run.device_label}{regime}: {run.technical_card_id} [{status}]")
        for diag in campaign.failed_device_diagnostics:
            typer.echo(f"  {diag.device_label}: FAILED ({diag.error})")
    if any(not run.validation_passed for run in campaign.successful_device_runs):
        raise typer.Exit(1)


def _deferred(card_id: str, operation: str) -> None:
    exc = FeatureNotAvailableError(
        f"model {operation} for {card_id!r} is not available in the control-plane CLI"
    )
    typer.echo(str(exc), err=True)
    raise typer.Exit(2) from exc


@app.command("inspect")
def inspect(card_id: str) -> None:
    """Reserved model-inspection command; not available in the current control-plane CLI."""

    _deferred(card_id, "inspect")


@app.command("verify")
def verify(
    target: Annotated[
        Path,
        typer.Option(
            "--target",
            exists=True,
            dir_okay=False,
            help="Path to a schema-2 RuntimeVerificationTarget JSON file.",
        ),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the verification result and evidence paths as JSON."),
    ] = False,
    offline: Annotated[
        bool,
        typer.Option(
            "--offline",
            help="Forbid network access while resolving runtime dependencies and checkpoint data.",
        ),
    ] = False,
) -> None:
    """Execute a schema-2 runtime target in its verified environment; no Model Card is required."""

    from torch_dae.onboarding.handoff import discover_repository_root
    from torch_dae.runtime_executor import execute_runtime_verification
    from torch_dae.runtime_verification import RuntimeVerificationTarget

    try:
        request = RuntimeVerificationTarget.model_validate_json(target.read_text())
        result = execute_runtime_verification(request, discover_repository_root(), offline=offline)
    except Exception as exc:
        typer.echo(json.dumps({"valid": False, "error": str(exc)}) if json_output else str(exc))
        raise typer.Exit(2) from exc
    payload = {
        "report": result.report.model_dump(mode="json"),
        "report_path": str(result.report_path),
        "environment_result": result.environment_result.model_dump(mode="json"),
        "evidence_directory": str(result.evidence_directory),
    }
    typer.echo(
        json.dumps(payload, indent=2)
        if json_output
        else f"{result.report.verification_status}: {result.report_path}"
    )
    if not result.report.successful:
        raise typer.Exit(1)
