"""Model inspection and card-independent runtime verification commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from torch_dae.core.errors import FeatureNotAvailableError

app = typer.Typer(no_args_is_help=True, help="Model commands.")


def _deferred(card_id: str, operation: str) -> None:
    exc = FeatureNotAvailableError(
        f"model {operation} for {card_id!r} is not available in the control-plane CLI"
    )
    typer.echo(str(exc), err=True)
    raise typer.Exit(2) from exc


@app.command("inspect")
def inspect(card_id: str) -> None:
    _deferred(card_id, "inspect")


@app.command("verify")
def verify(
    target: Annotated[Path, typer.Option("--target", exists=True, dir_okay=False)],
    json_output: Annotated[bool, typer.Option("--json")] = False,
    offline: Annotated[bool, typer.Option("--offline")] = False,
) -> None:
    """Execute a schema-2 runtime target in its verified environment; no model card required."""

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
