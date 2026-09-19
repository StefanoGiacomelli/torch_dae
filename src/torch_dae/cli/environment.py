"""Environment CLI commands."""

from __future__ import annotations

import json

import typer

from torch_dae.core.errors import (
    EnvironmentAlreadyExistsError,
    EnvironmentDependencyClosureError,
    EnvironmentMaterializationError,
    EnvironmentNotFoundError,
    EnvironmentVerificationError,
    GitUnavailableError,
    OfflineResourceUnavailableError,
    PythonInterpreterUnavailableError,
    TorchDaeError,
    UvUnavailableError,
)
from torch_dae.environment.manager import EnvironmentManager, ResolvedEnvironment
from torch_dae.environment.policy import ExecutionPolicy
from torch_dae.environment.results import EnvironmentFailureClassification

app = typer.Typer(no_args_is_help=True, help="Environment commands.")


def _policy(offline: bool, no_python_downloads: bool) -> ExecutionPolicy:
    return ExecutionPolicy(offline=offline, allow_python_downloads=not no_python_downloads)


def _manager(offline: bool = False, no_python_downloads: bool = False) -> EnvironmentManager:
    return EnvironmentManager.from_repository_root(policy=_policy(offline, no_python_downloads))


def _exit_for_error(
    exc: TorchDaeError,
    *,
    json_output: bool = False,
    environment_id: str | None = None,
) -> None:
    classification = (
        EnvironmentFailureClassification.INTERPRETER_UNAVAILABLE
        if isinstance(exc, PythonInterpreterUnavailableError)
        else EnvironmentFailureClassification.DEPENDENCY_CLOSURE
        if isinstance(exc, EnvironmentDependencyClosureError)
        else EnvironmentFailureClassification.VERIFICATION_SCRIPT
        if isinstance(exc, EnvironmentVerificationError)
        else EnvironmentFailureClassification.INVALID_SPECIFICATION
        if isinstance(exc, (EnvironmentMaterializationError, EnvironmentNotFoundError))
        else EnvironmentFailureClassification.EXTERNAL_COMMAND
    )
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "environment_id": environment_id,
                    "result_status": "failed",
                    "environment_fingerprint": None,
                    "evidence_path": None,
                    "failure_classification": classification.value,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            ),
            err=True,
        )
    else:
        typer.echo(str(exc), err=True)
    if isinstance(exc, (OfflineResourceUnavailableError, EnvironmentNotFoundError)):
        raise typer.Exit(3) from exc
    if isinstance(
        exc,
        (
            EnvironmentAlreadyExistsError,
            EnvironmentMaterializationError,
            EnvironmentVerificationError,
            GitUnavailableError,
            PythonInterpreterUnavailableError,
            UvUnavailableError,
        ),
    ):
        raise typer.Exit(4) from exc
    raise typer.Exit(2) from exc


@app.command("resolve")
def resolve(
    environment_id: str,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Resolve canonical environment artifacts without a model card."""

    try:
        result = _manager().resolve_environment(environment_id)
        if json_output:
            payload = result.model_dump(mode="json")
            payload.update(
                {
                    "result_status": "resolved",
                    "evidence_path": result.specification_path,
                    "failure_classification": None,
                }
            )
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            typer.echo(f"resolved: {result.environment_id} {result.environment_fingerprint}")
    except TorchDaeError as exc:
        _exit_for_error(exc, json_output=json_output, environment_id=environment_id)


@app.command("materialize")
def materialize(
    environment_id: str,
    expected_spec_sha256: str | None = typer.Option(None, "--expected-spec-sha256"),
    offline: bool = typer.Option(False, "--offline"),
    no_python_downloads: bool = typer.Option(False, "--no-python-downloads"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Materialize canonical environment artifacts without a model card."""

    try:
        result = _manager(offline, no_python_downloads).materialize_environment(
            environment_id,
            expected_spec_sha256=expected_spec_sha256,
        )
        if json_output:
            definition = _manager().resolve_environment(environment_id)
            payload = result.model_dump(mode="json")
            payload.update(
                {
                    "result_status": result.status,
                    "environment_fingerprint": definition.environment_fingerprint,
                    "evidence_path": result.materialization_record.path,
                }
            )
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            typer.echo(f"materialized: {result.environment_id} {result.status}")
    except TorchDaeError as exc:
        _exit_for_error(exc, json_output=json_output, environment_id=environment_id)


@app.command("preflight")
def preflight(
    environment_id: str,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Check local-wheel runtime requirements against the accepted lock without materializing."""

    try:
        result = _manager(offline=True, no_python_downloads=True).preflight_environment(
            environment_id
        )
        if json_output:
            typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
        else:
            typer.echo(f"preflight: {result.environment_id} {result.status}")
        if result.status != "passed":
            raise typer.Exit(4)
    except TorchDaeError as exc:
        _exit_for_error(exc, json_output=json_output, environment_id=environment_id)


@app.command("create")
def create(
    card_id: str,
    offline: bool = typer.Option(False, "--offline"),
    no_python_downloads: bool = typer.Option(False, "--no-python-downloads"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    try:
        resolved = _manager(offline, no_python_downloads).create(card_id)
        _print_resolved(resolved, json_output)
    except TorchDaeError as exc:
        _exit_for_error(exc)


@app.command("ensure")
def ensure(
    card_id: str,
    offline: bool = typer.Option(False, "--offline"),
    no_python_downloads: bool = typer.Option(False, "--no-python-downloads"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    try:
        resolved = _manager(offline, no_python_downloads).ensure(card_id)
        _print_resolved(resolved, json_output)
    except TorchDaeError as exc:
        _exit_for_error(exc)


@app.command("verify")
def verify(
    environment_id: str,
    offline: bool = typer.Option(False, "--offline"),
    no_python_downloads: bool = typer.Option(False, "--no-python-downloads"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    try:
        result = _manager(offline, no_python_downloads).verify_environment(environment_id)
        if json_output:
            payload = result.model_dump(mode="json")
            payload.update(
                {
                    "result_status": result.verification_status,
                    "evidence_path": result.result_path,
                }
            )
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            typer.echo(f"verified: {result.environment_id} {result.environment_fingerprint}")
    except TorchDaeError as exc:
        _exit_for_error(exc, json_output=json_output, environment_id=environment_id)


@app.command("remove")
def remove(card_id: str, json_output: bool = typer.Option(False, "--json")) -> None:
    try:
        _manager().remove(card_id)
        if json_output:
            import json

            typer.echo(json.dumps({"card_id": card_id, "removed": True}, indent=2, sort_keys=True))
        else:
            typer.echo(f"removed: {card_id}")
    except TorchDaeError as exc:
        _exit_for_error(exc)


@app.command("info")
def info(card_id: str, json_output: bool = typer.Option(False, "--json")) -> None:
    try:
        if json_output:
            typer.echo(_manager().info_json(card_id))
        else:
            observed = _manager().info(card_id)
            typer.echo(f"{observed.model_card_id}: {observed.status}")
    except TorchDaeError as exc:
        _exit_for_error(exc)


@app.command("run", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(
    ctx: typer.Context,
    card_id: str,
    offline: bool = typer.Option(False, "--offline"),
    no_python_downloads: bool = typer.Option(False, "--no-python-downloads"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    command = list(ctx.args)
    try:
        result = _manager(offline, no_python_downloads).run(card_id, command)
        if json_output:
            import json

            typer.echo(json.dumps(result.__dict__, indent=2, sort_keys=True))
        else:
            typer.echo(result.stdout, nl=False)
            if result.stderr:
                typer.echo(result.stderr, err=True, nl=False)
        raise typer.Exit(result.returncode)
    except TorchDaeError as exc:
        _exit_for_error(exc)


def _print_resolved(resolved: ResolvedEnvironment, json_output: bool) -> None:
    if json_output:
        import json

        typer.echo(
            json.dumps(
                {
                    "environment_id": resolved.environment_id,
                    "card_id": resolved.model_card_id,
                    "root": str(resolved.root),
                    "python_executable": str(resolved.python_executable),
                    "fingerprint": resolved.fingerprint,
                    "valid": resolved.valid,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        typer.echo(f"environment ready: {resolved.model_card_id} {resolved.fingerprint}")
