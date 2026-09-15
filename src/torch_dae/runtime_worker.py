"""Managed-interpreter verification worker; never import model code in the control plane."""

from __future__ import annotations

import importlib
import json
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from torch_dae.core.checkpoint import checkpoint_specification_fingerprint
from torch_dae.core.errors import UnsupportedCapabilityError
from torch_dae.environment.verification import (
    TensorObservation,
    VerificationCheck,
    VerificationReport,
)
from torch_dae.runtime_verification import RuntimeVerificationTarget


class VerificationProvider(Protocol):
    """Model package adapter; each check is dispatched once in target order.

    Providers raise for failed assertions and use UnsupportedCapabilityError only for
    capabilities the wrapper cannot supply. They expose observed tensors and embeddings.
    """

    tensor_observations: list[TensorObservation]
    embedding_results: list[str]

    def check(self, name: str) -> str:
        """Observe a declared check and return details; raise on failure or unsupported behavior."""


def run_checks(
    target: RuntimeVerificationTarget,
    provider: VerificationProvider,
    controls: dict[str, str],
) -> tuple[VerificationCheck, ...]:
    """Dispatch each declared check exactly once and preserve failures without promotion."""

    checks = []
    for name in (*target.required_check_ids, *target.optional_check_ids):
        status = "passed"
        try:
            details = controls[name] if name in controls else provider.check(name)
        except UnsupportedCapabilityError as exc:
            status = "unsupported" if name in target.optional_check_ids else "failed"
            details = str(exc) or "Provider does not support this capability."
        except Exception as exc:
            status = "failed"
            details = f"{type(exc).__name__}: {exc}\n" + traceback.format_exc(limit=5)
        checks.append(VerificationCheck(name=name, status=status, details=details))  # type: ignore[arg-type]
    return tuple(checks)


def execute_request(request: dict[str, Any]) -> VerificationReport:
    """Import the public entry point and model verification adapter inside this interpreter."""

    target = RuntimeVerificationTarget.model_validate(request["target"])
    if target.schema_version != "2.0.0":
        raise ValueError("worker requires schema 2")
    module_name, symbol = target.public_model_entry_point.split(":")
    model_class = getattr(importlib.import_module(module_name), symbol)
    provider_type = importlib.import_module(module_name + ".verification").Provider
    provider: VerificationProvider = provider_type(
        model_class, target, Path(request["checkpoint"]["path"])
    )
    checks = run_checks(target, provider, request["controls"])
    unsupported = tuple(item.name for item in checks if item.status == "unsupported")
    return VerificationReport(
        schema_version="2.0.0",
        report_id=target.target_id + "-runtime",
        runtime_target_id=target.target_id,
        workflow_id=target.workflow_id,
        integrated_variant_id=target.integrated_variant_id,
        checkpoint_id=target.checkpoint.checkpoint_id,
        public_model_entry_point=target.public_model_entry_point,
        integration_handoff_sha256=target.accepted_integration_handoff.sha256,
        environment_spec_sha256=target.environment_spec_sha256,
        source_manifest_sha256=target.source_manifest.sha256,
        model_card_id=target.future_card_id or target.registry_identity,
        environment_id=target.environment_id,
        environment_fingerprint=request["environment"]["environment_fingerprint"],
        created_at=datetime.now(UTC),
        platform=request["environment"]["platform"],
        device="cpu",
        checkpoint_sha256=request["checkpoint"]["sha256"],
        checkpoint_specification_fingerprint=checkpoint_specification_fingerprint(
            target.checkpoint
        ),
        checkpoint_materialization=request["checkpoint_materialization"],
        input_contracts=("waveform[B,C,T]", "sample_rate", "valid_lengths[B]"),
        tensor_observations=tuple(provider.tensor_observations),
        embedding_results=tuple(provider.embedding_results),
        passed_capabilities=tuple(item.name for item in checks if item.status == "passed"),
        unsupported_capabilities=unsupported,
        known_limitations=(
            *target.known_unresolved_items,
            *(item.details for item in checks if item.status == "unsupported" and item.details),
        ),
        required_check_ids=target.required_check_ids,
        optional_check_ids=target.optional_check_ids,
        checks=checks,
        verification_status="failed" if any(c.status == "failed" for c in checks) else "passed",
    )


def main() -> int:
    """Run an executor-created request and persist the strict report, including failures."""

    request_path, report_path = map(Path, sys.argv[1:])
    report = execute_request(json.loads(request_path.read_text()))
    report_path.write_text(report.model_dump_json(indent=2) + "\n")
    return 0 if report.successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
