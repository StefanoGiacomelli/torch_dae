"""Card-independent runtime execution through verified managed environments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from torch_dae.core.checkpoint import (
    CheckpointManager,
    ResolvedCheckpoint,
    checkpoint_specification_fingerprint,
)
from torch_dae.environment.manager import EnvironmentManager
from torch_dae.environment.policy import ExecutionPolicy
from torch_dae.environment.results import ArtifactEvidence, EnvironmentVerificationResult
from torch_dae.environment.runtime import RuntimeReportSink
from torch_dae.environment.sources import sha256_file
from torch_dae.environment.verification import VerificationReport
from torch_dae.runtime_verification import (
    RuntimeVerificationTarget,
    validate_runtime_verification_target,
)


@dataclass(frozen=True)
class RuntimeVerificationExecution:
    """Validated report and retained infrastructure evidence for one execution."""

    report: VerificationReport
    report_path: Path
    environment_result: EnvironmentVerificationResult
    checkpoint: ResolvedCheckpoint
    evidence_directory: Path


def validate_runtime_report(
    report: VerificationReport,
    target: RuntimeVerificationTarget,
    *,
    environment_fingerprint: str,
    checkpoint_sha256: str,
) -> VerificationReport:
    """Reject stale identities, altered check contracts, and incomplete output evidence.

    Parameters
    ----------
    report
        Candidate runtime report.
    target
        Validated schema-2 request.
    environment_fingerprint
        Verified managed environment identity.
    checkpoint_sha256
        Content identity returned by the checkpoint manager.

    Returns
    -------
    VerificationReport
        Report with matching identities, check coverage and output evidence.
    """

    report = VerificationReport.model_validate_json(report.model_dump_json())
    expected = {
        "schema_version": "2.0.0",
        "runtime_target_id": target.target_id,
        "workflow_id": target.workflow_id,
        "integrated_variant_id": target.integrated_variant_id,
        "checkpoint_id": target.checkpoint.checkpoint_id,
        "public_model_entry_point": target.public_model_entry_point,
        "integration_handoff_sha256": target.accepted_integration_handoff.sha256,
        "environment_spec_sha256": target.environment_spec_sha256,
        "source_manifest_sha256": target.source_manifest.sha256,
        "model_card_id": target.future_card_id or target.registry_identity,
        "environment_id": target.environment_id,
        "environment_fingerprint": environment_fingerprint,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_specification_fingerprint": checkpoint_specification_fingerprint(
            target.checkpoint
        ),
        "required_check_ids": target.required_check_ids,
        "optional_check_ids": target.optional_check_ids,
    }
    for name, value in expected.items():
        if getattr(report, name) != value:
            raise ValueError(f"runtime report {name} mismatch")
    if report.device not in target.permitted_devices:
        raise ValueError("runtime report device is outside target scope")
    if report.successful:
        observations = {item.name: item for item in report.tensor_observations}
        for output in target.expected_outputs:
            observed = observations.get(output.name)
            if observed is None or observed.rank != output.rank:
                raise ValueError(f"missing or invalid output observation: {output.name}")
            for expected_dimension, actual in zip(output.dimensions, observed.shape, strict=True):
                if expected_dimension.isdecimal() and actual.size != int(expected_dimension):
                    raise ValueError(f"output dimension mismatch: {output.name}")
        if target.default_embedding.embedding_id not in report.embedding_results:
            raise ValueError("missing default embedding observation")
    return report


def execute_runtime_verification(
    target: RuntimeVerificationTarget,
    repository_root: Path,
    *,
    offline: bool = False,
) -> RuntimeVerificationExecution:
    """Execute a schema-2 target without a card, returning passed or failed evidence.

    Parameters
    ----------
    target
        Validated schema-2 request; no model card is required.
    repository_root
        Repository containing accepted integration and environment artifacts.
    offline
        Require retained dependency, metadata and checkpoint caches.

    Returns
    -------
    RuntimeVerificationExecution
        Passed or failed report with retained infrastructure evidence.

    Notes
    -----
    The entry-point package supplies ``verification.Provider``; it is imported only by
    the managed child interpreter. A failed report is returned for inspection and must
    not be promoted. Infrastructure or malformed-provider evidence raises an exception.
    """

    root = Path(repository_root).resolve()
    target = RuntimeVerificationTarget.model_validate_json(target.model_dump_json())
    if target.schema_version != "2.0.0":
        raise ValueError("execution requires a schema-2 runtime target")
    validate_runtime_verification_target(target, root)
    policy = ExecutionPolicy(offline=offline)
    manager = EnvironmentManager(root, policy=policy)
    materialization = manager.materialize_environment(
        target.environment_id, expected_spec_sha256=target.environment_spec_sha256
    )
    environment = manager.verify_environment(
        target.environment_id, expected_fingerprint=materialization.environment_fingerprint
    )
    resolved = manager.resolved_environment(target.environment_id)
    if (
        environment.verification_status != "passed"
        or environment.environment_id != target.environment_id
        or environment.environment_spec_sha256 != target.environment_spec_sha256
        or environment.environment_fingerprint != resolved.fingerprint
    ):
        raise ValueError("verified environment identity mismatch")
    checkpoints = CheckpointManager(root, policy=policy)
    checkpoint = checkpoints.ensure_checkpoint(
        target.checkpoint,
        environment_id=target.environment_id,
        acquisition_policy=target.checkpoint_acquisition_policy,
    )
    if checkpoint.checkpoint_id != target.checkpoint.checkpoint_id:
        raise ValueError("checkpoint identity mismatch")
    for expected in (target.checkpoint.expected_sha256, target.checkpoint.observed_sha256):
        if expected is not None and expected != checkpoint.sha256:
            raise ValueError("checkpoint SHA-256 mismatch")
    controls = {
        "environment_evidence": "Verified managed environment and matching fingerprint.",
        "checkpoint_acquisition_integrity": "Managed cache validated size, checksums and SHA-256.",
    }
    controls["checkpoint-authority-metadata"] = (
        "Authority metadata validated by checkpoint manager."
    )
    controls["checkpoint-acquisition-integrity"] = controls["checkpoint_acquisition_integrity"]
    if {"offline_cache_reuse", "checkpoint-offline-cache-reuse"} & set(
        (*target.required_check_ids, *target.optional_check_ids)
    ):
        reused = CheckpointManager(root, policy=ExecutionPolicy(offline=True)).ensure_checkpoint(
            target.checkpoint,
            environment_id=target.environment_id,
            acquisition_policy=target.checkpoint_acquisition_policy.model_copy(
                update={"allow_network": False}
            ),
        )
        if reused != checkpoint:
            raise ValueError("offline checkpoint identity mismatch")
        controls["offline_cache_reuse"] = (
            "Offline manager revalidated retained bytes and provenance."
        )
    if "offline_cache_reuse" in controls:
        controls["checkpoint-offline-cache-reuse"] = controls["offline_cache_reuse"]
    directory = root / ".torch-dae/reports/runtime" / target.target_id / uuid4().hex
    directory.mkdir(parents=True)
    record_path = checkpoint.path.parent / "checkpoint-materialization.json"
    evidence = ArtifactEvidence(
        path=record_path.relative_to(root).as_posix(), sha256=sha256_file(record_path)
    )
    request = {
        "target": target.model_dump(mode="json"),
        "environment": environment.model_dump(mode="json"),
        "checkpoint": checkpoint.model_dump(mode="json"),
        "checkpoint_materialization": evidence.model_dump(mode="json"),
        "controls": controls,
    }
    request_path = directory / "request.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    (directory / "materialization.json").write_text(
        materialization.model_dump_json(indent=2) + "\n"
    )
    (directory / "environment.json").write_text(environment.model_dump_json(indent=2) + "\n")
    report_path = directory / "report.json"
    result = manager.executor.with_report_sink(RuntimeReportSink(directory, "commands")).run(
        [
            str(resolved.python_executable),
            "-I",
            "-m",
            "torch_dae.runtime_worker",
            str(request_path),
            str(report_path),
        ],
        cwd=directory,
        env={"PYTORCH_ENABLE_MPS_FALLBACK": "0"},
        env_remove=("PYTHONPATH", "PYTHONHOME"),
        timeout=target.verification_limits.timeout_seconds,
    )
    if not report_path.is_file():
        raise ValueError(
            f"runtime worker produced no report: {result.stderr}; evidence={directory}"
        )
    report = validate_runtime_report(
        VerificationReport.model_validate_json(report_path.read_text()),
        target,
        environment_fingerprint=resolved.fingerprint,
        checkpoint_sha256=checkpoint.sha256,
    )
    if result.returncode != (0 if report.successful else 1):
        raise ValueError(f"runtime worker exit status disagrees with report: {directory}")
    if report.checkpoint_materialization != evidence:
        raise ValueError("runtime report checkpoint materialization mismatch")
    return RuntimeVerificationExecution(report, report_path, environment, checkpoint, directory)
