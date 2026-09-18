"""Root-side profiling orchestration: device discovery, worker execution, Technical Card assembly.

Mirrors the isolation pattern of ``torch_dae.runtime_executor``: the accepted Model Card's
environment is materialized/verified once, then ``torch_dae.profiling_worker`` is executed inside
that environment's own interpreter as a bounded subprocess per (device, thread-regime) session.
CodeCarbon and `psutil` (profiling-tooling identity) run only in *this* root process, wrapped
around the subprocess call; they are never imported inside the model-runtime environment
(Section 25).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from torch_dae.cards.models import ModelCard
from torch_dae.contracts import REPO_RELATIVE_PATTERN
from torch_dae.core.checkpoint import CheckpointManager, checkpoint_cache_path
from torch_dae.environment.fingerprint import (
    LocalPackageProvenance,
    local_package_provenance,
    package_identity_from_content_digest,
)
from torch_dae.environment.manager import EnvironmentManager
from torch_dae.environment.policy import ExecutionPolicy
from torch_dae.profiling import fingerprint as fp
from torch_dae.profiling.contracts import (
    PROFILER_IMPLEMENTATION_VERSION,
    PROFILING_PROTOCOL_ID,
    PROFILING_PROTOCOL_VERSION,
    TECHNICAL_CARD_SCHEMA_VERSION,
    AcceleratorMemoryEvidence,
    ArchitectureEvidence,
    ColdStartEvidence,
    ComparabilityMetadata,
    ConditionStatus,
    ContributorMetadata,
    DeviceBackend,
    DeviceBackendMetadata,
    DeviceSmokeDiagnostic,
    EnergyEvidence,
    EnergyMeasurementKind,
    MeasurementCoverage,
    MinimumInputSearchResult,
    ModelReference,
    PrecisionMetadata,
    ProfilerImplementationProvenance,
    ProfilingCampaignReference,
    ProfilingCampaignResult,
    ProfilingConditions,
    RuntimeClassification,
    SoftwareMetadata,
    SuccessfulDeviceRun,
    SupersessionMetadata,
    TechnicalCard,
    ThreadRegime,
)
from torch_dae.profiling.devices import DeviceSelector, resolve_auto_device_labels
from torch_dae.profiling.eligibility import require_profiling_eligible_model_card
from torch_dae.profiling.energy import EnergyBackend, select_energy_backend
from torch_dae.profiling.identity import (
    TechnicalCardIdentityInputs,
    build_technical_card_identity,
    generate_nonce,
)
from torch_dae.profiling.memory_host import ChildProcessRssSampler, build_host_memory_evidence
from torch_dae.profiling.raw_assets import build_manifest, write_raw_npz
from torch_dae.profiling.synthetic_input import build_provenance
from torch_dae.profiling.timing import summarize_timing

if TYPE_CHECKING:
    import numpy as np

WARMUP_COUNT = 10
MEASURED_COUNT = 50
CANONICAL_BATCH_SIZES = (1, 2, 4, 8)
RESOURCE_PASS_REPEATS = 30
PROTOCOL_DEFAULT_DURATION_SECONDS = 10.0
SYNTHETIC_INPUT_SEED = 1_337
SMOKE_TIMEOUT_SECONDS = 120.0
FULL_RUN_TIMEOUT_SECONDS = 1_800.0


@dataclass(frozen=True)
class _ReadOnlyCheckpoint:
    """Minimal checkpoint handle for an already-valid cached payload (`.path`, `.sha256` only)."""

    path: Path
    sha256: str


def _resolve_checkpoint_read_only(
    root: Path, model_card: ModelCard, environment_id: str
) -> _ReadOnlyCheckpoint:
    """Reuse an already-cached, hash-verified checkpoint payload without any cache mutation.

    `CheckpointManager.ensure_checkpoint` revalidates a strict specification-fingerprint match
    before treating a cache entry as reusable; that fingerprint can legitimately drift from an
    older cached acquisition as the checkpoint contract evolves, forcing a network re-download
    that rewrites `checkpoint-materialization.json` with fresh, non-deterministic provenance
    (a new timestamp and command-log references). That rewrite is harmless to the checkpoint
    *payload* (same bytes, same SHA-256) but invalidates the byte-pinned local-evidence
    cross-check that already-accepted `verification_reports/*.json` perform against it. Profiling
    only needs the verified payload, so it is read read-only here first; network/cache-mutating
    acquisition through `CheckpointManager` remains the fallback for a genuine first-time
    acquisition.
    """

    checkpoint = model_card.checkpoint
    observed = checkpoint.observed_sha256
    if observed and checkpoint.filename:
        candidate = checkpoint_cache_path(root / ".torch-dae", checkpoint.checkpoint_id, observed)
        payload = candidate / checkpoint.filename
        if payload.is_file() and _sha256_file(payload) == observed:
            return _ReadOnlyCheckpoint(path=payload, sha256=observed)

    checkpoints = CheckpointManager(root, policy=ExecutionPolicy(offline=False))
    resolved = checkpoints.ensure_checkpoint(checkpoint, environment_id=environment_id)
    return _ReadOnlyCheckpoint(path=resolved.path, sha256=resolved.sha256)


@dataclass(frozen=True)
class ProfilingRunPlan:
    """One (device, thread-regime) session to execute."""

    device: DeviceSelector
    thread_regime: ThreadRegime | None


@dataclass(frozen=True)
class DeviceRunOutcome:
    """One profiled session's produced Technical Card and raw-asset path (before validation)."""

    plan: ProfilingRunPlan
    technical_card: TechnicalCard
    card_path: Path
    npz_path: Path


def _repository_relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _software_metadata_from_worker(worker_software: dict[str, Any]) -> SoftwareMetadata:
    return SoftwareMetadata(
        os_name=worker_software["os_name"],
        os_version=worker_software["os_version"],
        python_implementation=worker_software["python_implementation"],
        python_version=worker_software["python_version"],
        torch_version=worker_software["torch_version"],
        cuda_runtime_version=worker_software.get("cuda_runtime_version"),
        cuda_driver_version=worker_software.get("cuda_driver_version"),
        cudnn_version=worker_software.get("cudnn_version"),
        mps_backend_info=worker_software.get("mps_backend_info"),
    )


def _run_worker_subprocess(
    *,
    python_executable: Path,
    request: dict[str, Any],
    workspace: Path,
    timeout: float,
) -> tuple[subprocess.Popen[str], int]:
    """Launch `profiling_worker` as a plain subprocess and return (process, pid)."""

    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    request_path = workspace / "request.json"
    result_path = workspace / "result.json"
    request_path.write_text(json.dumps(request, indent=2))
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    process = subprocess.Popen(
        [
            str(python_executable),
            "-I",
            "-m",
            "torch_dae.profiling_worker",
            str(request_path),
            str(result_path),
        ],
        cwd=str(workspace),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return process, process.pid


def discover_devices(
    *,
    python_executable: Path,
    wrapper_entry_point: str,
    checkpoint_path: Path,
    sample_rate: int,
    requested_labels: tuple[str, ...],
    workspace: Path,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[DeviceSmokeDiagnostic, ...]]:
    """Smoke-test each requested device label; return (attempted, successful, failed) tuples."""

    attempted: list[str] = []
    successful: list[str] = []
    failed: list[DeviceSmokeDiagnostic] = []
    for label in requested_labels:
        backend, index = _split_label(label)
        attempted.append(label)
        request = {
            "mode": "smoke",
            "wrapper_entry_point": wrapper_entry_point,
            "checkpoint_path": str(checkpoint_path),
            "device": {"backend": backend, "index": index},
            "sample_rate": sample_rate,
            "seed": SYNTHETIC_INPUT_SEED,
        }
        session_dir = workspace / "smoke" / label.replace(":", "_")
        process, _pid = _run_worker_subprocess(
            python_executable=python_executable,
            request=request,
            workspace=session_dir,
            timeout=SMOKE_TIMEOUT_SECONDS,
        )
        try:
            _stdout, stderr = process.communicate(timeout=SMOKE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            failed.append(DeviceSmokeDiagnostic(device_label=label, error="smoke timeout"))
            continue
        result_path = session_dir / "result.json"
        if not result_path.is_file():
            failed.append(
                DeviceSmokeDiagnostic(
                    device_label=label,
                    error=f"worker produced no result (stderr={stderr[-2000:]})",
                )
            )
            continue
        result = json.loads(result_path.read_text())
        if result.get("smoke_passed"):
            successful.append(label)
        else:
            failed.append(
                DeviceSmokeDiagnostic(
                    device_label=label, error=str(result.get("smoke_error") or result.get("error"))
                )
            )
    return tuple(attempted), tuple(successful), tuple(failed)


def _split_label(label: str) -> tuple[str, int | None]:
    if ":" in label:
        backend, index = label.split(":")
        return backend, int(index)
    return label, None


def query_capabilities(*, python_executable: Path, workspace: Path) -> dict[str, Any]:
    """Query `cuda_available`/`cuda_device_count`/`mps_available` from inside the model env."""

    request: dict[str, Any] = {"mode": "capabilities"}
    session_dir = workspace / "capabilities"
    process, _pid = _run_worker_subprocess(
        python_executable=python_executable,
        request=request,
        workspace=session_dir,
        timeout=SMOKE_TIMEOUT_SECONDS,
    )
    process.communicate(timeout=SMOKE_TIMEOUT_SECONDS)
    result_path = session_dir / "result.json"
    if not result_path.is_file():
        return {"cuda_available": False, "cuda_device_count": 0, "mps_available": False}
    return cast(dict[str, Any], json.loads(result_path.read_text()))


def run_profiling_session(
    *,
    repository_root: Path,
    model_card: ModelCard,
    model_card_path: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    python_executable: Path,
    plan: ProfilingRunPlan,
    energy_mode: str,
    allow_privileged_energy: bool,
    campaign_id: str,
    output_dir: Path,
    content_provenance: LocalPackageProvenance,
) -> DeviceRunOutcome:
    """Execute one full (device, thread-regime) profiling session and assemble a Technical Card.

    ``content_provenance`` is computed exactly once per campaign (by the caller, ``run_campaign``)
    and reused unchanged across every device/thread condition, so a single campaign executed
    without source-code modification carries one stable ``torch_dae_content_identity`` -- it must
    never be recomputed per device run, which is what previously let real mid-campaign source
    edits (or any timing-dependent skew) silently produce mixed-provenance Technical Cards.
    """

    run_id = uuid4().hex
    workspace = repository_root / ".torch-dae/profiling" / campaign_id / run_id
    device = plan.device
    backend_value = device.backend.value if device.backend else "cpu"
    request = {
        "mode": "full",
        "wrapper_entry_point": model_card.identity.wrapper_entry_point,
        "checkpoint_path": str(checkpoint_path),
        "device": {"backend": backend_value, "index": device.index},
        "thread_regime": plan.thread_regime.value if plan.thread_regime else None,
        "sample_rate": model_card.input.sample_rate_hz,
        "canonical_duration_seconds": PROTOCOL_DEFAULT_DURATION_SECONDS,
        "canonical_duration_source": "protocol_default",
        "seed": SYNTHETIC_INPUT_SEED,
        "warmup_count": WARMUP_COUNT,
        "measured_count": MEASURED_COUNT,
        "batch_sizes": list(CANONICAL_BATCH_SIZES),
        "resource_pass_repeats": RESOURCE_PASS_REPEATS,
    }

    energy_backend = cast(
        "EnergyBackend[tuple[str, str, int]]",
        select_energy_backend(
            energy_mode,
            allow_privileged_energy=allow_privileged_energy,
        ),
    )
    process, pid = _run_worker_subprocess(
        python_executable=python_executable,
        request=request,
        workspace=workspace,
        timeout=FULL_RUN_TIMEOUT_SECONDS,
    )

    def wait() -> tuple[str, str, int]:
        stdout, stderr = process.communicate(timeout=FULL_RUN_TIMEOUT_SECONDS)
        return stdout, stderr, process.returncode

    with ChildProcessRssSampler(pid=pid) as sampler:
        (_stdout, stderr, returncode), energy_evidence = energy_backend.measure_block(
            wait, device_backend=device.backend or DeviceBackend.CPU
        )

    result_path = workspace / "result.json"
    if not result_path.is_file():
        raise RuntimeError(
            f"profiling worker produced no result for {plan}: returncode={returncode} "
            f"stderr={stderr[-4000:]}"
        )
    result = json.loads(result_path.read_text())
    if result.get("error"):
        raise RuntimeError(f"profiling worker failed for {plan}: {result['error']}")

    software = _software_metadata_from_worker(result["software"])
    accelerator_hw = result.get("accelerator_hardware") or {}
    hardware = fp.gather_hardware_metadata(
        accelerator_vendor=accelerator_hw.get("accelerator_vendor"),
        accelerator_model=accelerator_hw.get("accelerator_model"),
        accelerator_memory_bytes=accelerator_hw.get("accelerator_memory_bytes"),
    )
    content_identity = package_identity_from_content_digest(
        content_provenance.package_content_sha256
    )
    provenance = content_provenance
    runtime_classification = (
        RuntimeClassification.CANONICAL
        if provenance.repository_dirty is False
        else RuntimeClassification.MODIFIED_RUNTIME
        if provenance.repository_dirty
        else RuntimeClassification.UNKNOWN_RUNTIME
    )

    native_precision = "float32"
    execution_context = fp.build_execution_context_metadata(
        hardware=hardware,
        software=software,
        torch_dae_content_identity=content_identity,
        device_backend=device.backend or DeviceBackend.CPU,
        native_precision=native_precision,
        thread_regime=plan.thread_regime,
        profiling_protocol_id=PROFILING_PROTOCOL_ID,
        profiling_protocol_version=PROFILING_PROTOCOL_VERSION,
        profiler_implementation_version=PROFILER_IMPLEMENTATION_VERSION,
    )

    import numpy as np

    conditions: list[ProfilingConditions] = []
    raw_arrays: dict[str, np.ndarray] = {}
    for raw_condition in result["conditions"]:
        status = ConditionStatus(raw_condition["status"])
        timing = None
        if status == ConditionStatus.SUCCESS:
            raw_ns = raw_condition["raw_ns"]
            array_name = f"raw_ns__{raw_condition['condition_id']}"
            raw_arrays[array_name] = np.asarray(raw_ns, dtype=np.int64)
            timing = summarize_timing(
                raw_ns,
                batch_size=raw_condition["batch_size"],
                input_duration_seconds=raw_condition["duration_seconds"],
            )
        conditions.append(
            ProfilingConditions(
                condition_id=raw_condition["condition_id"],
                duration_seconds=raw_condition["duration_seconds"],
                duration_source=raw_condition["duration_source"],
                batch_size=raw_condition["batch_size"],
                sample_count=raw_condition["sample_count"],
                status=status,
                unsupported_reason=raw_condition.get("unsupported_reason"),
                timing=timing,
                raw_timing_array=(
                    f"raw_ns__{raw_condition['condition_id']}"
                    if status == ConditionStatus.SUCCESS
                    else None
                ),
            )
        )

    minimum_search = MinimumInputSearchResult.model_validate(result["minimum_input_search"])
    architecture = ArchitectureEvidence.model_validate(result["architecture"])
    cold_start = ColdStartEvidence.model_validate(result["cold_start"])

    host_memory = build_host_memory_evidence(
        rss_before_model_load_bytes=result["rss_before_model_load_bytes"],
        rss_after_model_load_bytes=result["rss_after_model_load_bytes"],
        rss_before_resource_pass_bytes=result["rss_before_resource_pass_bytes"],
        rss_sampled_peak_bytes=sampler.peak_bytes,
        rss_after_resource_pass_bytes=result["rss_after_resource_pass_bytes"],
    )
    accelerator_memory = (
        AcceleratorMemoryEvidence.model_validate(result["accelerator_memory"])
        if result.get("accelerator_memory")
        else None
    )

    synthetic_input = build_provenance(
        seed=SYNTHETIC_INPUT_SEED,
        sample_rate=model_card.input.sample_rate_hz,
        batch_size=1,
        channel_count=1,
        sample_count=round(PROTOCOL_DEFAULT_DURATION_SECONDS * model_card.input.sample_rate_hz),
    )

    tested = tuple(c.condition_id for c in conditions if c.status == ConditionStatus.SUCCESS)
    unsupported = tuple(
        c.condition_id for c in conditions if c.status == ConditionStatus.UNSUPPORTED
    )
    coverage = MeasurementCoverage(
        tested_conditions=tested,
        unsupported_conditions=unsupported,
        not_tested_conditions=(),
        architecture_status=architecture.flops_macs_status,
        energy_status=energy_evidence.measurement_kind,
        host_memory_available=host_memory.rss_before_model_load_bytes is not None,
        accelerator_memory_available=accelerator_memory is not None,
    )
    comparability = ComparabilityMetadata(
        protocol_id=PROFILING_PROTOCOL_ID,
        protocol_version=PROFILING_PROTOCOL_VERSION,
        execution_context_fingerprint=execution_context.execution_context_fingerprint,
        runtime_classification=runtime_classification,
    )

    codecarbon_version = energy_evidence.codecarbon_version
    from torch_dae.profiling.memory_host import psutil_available

    _, psutil_version = psutil_available()
    profiler = ProfilerImplementationProvenance(
        profiler_implementation_version=PROFILER_IMPLEMENTATION_VERSION,
        profiling_protocol_id=PROFILING_PROTOCOL_ID,
        profiling_protocol_version=PROFILING_PROTOCOL_VERSION,
        torch_dae_package_version=_installed_package_version(),
        torch_dae_content_identity=content_identity,
        torch_dae_repository_head=provenance.repository_head,
        torch_dae_repository_dirty=provenance.repository_dirty,
        runtime_classification=runtime_classification,
        codecarbon_version=codecarbon_version,
        psutil_version=psutil_version,
    )

    identity_inputs = TechnicalCardIdentityInputs(
        technical_card_schema_version=TECHNICAL_CARD_SCHEMA_VERSION,
        model_card_id=model_card.card_id,
        model_card_sha256=_sha256_file(model_card_path),
        checkpoint_sha256=checkpoint_sha256,
        profiling_protocol_id=PROFILING_PROTOCOL_ID,
        profiling_protocol_version=PROFILING_PROTOCOL_VERSION,
        torch_dae_content_identity=content_identity,
        source_revision=provenance.repository_head,
        repository_dirty=provenance.repository_dirty,
        hardware_fingerprint=execution_context.hardware_fingerprint,
        execution_context_fingerprint=execution_context.execution_context_fingerprint,
        device_backend=backend_value,
        device_index=device.index,
        nonce=generate_nonce(),
    )
    identity = build_technical_card_identity(identity_inputs)

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{identity.technical_card_id}.npz"
    write_raw_npz(npz_path, raw_arrays)
    manifest = build_manifest(
        card_relative_path=npz_path.name,
        path=npz_path,
        arrays=raw_arrays,
    )

    card = TechnicalCard(
        technical_card_schema_version=TECHNICAL_CARD_SCHEMA_VERSION,
        identity=identity,
        model=ModelReference(
            model_card_id=model_card.card_id,
            model_card_path=_repository_relative(repository_root, model_card_path),
            model_card_sha256=identity_inputs.model_card_sha256,
            model_card_status=model_card.card_status.value,
            checkpoint_id=model_card.checkpoint.checkpoint_id,
            checkpoint_sha256=checkpoint_sha256,
            wrapper_entry_point=model_card.identity.wrapper_entry_point,
        ),
        profiler=profiler,
        contributor=ContributorMetadata(),
        hardware=hardware,
        software=software,
        device=DeviceBackendMetadata(
            backend=device.backend or DeviceBackend.CPU,
            device_index=device.index,
            device_label=device.label,
        ),
        execution_context=execution_context,
        precision=PrecisionMetadata(dtype=native_precision),
        synthetic_input=synthetic_input,
        canonical_duration_seconds=PROTOCOL_DEFAULT_DURATION_SECONDS,
        canonical_duration_source="protocol_default",
        architecture=architecture,
        cold_start=cold_start,
        minimum_input_search=minimum_search,
        conditions=tuple(conditions),
        host_memory=host_memory,
        accelerator_memory=accelerator_memory,
        energy=energy_evidence,
        raw_measurements=manifest,
        coverage=coverage,
        comparability=comparability,
        limitations=_derive_limitations(minimum_search, coverage, energy_evidence, architecture),
        campaign=ProfilingCampaignReference(campaign_id=campaign_id, run_id=run_id),
        supersession=SupersessionMetadata(),
        created_at=datetime.now(UTC).isoformat(),
    )
    card_path = output_dir / f"{identity.technical_card_id}.json"
    card_path.write_text(card.model_dump_json(indent=2) + "\n")
    return DeviceRunOutcome(plan=plan, technical_card=card, card_path=card_path, npz_path=npz_path)


def _derive_limitations(
    minimum_search: MinimumInputSearchResult,
    coverage: MeasurementCoverage,
    energy: EnergyEvidence,
    architecture: ArchitectureEvidence,
) -> tuple[str, ...]:
    limitations: list[str] = []
    if minimum_search.status.value != "found":
        limitations.append(f"minimum-input search status: {minimum_search.status.value}")
    if coverage.unsupported_conditions:
        limitations.append(f"unsupported conditions: {', '.join(coverage.unsupported_conditions)}")
    if energy.measurement_kind in (EnergyMeasurementKind.UNAVAILABLE, EnergyMeasurementKind.FAILED):
        limitations.append(f"energy measurement: {energy.measurement_kind.value}")
    if architecture.flops_macs_status.value != "complete":
        limitations.append("FLOPs/MACs coverage: unavailable in Profiling v1")
    return tuple(limitations)


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_campaign(
    *,
    repository_root: Path,
    model_card_id: str,
    requested_devices: tuple[str, ...],
    energy_mode: str,
    output_dir: Path,
    allow_privileged_energy: bool = False,
) -> tuple[ProfilingCampaignResult, list[DeviceRunOutcome]]:
    """Discover devices, profile every successful one, and assemble a campaign result.

    Real profiling artifacts land under ``output_dir`` (a managed candidate workspace, never the
    official ``technical_cards/`` tree). Candidate promotion is a separate, later step.
    """

    from torch_dae.profiling.devices import parse_device_selector
    from torch_dae.profiling.storage import technical_cards_root

    root = repository_root.resolve()
    resolved_output_dir = output_dir.resolve()
    if resolved_output_dir == root:
        raise ValueError(
            "profiling output-dir must be a repository-local candidate/workspace directory, "
            "not the repository root itself"
        )
    try:
        relative_output_dir = resolved_output_dir.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(
            "profiling output-dir must resolve inside the repository root "
            f"{root} (got {resolved_output_dir})"
        ) from exc
    if re.fullmatch(REPO_RELATIVE_PATTERN, relative_output_dir) is None:
        raise ValueError(
            "profiling output-dir must be representable as a repository-relative evidence path "
            f"matching {REPO_RELATIVE_PATTERN!r} (got {relative_output_dir!r})"
        )
    canonical_root = technical_cards_root(root).resolve()
    if resolved_output_dir == canonical_root or canonical_root in resolved_output_dir.parents:
        raise ValueError(
            f"profiling output-dir must not be the canonical technical_cards/ tree or a "
            f"descendant of it (got {resolved_output_dir}); write to a candidate/workspace "
            f"location instead and promote separately"
        )

    campaign_id = f"profiling-{uuid4().hex}"
    family = model_card_id.split("-", 1)[0]
    model_card_path = root / "model_cards" / family / f"{model_card_id}.json"
    if not model_card_path.is_file():
        raise FileNotFoundError(f"model card not found: {model_card_path}")
    model_card = ModelCard.model_validate_json(model_card_path.read_text())
    require_profiling_eligible_model_card(model_card)

    environment_id = model_card.usage.recommended_environment.environment_id
    manager = EnvironmentManager(root, policy=ExecutionPolicy(offline=False))
    materialization = manager.materialize_environment(environment_id)
    verification = manager.verify_environment(
        environment_id, expected_fingerprint=materialization.environment_fingerprint
    )
    if verification.verification_status != "passed":
        raise RuntimeError(f"environment {environment_id} failed verification for profiling")
    resolved = manager.resolved_environment(environment_id)

    checkpoint = _resolve_checkpoint_read_only(root, model_card, environment_id)

    content_provenance = local_package_provenance(root)

    workspace = root / ".torch-dae/profiling" / campaign_id
    workspace.mkdir(parents=True, exist_ok=True)

    parsed = [parse_device_selector(raw) for raw in requested_devices]
    auto_requested = any(item.auto for item in parsed)
    if auto_requested:
        capabilities = query_capabilities(
            python_executable=resolved.python_executable, workspace=workspace
        )
        labels = resolve_auto_device_labels(
            mps_available=bool(capabilities.get("mps_available")),
            cuda_device_count=int(capabilities.get("cuda_device_count") or 0),
        )
    else:
        labels = tuple(item.label for item in parsed)

    attempted, successful_labels, failed_diagnostics = discover_devices(
        python_executable=resolved.python_executable,
        wrapper_entry_point=model_card.identity.wrapper_entry_point,
        checkpoint_path=checkpoint.path,
        sample_rate=model_card.input.sample_rate_hz,
        requested_labels=labels,
        workspace=workspace,
    )

    plans: list[ProfilingRunPlan] = []
    for label in successful_labels:
        backend, index = _split_label(label)
        selector = DeviceSelector(backend=DeviceBackend(backend), index=index, label=label)
        if backend == "cpu":
            plans.append(
                ProfilingRunPlan(device=selector, thread_regime=ThreadRegime.SINGLE_THREAD)
            )
            plans.append(
                ProfilingRunPlan(device=selector, thread_regime=ThreadRegime.NATIVE_DEFAULT)
            )
        else:
            plans.append(ProfilingRunPlan(device=selector, thread_regime=None))

    outcomes: list[DeviceRunOutcome] = []
    successful_runs: list[SuccessfulDeviceRun] = []
    for plan in plans:
        outcome = run_profiling_session(
            repository_root=root,
            model_card=model_card,
            model_card_path=model_card_path,
            checkpoint_path=checkpoint.path,
            checkpoint_sha256=checkpoint.sha256,
            python_executable=resolved.python_executable,
            plan=plan,
            energy_mode=energy_mode,
            allow_privileged_energy=allow_privileged_energy,
            campaign_id=campaign_id,
            output_dir=resolved_output_dir,
            content_provenance=content_provenance,
        )
        outcomes.append(outcome)
        from torch_dae.profiling.validation import validate_technical_card

        validation = validate_technical_card(
            outcome.technical_card, repository_root=root, card_path=outcome.card_path
        )
        successful_runs.append(
            SuccessfulDeviceRun(
                device_label=plan.device.label,
                thread_regime=plan.thread_regime,
                technical_card_id=outcome.technical_card.identity.technical_card_id,
                technical_card_path=_repository_relative(root, outcome.card_path),
                raw_asset_path=_repository_relative(root, outcome.npz_path),
                validation_passed=validation.valid,
                validation_errors=tuple(validation.errors),
            )
        )

    if plans:
        final_provenance = local_package_provenance(root)
        if final_provenance.package_content_sha256 != content_provenance.package_content_sha256:
            raise RuntimeError(
                "torch-dae source content changed during profiling campaign "
                f"{campaign_id} (content digest at start="
                f"{content_provenance.package_content_sha256}, at end="
                f"{final_provenance.package_content_sha256}); refusing to report Technical "
                "Cards with mixed-provenance content identity. Re-run the campaign against an "
                "unmodified working tree."
            )

    campaign = ProfilingCampaignResult(
        campaign_id=campaign_id,
        model_card_id=model_card_id,
        requested_devices=requested_devices,
        detected_devices=labels,
        attempted_devices=attempted,
        successful_device_runs=tuple(successful_runs),
        failed_device_diagnostics=failed_diagnostics,
        profiling_protocol_id=PROFILING_PROTOCOL_ID,
        profiling_protocol_version=PROFILING_PROTOCOL_VERSION,
        profiler_implementation_version=PROFILER_IMPLEMENTATION_VERSION,
        energy_mode=energy_mode,  # type: ignore[arg-type]
        created_at=datetime.now(UTC).isoformat(),
        workspace_directory=_repository_relative(root, workspace),
    )
    (workspace / "campaign-result.json").write_text(campaign.model_dump_json(indent=2) + "\n")
    return campaign, outcomes


def _installed_package_version() -> str:
    try:
        from importlib.metadata import version

        return version("torch-deepaudioembedding")
    except Exception:
        return "0.0.0+unknown"
