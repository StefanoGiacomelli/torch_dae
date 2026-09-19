from __future__ import annotations

import hashlib
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from torch_dae.profiling.contracts import (
    PROFILER_IMPLEMENTATION_VERSION,
    PROFILING_PROTOCOL_ID,
    PROFILING_PROTOCOL_VERSION,
    TECHNICAL_CARD_SCHEMA_VERSION,
    ArchitectureEvidence,
    ColdStartEvidence,
    ComparabilityMetadata,
    ConditionStatus,
    ContributorMetadata,
    DeviceBackend,
    DeviceBackendMetadata,
    EnergyEvidence,
    EnergyMeasurementKind,
    ExecutionContextMetadata,
    HardwareMetadata,
    HostMemoryEvidence,
    MeasurementCoverage,
    MinimumInputSearchResult,
    MinimumInputStatus,
    ModelReference,
    PrecisionMetadata,
    ProfilerImplementationProvenance,
    ProfilingCampaignReference,
    ProfilingConditions,
    RuntimeClassification,
    SoftwareMetadata,
    SupersessionMetadata,
    SyntheticInputProvenance,
    TechnicalCard,
    ThreadRegime,
)
from torch_dae.profiling.identity import TechnicalCardIdentityInputs, build_technical_card_identity
from torch_dae.profiling.raw_assets import build_manifest, write_raw_npz
from torch_dae.profiling.timing import summarize_timing


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture
def repository_root() -> Path:
    return _repository_root()


@pytest.fixture
def real_model_card_path(repository_root: Path) -> Path:
    return repository_root / "model_cards/panns/panns-cnn14-16k-map-0438.json"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def technical_card_workspace(repository_root: Path):
    # Repo-relative but outside both the ignored `.torch-dae/` tree (whose paths cannot satisfy
    # the strict repo-relative artifact pattern) and the official `technical_cards/` tree.
    workspace = repository_root / "candidate_technical_cards_pytest"
    workspace.mkdir(parents=True, exist_ok=True)
    yield workspace
    shutil.rmtree(workspace, ignore_errors=True)


def build_valid_card(
    *,
    repository_root: Path,
    model_card_path: Path,
    workspace: Path,
    technical_card_id_seed: str = "seed-1",
) -> tuple[TechnicalCard, Path]:
    """Build one fully valid Technical Card + `.npz` pair inside `workspace` (repo-relative)."""

    import json as _json

    real_checkpoint_sha256 = _json.loads(model_card_path.read_text())["checkpoint"][
        "observed_sha256"
    ]

    raw_ns = [1_000_000 + i * 1_000 for i in range(50)]
    npz_path = workspace / f"tc-fixture-{technical_card_id_seed}.npz"
    arrays = {"raw_ns__canonical-b1": np.asarray(raw_ns, dtype=np.int64)}
    write_raw_npz(npz_path, arrays)
    manifest = build_manifest(
        card_relative_path=npz_path.name,
        path=npz_path,
        arrays=arrays,
    )

    software = SoftwareMetadata(
        os_name="Darwin",
        os_version="25.5.0",
        python_implementation="CPython",
        python_version="3.12.13",
        torch_version="2.13.0",
    )
    hardware = HardwareMetadata(cpu_architecture="arm64", cpu_model="Apple M4 Pro")
    execution_context = ExecutionContextMetadata(
        hardware_fingerprint="a" * 64,
        execution_context_fingerprint="b" * 64,
        os_name=software.os_name,
        os_version=software.os_version,
        python_version=software.python_version,
        torch_version=software.torch_version,
        torch_dae_content_identity="content-sha256:" + "c" * 64,
        device_backend=DeviceBackend.CPU,
        native_precision="float32",
        thread_regime=ThreadRegime.NATIVE_DEFAULT,
        profiling_protocol_id=PROFILING_PROTOCOL_ID,
        profiling_protocol_version=PROFILING_PROTOCOL_VERSION,
        profiler_implementation_version=PROFILER_IMPLEMENTATION_VERSION,
    )
    identity_inputs = TechnicalCardIdentityInputs(
        technical_card_schema_version=TECHNICAL_CARD_SCHEMA_VERSION,
        model_card_id="panns-cnn14-16k-map-0438",
        model_card_sha256=_sha256_file(model_card_path),
        checkpoint_sha256=real_checkpoint_sha256,
        profiling_protocol_id=PROFILING_PROTOCOL_ID,
        profiling_protocol_version=PROFILING_PROTOCOL_VERSION,
        torch_dae_content_identity="content-sha256:" + "c" * 64,
        source_revision="e" * 40,
        repository_dirty=False,
        hardware_fingerprint=execution_context.hardware_fingerprint,
        execution_context_fingerprint=execution_context.execution_context_fingerprint,
        device_backend="cpu",
        device_index=None,
        nonce=f"nonce-{technical_card_id_seed}".ljust(16, "0"),
    )
    identity = build_technical_card_identity(identity_inputs)
    timing = summarize_timing(raw_ns, batch_size=1, input_duration_seconds=10.0)
    condition = ProfilingConditions(
        condition_id="canonical-b1",
        duration_seconds=10.0,
        duration_source="protocol_default",
        batch_size=1,
        sample_count=160_000,
        status=ConditionStatus.SUCCESS,
        timing=timing,
        raw_timing_array="raw_ns__canonical-b1",
    )
    card = TechnicalCard(
        technical_card_schema_version=TECHNICAL_CARD_SCHEMA_VERSION,
        identity=identity,
        model=ModelReference(
            model_card_id="panns-cnn14-16k-map-0438",
            model_card_path=model_card_path.relative_to(repository_root).as_posix(),
            model_card_sha256=identity_inputs.model_card_sha256,
            model_card_status="runtime_verified",
            checkpoint_id="panns-cnn14-16k-map-0438",
            checkpoint_sha256=real_checkpoint_sha256,
            wrapper_entry_point="torch_dae.models.panns:PannsCnn14_16kMap0438",
        ),
        profiler=ProfilerImplementationProvenance(
            profiler_implementation_version=PROFILER_IMPLEMENTATION_VERSION,
            profiling_protocol_id=PROFILING_PROTOCOL_ID,
            profiling_protocol_version=PROFILING_PROTOCOL_VERSION,
            torch_dae_package_version="0.1.0",
            torch_dae_content_identity="content-sha256:" + "c" * 64,
            torch_dae_repository_head="e" * 40,
            torch_dae_repository_dirty=False,
            runtime_classification=RuntimeClassification.CANONICAL,
        ),
        contributor=ContributorMetadata(),
        hardware=hardware,
        software=software,
        device=DeviceBackendMetadata(
            backend=DeviceBackend.CPU, device_index=None, device_label="cpu"
        ),
        execution_context=execution_context,
        precision=PrecisionMetadata(dtype="float32"),
        synthetic_input=SyntheticInputProvenance(
            prng_implementation="numpy.random.Generator(PCG64)",
            seed=1337,
            sample_rate=16_000,
            sample_count=160_000,
            batch_size=1,
            channel_count=1,
        ),
        canonical_duration_seconds=10.0,
        canonical_duration_source="protocol_default",
        architecture=ArchitectureEvidence(
            total_parameters=1_000,
            trainable_parameters=1_000,
            non_trainable_parameters=0,
            parameter_bytes=4_000,
            buffer_bytes=0,
            state_dict_tensor_bytes=4_000,
            dtype_distribution={"torch.float32": 1_000},
            module_count=5,
        ),
        cold_start=ColdStartEvidence(),
        minimum_input_search=MinimumInputSearchResult(
            status=MinimumInputStatus.FOUND,
            minimum_sample_count=4_960,
            minimum_duration_seconds=0.31,
            boundary_verified=True,
            below_boundary_verified=True,
        ),
        conditions=(condition,),
        host_memory=HostMemoryEvidence(),
        accelerator_memory=None,
        energy=EnergyEvidence(measurement_kind=EnergyMeasurementKind.UNAVAILABLE),
        raw_measurements=manifest,
        coverage=MeasurementCoverage(
            tested_conditions=("canonical-b1",),
            architecture_status="unavailable",
            energy_status=EnergyMeasurementKind.UNAVAILABLE,
            host_memory_available=False,
            accelerator_memory_available=False,
        ),
        comparability=ComparabilityMetadata(
            protocol_id=PROFILING_PROTOCOL_ID,
            protocol_version=PROFILING_PROTOCOL_VERSION,
            execution_context_fingerprint=execution_context.execution_context_fingerprint,
            runtime_classification=RuntimeClassification.CANONICAL,
        ),
        campaign=ProfilingCampaignReference(campaign_id="campaign-1", run_id="run-1"),
        supersession=SupersessionMetadata(),
        created_at=datetime.now(UTC).isoformat(),
    )
    card_path = workspace / f"{identity.technical_card_id}-{technical_card_id_seed}.json"
    card_path.write_text(card.model_dump_json(indent=2))
    return card, card_path
