"""Deterministic onboarding handoff, bundle, and managed-workspace operations."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import posixpath
import shutil
import stat
import subprocess
import tarfile
import tempfile
import uuid
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import ValidationError

from torch_dae.contracts import contained_path
from torch_dae.environment.runtime import write_json_atomic
from torch_dae.onboarding.contracts import (
    CleanupConsumedRunManifest,
    CleanupExternalProtectionConflict,
    CleanupPathRecord,
    CleanupReceipt,
    CleanupRetentionConflict,
    EnvironmentResolutionReport,
    HandoffArtifactReference,
    HandoffStatus,
    ManagedRunManifest,
    OnboardingPhase,
    PhaseHandoffManifest,
    WorkflowRecord,
    WorkflowStatus,
)

ONBOARDING_REPORTS = "onboarding_reports"
WORKSPACE_ROOT = ".torch-dae/workspaces"
PHASE_ORDER = (
    OnboardingPhase.ANALYZE,
    OnboardingPhase.RESOLVE_ENVIRONMENT,
    OnboardingPhase.INTEGRATE,
    OnboardingPhase.VERIFY,
    OnboardingPhase.CARD,
)
PHASE_INDEX = {phase: index for index, phase in enumerate(PHASE_ORDER)}
FORBIDDEN_BUNDLE_PARTS = {
    ".git",
    ".torch-dae",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "build",
    "dist",
    "htmlcov",
    "__MACOSX",
}
FORBIDDEN_BUNDLE_NAMES = {".DS_Store", ".coverage", "coverage.xml", "coverage.json"}
FORBIDDEN_BUNDLE_SUFFIXES = {
    ".ckpt",
    ".coverage",
    ".egg",
    ".gz",
    ".onnx",
    ".pth",
    ".pt",
    ".pyc",
    ".tar",
    ".whl",
    ".zip",
}
SHARED_CONTROL_PLANE_ARTIFACTS = frozenset(
    {
        ".gitattributes",
        "CHANGELOG.md",
        "README.md",
        "docs/api/public-api.toml",
        "docs/development/testing.md",
        "docs/index.md",
        "docs/reference/cli.md",
        "docs/skill/integrate.md",
        "pyproject.toml",
        "schemas/phase-handoff.schema.json",
        "scripts/check_worktree_patch.py",
        "scripts/validate_repository.py",
        "skills/audio-model-onboarding/SKILL.md",
        "skills/audio-model-onboarding/references/integration-planning.md",
        "skills/audio-model-onboarding/templates/agent-request.md",
        "skills/audio-model-onboarding/templates/agent-response.md",
        "src/torch_dae/contracts.py",
        "src/torch_dae/environment/manager.py",
        "src/torch_dae/onboarding/contracts.py",
        "src/torch_dae/onboarding/handoff.py",
        "tests/onboarding/test_handoff_management.py",
        "tests/test_check_worktree_patch.py",
        "tests/test_public_metadata.py",
    }
)


class HandoffManagementError(ValueError):
    """Expected, traceback-free handoff management failure."""


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_text(value: object) -> str:
    """Serialize a committed control record deterministically."""

    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def skill_fingerprint(repository_root: Path) -> str:
    """Hash the canonical skill tree with path framing."""

    skill_root = repository_root / "skills/audio-model-onboarding"
    if not skill_root.is_dir():
        raise HandoffManagementError("canonical audio-model-onboarding skill is missing")
    digest = hashlib.sha256()
    for path in sorted(
        (
            item
            for item in skill_root.rglob("*")
            if item.is_file()
            and not item.is_symlink()
            and "__pycache__" not in item.parts
            and item.suffix != ".pyc"
        ),
        key=lambda item: item.relative_to(skill_root).as_posix(),
    ):
        relative = path.relative_to(skill_root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def discover_repository_root(start: Path | None = None) -> Path:
    """Locate the repository containing the normative specification."""

    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "project_spec.md").is_file():
            return candidate
    raise HandoffManagementError("could not discover repository root")


def load_workflow(path: Path) -> WorkflowRecord:
    """Load a strict workflow record."""

    try:
        return WorkflowRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        raise HandoffManagementError(f"invalid workflow record {path}: {exc}") from exc


def load_handoff(path: Path) -> PhaseHandoffManifest:
    """Load a strict phase-handoff manifest."""

    try:
        return PhaseHandoffManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        raise HandoffManagementError(f"invalid handoff manifest {path}: {exc}") from exc


def validate_workflow(
    repository_root: Path,
    workflow_id: str,
    *,
    phase: OnboardingPhase | None = None,
    workflow_root_override: Path | None = None,
    pending_candidate_phase: OnboardingPhase | None = None,
) -> dict[str, object]:
    """Validate accepted history and, when supplied, one pending promotion candidate."""

    canonical_root = repository_root / ONBOARDING_REPORTS / workflow_id
    workflow_root = workflow_root_override or canonical_root
    workflow = load_workflow(workflow_root / "workflow.json")
    if workflow.workflow_id != workflow_id:
        raise HandoffManagementError("workflow ID does not match its directory")
    references = {item.phase: item for item in workflow.accepted_phase_paths}
    accepted_phases = tuple(sorted(references, key=PHASE_INDEX.__getitem__))
    selected_phases = (phase,) if phase is not None else accepted_phases
    if phase is not None and phase not in references:
        raise HandoffManagementError(
            f"required phase is missing for workflow {workflow_id}: {phase.value}"
        )
    errors: list[str] = []
    validated: list[str] = []
    expected_spec_hash = sha256_file(repository_root / "project_spec.md")
    expected_skill_hash = skill_fingerprint(repository_root)
    handoffs: dict[OnboardingPhase, PhaseHandoffManifest] = {}
    handoff_hashes: dict[OnboardingPhase, str] = {}
    for accepted_phase in accepted_phases:
        reference = references[accepted_phase]
        handoff_path = _resolve_artifact_path(
            repository_root,
            workflow_id,
            reference.handoff_path,
            workflow_root,
        )
        if not handoff_path.is_file():
            errors.append(f"accepted handoff is missing: {reference.handoff_path}")
            continue
        try:
            handoff = load_handoff(handoff_path)
        except HandoffManagementError as exc:
            errors.append(str(exc))
            continue
        if handoff.workflow_id != workflow_id or handoff.phase != accepted_phase:
            errors.append(f"handoff identity mismatch: {reference.handoff_path}")
        if handoff.handoff_status != HandoffStatus.ACCEPTED:
            errors.append(f"canonical handoff is not accepted: {reference.handoff_path}")
        if accepted_phase == pending_candidate_phase:
            if handoff.project_spec_sha256 != expected_spec_hash:
                errors.append(
                    "pending candidate project specification hash mismatch: "
                    f"{reference.handoff_path}"
                )
            if handoff.canonical_skill_fingerprint != expected_skill_hash:
                errors.append(
                    "pending candidate canonical skill fingerprint mismatch: "
                    f"{reference.handoff_path}"
                )
        if (
            handoff.target_variant_ids != workflow.target_variant_ids
            or handoff.target_checkpoint_ids != workflow.target_checkpoint_ids
            or handoff.target_card_ids != workflow.target_card_ids
        ):
            errors.append(f"handoff target scope mismatch: {reference.handoff_path}")
        handoffs[accepted_phase] = handoff
        handoff_hashes[accepted_phase] = sha256_file(handoff_path)
        if accepted_phase in selected_phases:
            validated.append(reference.handoff_path)

    supersession_errors, latest_external, supersessions = _validate_artifact_supersession_chains(
        handoffs,
        handoff_hashes,
    )
    errors.extend(supersession_errors)
    for selected_phase in selected_phases:
        selected_handoff = handoffs.get(selected_phase)
        if selected_handoff is None:
            continue
        errors.extend(
            _validate_handoff_artifacts(
                repository_root,
                workflow_id,
                workflow_root,
                selected_handoff,
                latest_external=latest_external,
            )
        )
    errors.extend(
        _validate_latest_external_artifacts(
            repository_root,
            latest_external,
        )
    )
    if errors:
        raise HandoffManagementError("; ".join(sorted(set(errors))))
    historical_control_planes = [
        _historical_control_plane(
            handoffs[item],
            current_spec_hash=expected_spec_hash,
            current_skill_hash=expected_skill_hash,
        )
        for item in accepted_phases
        if item in handoffs
    ]
    selected_control_planes = [
        item
        for item in historical_control_planes
        if item["phase"] in {selected.value for selected in selected_phases}
    ]
    current_control_plane = {
        "canonical_skill_fingerprint": expected_skill_hash,
        "project_spec_sha256": expected_spec_hash,
    }
    control_plane_drift = _control_plane_drift_summary(selected_control_planes)
    selected_historical_phase = phase or workflow.current_accepted_phase
    selected_historical_control_plane = (
        next(
            (
                item
                for item in historical_control_planes
                if item["phase"] == selected_historical_phase.value
            ),
            None,
        )
        if selected_historical_phase is not None
        else None
    )
    return {
        "valid": True,
        "workflow_id": workflow_id,
        "workflow_status": workflow.status.value,
        "validated_phases": [
            item.value for item in sorted(selected_phases, key=PHASE_INDEX.__getitem__)
        ],
        "validated_handoffs": validated,
        "project_spec_sha256": expected_spec_hash,
        "canonical_skill_fingerprint": expected_skill_hash,
        "historical_control_plane": selected_historical_control_plane,
        "historical_control_planes": historical_control_planes,
        "current_control_plane": current_control_plane,
        **control_plane_drift,
        "validated_supersession_count": len(supersessions),
        "superseded_artifact_paths": sorted({str(item["path"]) for item in supersessions}),
        "artifact_supersessions": supersessions,
        "current_external_artifacts": [
            {
                "path": path,
                "originating_phase": state[0].value,
                "sha256": state[1],
                "validation_scope": (
                    "historical-shared-control-plane"
                    if path in SHARED_CONTROL_PLANE_ARTIFACTS
                    else "current-worktree"
                ),
            }
            for path, state in sorted(latest_external.items())
        ],
    }


def _historical_control_plane(
    handoff: PhaseHandoffManifest,
    *,
    current_spec_hash: str,
    current_skill_hash: str,
) -> dict[str, object]:
    """Describe one immutable accepted handoff's declared control-plane identity."""

    canonical_skill_drift = handoff.canonical_skill_fingerprint != current_skill_hash
    project_spec_drift = handoff.project_spec_sha256 != current_spec_hash
    return {
        "phase": handoff.phase.value,
        "canonical_skill_fingerprint": handoff.canonical_skill_fingerprint,
        "project_spec_sha256": handoff.project_spec_sha256,
        "canonical_skill_drift": canonical_skill_drift,
        "project_spec_drift": project_spec_drift,
        "control_plane_drift": canonical_skill_drift or project_spec_drift,
    }


def _control_plane_drift_summary(
    historical_control_planes: Sequence[dict[str, object]],
) -> dict[str, bool]:
    """Summarize skill and specification drift without treating either as corruption."""

    canonical_skill_drift = any(
        item["canonical_skill_drift"] is True for item in historical_control_planes
    )
    project_spec_drift = any(
        item["project_spec_drift"] is True for item in historical_control_planes
    )
    return {
        "canonical_skill_drift": canonical_skill_drift,
        "project_spec_drift": project_spec_drift,
        "control_plane_drift": canonical_skill_drift or project_spec_drift,
    }


def _validate_artifact_supersession_chains(
    handoffs: dict[OnboardingPhase, PhaseHandoffManifest],
    handoff_hashes: dict[OnboardingPhase, str],
) -> tuple[
    list[str],
    dict[str, tuple[OnboardingPhase, str]],
    list[dict[str, object]],
]:
    """Validate ordered external-output declarations and explicit hash transitions."""

    errors: list[str] = []
    latest: dict[str, tuple[OnboardingPhase, str]] = {}
    supersessions: list[dict[str, object]] = []
    for phase in sorted(handoffs, key=PHASE_INDEX.__getitem__):
        handoff = handoffs[phase]
        declared = {item.path: item for item in handoff.artifact_supersessions}
        external_outputs = [
            item
            for item in handoff.output_artifacts
            if item.path is not None and not item.path.startswith(f"{ONBOARDING_REPORTS}/")
        ]
        output_by_path = {item.path: item for item in external_outputs}
        for path, transition in declared.items():
            prior = latest.get(path)
            if prior is None:
                errors.append(f"artifact supersession has no earlier accepted declaration: {path}")
                continue
            if transition.prior_originating_phase != prior[0]:
                errors.append(f"artifact supersession prior phase is not latest for path: {path}")
            if transition.prior_sha256 != prior[1]:
                errors.append(f"artifact supersession prior_sha256 is stale for path: {path}")
            if transition.prior_handoff_sha256 is not None:
                expected_handoff_hash = handoff_hashes.get(transition.prior_originating_phase)
                if transition.prior_handoff_sha256 != expected_handoff_hash:
                    errors.append(f"artifact supersession prior handoff hash mismatch: {path}")
            output = output_by_path.get(path)
            if output is None:
                errors.append(f"artifact supersession lacks a later output declaration: {path}")
                continue
            if output.sha256 != transition.new_sha256:
                errors.append(f"artifact supersession new_sha256 disagrees with output: {path}")
            supersessions.append(
                {
                    **transition.model_dump(mode="json"),
                    "superseding_phase": phase.value,
                }
            )
        for output in external_outputs:
            assert output.path is not None
            prior = latest.get(output.path)
            declared_transition = declared.get(output.path)
            if prior is not None and output.sha256 != prior[1] and declared_transition is None:
                errors.append(f"changed artifact lacks explicit supersession: {output.path}")
            latest[output.path] = (phase, output.sha256)
    return errors, latest, supersessions


def _validate_handoff_artifacts(
    repository_root: Path,
    workflow_id: str,
    workflow_root: Path,
    handoff: PhaseHandoffManifest,
    *,
    latest_external: dict[str, tuple[OnboardingPhase, str]],
) -> list[str]:
    errors: list[str] = []
    local_artifacts = [
        item
        for item in (*handoff.input_artifacts, *handoff.output_artifacts)
        if item.path is not None
    ]
    for artifact in local_artifacts:
        assert artifact.path is not None
        if not artifact.path.startswith(f"{ONBOARDING_REPORTS}/"):
            if artifact.path in latest_external:
                continue
        path = _resolve_artifact_path(
            repository_root,
            workflow_id,
            artifact.path,
            workflow_root,
        )
        if not path.is_file():
            errors.append(f"referenced artifact is missing: {artifact.path}")
            continue
        if sha256_file(path) != artifact.sha256:
            errors.append(f"artifact hash mismatch: {artifact.path}")
    if (
        handoff.lifecycle_promotion is not None
        and handoff.phase == OnboardingPhase.RESOLVE_ENVIRONMENT
    ):
        reports = [
            item
            for item in handoff.output_artifacts
            if item.canonical_role == "environment-resolution-report" and item.path is not None
        ]
        if len(reports) != 1:
            errors.append("environment lifecycle promotion requires one resolution report")
        else:
            report_path = _resolve_artifact_path(
                repository_root,
                workflow_id,
                reports[0].path or "",
                workflow_root,
            )
            try:
                report = EnvironmentResolutionReport.model_validate_json(
                    report_path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                errors.append(f"environment resolution report is invalid: {exc}")
            else:
                if report.next_lifecycle_status != handoff.lifecycle_promotion:
                    errors.append("handoff lifecycle promotion is not claimed by its report")
    return errors


def _validate_latest_external_artifacts(
    repository_root: Path,
    latest_external: dict[str, tuple[OnboardingPhase, str]],
) -> list[str]:
    """Validate the filesystem only against each accepted chain's latest declaration."""

    errors: list[str] = []
    for relative, state in sorted(latest_external.items()):
        if relative in SHARED_CONTROL_PLANE_ARTIFACTS:
            continue
        path = contained_path(repository_root, relative)
        if path.is_symlink() or not path.is_file():
            errors.append(f"referenced artifact is missing: {relative}")
            continue
        if sha256_file(path) != state[1]:
            errors.append(f"artifact hash mismatch: {relative}")
    return errors


def _resolve_artifact_path(
    repository_root: Path,
    workflow_id: str,
    relative: str,
    workflow_root: Path,
) -> Path:
    prefix = f"{ONBOARDING_REPORTS}/{workflow_id}/"
    if relative.startswith(prefix):
        suffix = relative.removeprefix(prefix)
        return contained_path(workflow_root, suffix)
    return contained_path(repository_root, relative)


def discover_handoff(
    repository_root: Path,
    *,
    workflow_id: str | None,
    required_phase: OnboardingPhase,
    attachments: Sequence[Path] = (),
    include_superseded: bool = False,
) -> dict[str, object]:
    """Find and validate one accepted prerequisite handoff."""

    selected_id = workflow_id or _auto_select_workflow(repository_root, required_phase)
    validation = validate_workflow(repository_root, selected_id, phase=required_phase)
    workflow_root = repository_root / ONBOARDING_REPORTS / selected_id
    workflow = load_workflow(workflow_root / "workflow.json")
    reference = next(item for item in workflow.accepted_phase_paths if item.phase == required_phase)
    handoff = load_handoff(
        _resolve_artifact_path(
            repository_root,
            selected_id,
            reference.handoff_path,
            workflow_root,
        )
    )
    canonical = [
        item.path for item in (*handoff.input_artifacts, *handoff.output_artifacts) if item.path
    ]
    current_entries = cast(
        list[dict[str, object]],
        validation["current_external_artifacts"],
    )
    current_external = {str(item["path"]): item for item in current_entries}
    superseded_external: list[dict[str, object]] = []
    superseded_paths = set(cast(list[str], validation["superseded_artifact_paths"]))
    for artifact in handoff.output_artifacts:
        if artifact.path is None or artifact.path not in superseded_paths:
            continue
        current = current_external[artifact.path]
        if current["sha256"] == artifact.sha256:
            continue
        superseded_external.append(
            {
                "path": artifact.path,
                "historical_originating_phase": handoff.phase.value,
                "historical_sha256": artifact.sha256,
                "latest_originating_phase": current["originating_phase"],
                "latest_sha256": current["sha256"],
                "status": "accepted_supersession",
            }
        )
    _validate_duplicate_attachments(repository_root, handoff, attachments)
    payload: dict[str, object] = {
        **validation,
        "handoff_path": reference.handoff_path,
        "canonical_artifact_paths": canonical,
        "output_artifact_paths": [item.path for item in handoff.output_artifacts],
        "allowed_next_modes": [item.value for item in handoff.allowed_next_modes],
        "unresolved_items": [
            item.model_dump(mode="json") for item in handoff.unresolved_items_carried_forward
        ],
        "user_decisions": [
            item.model_dump(mode="json") for item in handoff.user_decisions_consumed
        ],
        "superseded_external_artifacts": superseded_external,
    }
    if include_superseded:
        payload["superseded_handoffs"] = [
            path.relative_to(repository_root).as_posix()
            for path in sorted(
                workflow_root.joinpath(required_phase.value).glob("handoff.*.superseded.json")
            )
        ]
    return payload


def _auto_select_workflow(repository_root: Path, phase: OnboardingPhase) -> str:
    root = repository_root / ONBOARDING_REPORTS
    compatible: list[str] = []
    if root.is_dir():
        for path in sorted(root.glob("*/workflow.json")):
            try:
                workflow = load_workflow(path)
            except HandoffManagementError:
                continue
            if workflow.status == WorkflowStatus.ACTIVE and any(
                item.phase == phase for item in workflow.accepted_phase_paths
            ):
                compatible.append(workflow.workflow_id)
    if not compatible:
        raise HandoffManagementError(f"no active workflow has accepted phase {phase.value}")
    if len(compatible) != 1:
        raise HandoffManagementError(f"ambiguous active workflows for {phase.value}: {compatible}")
    return compatible[0]


def _validate_duplicate_attachments(
    repository_root: Path,
    handoff: PhaseHandoffManifest,
    attachments: Sequence[Path],
) -> None:
    by_name: dict[str, list[HandoffArtifactReference]] = {}
    for artifact in (*handoff.input_artifacts, *handoff.output_artifacts):
        if artifact.path:
            by_name.setdefault(Path(artifact.path).name, []).append(artifact)
    for attachment in attachments:
        matches = by_name.get(attachment.name, [])
        if len(matches) != 1:
            raise HandoffManagementError(
                f"duplicate attachment does not identify one canonical artifact: {attachment.name}"
            )
        if not attachment.is_file():
            raise HandoffManagementError(f"attachment is missing: {attachment}")
        if sha256_file(attachment) != matches[0].sha256:
            raise HandoffManagementError(
                f"attachment hash mismatch for canonical artifact: {attachment.name}"
            )


def promote_phase(
    repository_root: Path,
    *,
    source_dir: Path,
    workflow_id: str,
    phase: OnboardingPhase,
    supersede: bool = False,
) -> dict[str, object]:
    """Atomically promote one validated managed-workspace phase."""

    if source_dir.is_symlink():
        raise HandoffManagementError("promotion source must not be a symlink")
    source = source_dir.resolve()
    managed_parent = (repository_root / WORKSPACE_ROOT / workflow_id).resolve()
    try:
        source.relative_to(managed_parent)
    except ValueError as exc:
        raise HandoffManagementError(
            "promotion source must be in the selected managed workspace"
        ) from exc
    source_workflow = load_workflow(source / "workflow.json")
    if source_workflow.workflow_id != workflow_id:
        raise HandoffManagementError("promotion workflow identity mismatch")
    source_phase = source / phase.value
    source_handoff_path = source_phase / "handoff.json"
    source_handoff = load_handoff(source_handoff_path)
    if source_handoff.workflow_id != workflow_id or source_handoff.phase != phase:
        raise HandoffManagementError("promotion handoff identity mismatch")
    if source_handoff.handoff_status != HandoffStatus.ACCEPTED:
        raise HandoffManagementError("only accepted handoffs may be promoted")
    _validate_promotion_source(
        repository_root,
        source,
        workflow_id,
        phase,
        source_handoff,
    )

    reports_root = repository_root / ONBOARDING_REPORTS
    destination = reports_root / workflow_id
    reports_root.mkdir(exist_ok=True)
    token = uuid.uuid4().hex
    candidate = reports_root / f".{workflow_id}.promote-{token}"
    backup = reports_root / f".{workflow_id}.backup-{token}"
    if candidate.exists() or backup.exists():
        raise HandoffManagementError("promotion temporary path collision")
    previous_hash: str | None = None
    candidate_validation: dict[str, object] | None = None
    try:
        if destination.exists():
            shutil.copytree(destination, candidate, symlinks=True)
            existing_handoff = destination / phase.value / "handoff.json"
            if existing_handoff.exists():
                previous_hash = sha256_file(existing_handoff)
                if not supersede:
                    raise HandoffManagementError(
                        "accepted handoff already exists; explicit supersession is required"
                    )
                if source_handoff.superseded_handoff_sha256 != previous_hash:
                    raise HandoffManagementError(
                        "superseding handoff does not record the prior accepted handoff hash"
                    )
            candidate_phase = candidate / phase.value
            historical_handoffs = (
                tuple(sorted((destination / phase.value).glob("handoff.*.superseded.json")))
                if (destination / phase.value).is_dir()
                else ()
            )
            if candidate_phase.exists():
                shutil.rmtree(candidate_phase)
            shutil.copytree(source_phase, candidate_phase)
            for historical_handoff in historical_handoffs:
                shutil.copy2(historical_handoff, candidate_phase / historical_handoff.name)
            if previous_hash is not None:
                archive_path = candidate_phase / f"handoff.{previous_hash}.superseded.json"
                shutil.copy2(existing_handoff, archive_path)
            shutil.copy2(source / "workflow.json", candidate / "workflow.json")
        else:
            shutil.copytree(source, candidate, symlinks=True)
        _canonicalize_control_records(candidate, phase)
        candidate_validation = validate_workflow(
            repository_root,
            workflow_id,
            workflow_root_override=candidate,
            pending_candidate_phase=phase,
        )
        if destination.exists():
            os.replace(destination, backup)
        try:
            os.replace(candidate, destination)
        except Exception:
            if backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if candidate.exists():
            shutil.rmtree(candidate)
        if backup.exists() and destination.exists():
            shutil.rmtree(backup)
    assert candidate_validation is not None
    return {
        "promoted": True,
        "workflow_id": workflow_id,
        "phase": phase.value,
        "handoff_path": (destination / phase.value / "handoff.json")
        .relative_to(repository_root)
        .as_posix(),
        "superseded_handoff_sha256": previous_hash,
        "validated_supersession_count": candidate_validation["validated_supersession_count"],
        "superseded_artifact_paths": candidate_validation["superseded_artifact_paths"],
    }


def _canonicalize_control_records(workflow_root: Path, phase: OnboardingPhase) -> None:
    workflow_path = workflow_root / "workflow.json"
    handoff_path = workflow_root / phase.value / "handoff.json"
    workflow = load_workflow(workflow_path)
    handoff = load_handoff(handoff_path)
    workflow_path.write_text(
        canonical_json_text(workflow.model_dump(mode="json")),
        encoding="utf-8",
    )
    handoff_path.write_text(
        canonical_json_text(handoff.model_dump(mode="json")),
        encoding="utf-8",
    )


def _validate_promotion_source(
    repository_root: Path,
    source: Path,
    workflow_id: str,
    phase: OnboardingPhase,
    handoff: PhaseHandoffManifest,
) -> None:
    """Require the managed source to contain exactly the selected phase's declared files."""

    phase_prefix = f"{ONBOARDING_REPORTS}/{workflow_id}/{phase.value}/"
    workflow_prefix = f"{ONBOARDING_REPORTS}/{workflow_id}/"
    allowed_files = {
        source / "workflow.json",
        source / phase.value / "handoff.json",
    }
    for artifact in handoff.output_artifacts:
        if artifact.path is None:
            continue
        if artifact.path.startswith(phase_prefix):
            relative = artifact.path.removeprefix(workflow_prefix)
            managed_path = contained_path(source, relative)
            if managed_path.is_symlink() or not managed_path.is_file():
                raise HandoffManagementError(
                    f"declared phase-local output is missing: {artifact.path}"
                )
            if sha256_file(managed_path) != artifact.sha256:
                raise HandoffManagementError(
                    f"artifact hash mismatch for declared phase-local output: {artifact.path}"
                )
            allowed_files.add(managed_path)
        else:
            canonical_path = contained_path(repository_root, artifact.path)
            if canonical_path.is_symlink() or not canonical_path.is_file():
                raise HandoffManagementError(
                    f"declared external repository output is missing: {artifact.path}"
                )
            if sha256_file(canonical_path) != artifact.sha256:
                raise HandoffManagementError(
                    f"declared external repository output hash mismatch: {artifact.path}"
                )

    for path in source.rglob("*"):
        if path.is_symlink():
            raise HandoffManagementError("promotion source must not contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise HandoffManagementError(f"unsupported promotion source entry: {path}")
        if path not in allowed_files:
            raise HandoffManagementError(
                f"unreferenced promotion source artifact: {path.relative_to(source).as_posix()}"
            )
    missing = sorted(
        path.relative_to(source).as_posix()
        for path in allowed_files
        if path.is_symlink() or not path.is_file()
    )
    if missing:
        raise HandoffManagementError(f"required promotion source files are missing: {missing}")


def bundle_workflow(
    repository_root: Path,
    *,
    workflow_id: str,
    through_phase: OnboardingPhase,
    output_dir: Path,
    include_working_tree: bool,
) -> dict[str, object]:
    """Build a normalized deterministic review bundle exclusively with Python tarfile."""

    validation = validate_workflow(repository_root, workflow_id)
    workflow_root = repository_root / ONBOARDING_REPORTS / workflow_id
    workflow = load_workflow(workflow_root / "workflow.json")
    included_phases = [
        item.phase
        for item in workflow.accepted_phase_paths
        if PHASE_INDEX[item.phase] <= PHASE_INDEX[through_phase]
    ]
    if through_phase not in included_phases:
        raise HandoffManagementError(
            f"through phase is not accepted for workflow {workflow_id}: {through_phase.value}"
        )
    included_phase_values = {item.value for item in included_phases}
    historical_control_planes = [
        item
        for item in cast(list[dict[str, object]], validation["historical_control_planes"])
        if item["phase"] in included_phase_values
    ]
    current_control_plane = cast(dict[str, str], validation["current_control_plane"])
    control_plane_drift = _control_plane_drift_summary(historical_control_planes)
    output = output_dir.resolve()
    if output == repository_root or repository_root in output.parents:
        raise HandoffManagementError("review bundle output must be outside the repository")
    output.mkdir(parents=True, exist_ok=True)
    archive_name = f"{workflow_id}-through-{through_phase.value}.tar.gz"
    archive_path = output / archive_name
    result_path = output / f"{archive_name}.result.json"
    checksum_sidecar_path = output / f"{archive_name}.sha256"
    branch = _git(repository_root, "branch", "--show-current").strip()
    head = _git(repository_root, "rev-parse", "HEAD").strip()
    status_full = _git(repository_root, "status", "--untracked-files=all")
    status_porcelain = _git(repository_root, "status", "--porcelain=v1", "-uall")

    with tempfile.TemporaryDirectory(prefix="torch-dae-bundle-", dir=output) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        metadata = staging / "metadata"
        metadata.mkdir()
        _write_text(metadata / "branch.txt", branch + "\n")
        _write_text(metadata / "head.txt", head + "\n")
        _write_text(metadata / "status-full.txt", status_full)
        _write_text(metadata / "status-porcelain.txt", status_porcelain)
        _write_text(metadata / "staged.diff", _git(repository_root, "diff", "--cached", "--binary"))
        _write_text(metadata / "unstaged.diff", _git(repository_root, "diff", "--binary"))
        _write_text(
            metadata / "staged-files.txt",
            _git(repository_root, "diff", "--cached", "--name-only"),
        )
        _write_text(
            metadata / "unstaged-files.txt",
            _git(repository_root, "diff", "--name-only"),
        )
        _write_text(
            metadata / "untracked-files.txt",
            _git(repository_root, "ls-files", "--others", "--exclude-standard"),
        )
        _write_text(
            metadata / "project-spec.sha256",
            f"{sha256_file(repository_root / 'project_spec.md')}  project_spec.md\n",
        )
        _write_text(
            metadata / "skill-fingerprint.txt",
            f"{skill_fingerprint(repository_root)}  skills/audio-model-onboarding\n",
        )
        _write_json(
            metadata / "artifact-supersessions.json",
            validation["artifact_supersessions"],
        )
        _write_json(
            metadata / "current-external-artifacts.json",
            validation["current_external_artifacts"],
        )
        _write_json(
            metadata / "historical-control-planes.json",
            historical_control_planes,
        )
        _write_json(metadata / "current-control-plane.json", current_control_plane)
        _write_json(metadata / "control-plane-drift.json", control_plane_drift)

        artifact_paths = _bundle_artifact_paths(
            repository_root,
            workflow,
            included_phases,
        )
        artifacts_root = staging / "artifacts"
        for relative in artifact_paths:
            _copy_snapshot_entry(
                repository_root / relative,
                artifacts_root / relative,
                relative,
                repository_root=repository_root,
            )
        if include_working_tree:
            _snapshot_working_tree(repository_root, staging / "repository")
        tracked_manifest = _tracked_file_manifest(repository_root)
        _write_json(metadata / "tracked-file-manifest.json", tracked_manifest)
        artifact_manifest = _artifact_manifest(staging)
        _write_json(metadata / "artifact-manifest.json", artifact_manifest)
        internal_result = {
            "schema_version": "1.0.0",
            "archive_name": archive_name,
            "workflow_id": workflow_id,
            "included_phases": [item.value for item in included_phases],
            "repository_head": head,
            "repository_branch": branch,
            "repository_clean": not bool(status_porcelain.strip()),
            "include_working_tree": include_working_tree,
            "artifact_manifest_scope": {
                "includes": "all staged archive files created before artifact-manifest.json",
                "self_excluded_metadata_records": [
                    "metadata/artifact-manifest.json",
                    "metadata/bundle-result.json",
                    "metadata/declared-archive-inventory.json",
                    "metadata/actual-archive-inventory.json",
                ],
            },
            "validated_supersession_count": validation["validated_supersession_count"],
            "superseded_artifact_paths": validation["superseded_artifact_paths"],
            "historical_control_planes": historical_control_planes,
            "current_control_plane": current_control_plane,
            **control_plane_drift,
        }
        _write_json(metadata / "bundle-result.json", internal_result)
        inventory_names = _staging_member_names(staging)
        inventory_names.extend(
            [
                "metadata/actual-archive-inventory.json",
                "metadata/declared-archive-inventory.json",
            ]
        )
        final_inventory = sorted(set(inventory_names))
        _write_json(metadata / "declared-archive-inventory.json", final_inventory)
        _write_json(metadata / "actual-archive-inventory.json", final_inventory)
        if _staging_member_names(staging) != final_inventory:
            raise HandoffManagementError("declared bundle inventory does not match staged files")
        _write_normalized_tar_gz(staging, archive_path)

    actual_inventory = _archive_inventory(archive_path)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        declared_member = archive.extractfile("metadata/declared-archive-inventory.json")
        if declared_member is None:
            raise HandoffManagementError("archive lacks declared inventory")
        declared_inventory = json.loads(declared_member.read())
    if actual_inventory != declared_inventory:
        archive_path.unlink(missing_ok=True)
        raise HandoffManagementError("declared and actual archive inventories differ")
    gzip_metadata_normalized = _gzip_metadata_normalized(archive_path)
    if not gzip_metadata_normalized:
        archive_path.unlink(missing_ok=True)
        raise HandoffManagementError("gzip metadata is not normalized")
    archive_sha256 = sha256_file(archive_path)
    _write_text_atomic(
        checksum_sidecar_path,
        f"{archive_sha256}  {archive_path.name}\n",
    )
    result: dict[str, object] = {
        "archive_path": str(archive_path),
        "result_path": str(result_path),
        "result_json_path": str(result_path),
        "checksum_sidecar_path": str(checksum_sidecar_path),
        "sha256": archive_sha256,
        "archive_sha256": archive_sha256,
        "byte_size": archive_path.stat().st_size,
        "member_count": len(actual_inventory),
        "workflow": workflow_id,
        "included_workflow": workflow_id,
        "included_phases": [item.value for item in included_phases],
        "repository_head": head,
        "repository_branch": branch,
        "repository_clean": not bool(status_porcelain.strip()),
        "archive_clean": True,
        "archive_metadata_normalized": True,
        "gzip_metadata_normalized": gzip_metadata_normalized,
        "declared_actual_inventory_match": True,
        "validation": validation,
        "validated_supersession_count": validation["validated_supersession_count"],
        "superseded_artifact_paths": validation["superseded_artifact_paths"],
        "historical_control_planes": historical_control_planes,
        "current_control_plane": current_control_plane,
        **control_plane_drift,
        "declared_actual_inventory_equal": True,
    }
    write_json_atomic(result_path, result)
    return result


def _bundle_artifact_paths(
    repository_root: Path,
    workflow: WorkflowRecord,
    phases: Sequence[OnboardingPhase],
) -> list[str]:
    paths = {f"{ONBOARDING_REPORTS}/{workflow.workflow_id}/workflow.json"}
    for phase in phases:
        reference = next(item for item in workflow.accepted_phase_paths if item.phase == phase)
        paths.add(reference.handoff_path)
        handoff = load_handoff(repository_root / reference.handoff_path)
        phase_root = repository_root / ONBOARDING_REPORTS / workflow.workflow_id / phase.value
        paths.update(
            path.relative_to(repository_root).as_posix()
            for path in phase_root.glob("handoff.*.superseded.json")
            if path.is_file() and not path.is_symlink()
        )
        for artifact in (*handoff.input_artifacts, *handoff.output_artifacts):
            if artifact.path:
                paths.add(artifact.path)
    missing = [path for path in sorted(paths) if not (repository_root / path).is_file()]
    if missing:
        raise HandoffManagementError(f"bundle artifacts are missing: {missing}")
    return sorted(paths)


def _snapshot_working_tree(repository_root: Path, target: Path) -> None:
    entries = _git_z(repository_root, "ls-files", "-co", "--exclude-standard", "-z")
    for relative in entries:
        if _bundle_path_forbidden(relative):
            continue
        source = repository_root / relative
        if source.exists() or source.is_symlink():
            _copy_snapshot_entry(
                source,
                target / relative,
                relative,
                repository_root=repository_root,
            )


def _copy_snapshot_entry(
    source: Path,
    destination: Path,
    relative: str,
    *,
    repository_root: Path,
) -> None:
    if _bundle_path_forbidden(relative):
        raise HandoffManagementError(f"forbidden bundle input: {relative}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        link = os.readlink(source)
        try:
            source.resolve().relative_to(repository_root.resolve())
        except ValueError as exc:
            raise HandoffManagementError(f"unsafe symlink in bundle input: {relative}") from exc
        if Path(link).is_absolute():
            raise HandoffManagementError(f"unsafe symlink in bundle input: {relative}")
        destination.symlink_to(link)
    elif source.is_file():
        shutil.copyfile(source, destination)
        if source.stat().st_mode & stat.S_IXUSR:
            destination.chmod(0o755)
        else:
            destination.chmod(0o644)
    else:
        raise HandoffManagementError(f"bundle input is not a regular file: {relative}")


def _bundle_path_forbidden(relative: str) -> bool:
    path = Path(relative)
    if any(part in FORBIDDEN_BUNDLE_PARTS or part.startswith("._") for part in path.parts):
        return True
    if path.name in FORBIDDEN_BUNDLE_NAMES or path.name.endswith("~"):
        return True
    return path.suffix.lower() in FORBIDDEN_BUNDLE_SUFFIXES


def _tracked_file_manifest(repository_root: Path) -> list[dict[str, object]]:
    manifest: list[dict[str, object]] = []
    for relative in _git_z(repository_root, "ls-files", "-z"):
        path = repository_root / relative
        item: dict[str, object] = {"path": relative}
        if path.is_symlink():
            item["kind"] = "symlink"
            item["target"] = os.readlink(path)
        elif path.is_file():
            item["kind"] = "file"
            item["byte_size"] = path.stat().st_size
            item["sha256"] = sha256_file(path)
        else:
            item["kind"] = "missing"
        manifest.append(item)
    return manifest


def _artifact_manifest(staging: Path) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for relative in _staging_member_names(staging):
        path = staging / relative
        if path.is_symlink():
            target = os.readlink(path)
            items.append(
                {
                    "path": relative,
                    "kind": "symlink",
                    "target": target,
                    "sha256": hashlib.sha256(target.encode()).hexdigest(),
                }
            )
        else:
            items.append(
                {
                    "path": relative,
                    "kind": "file",
                    "byte_size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return items


def _staging_member_names(staging: Path) -> list[str]:
    return sorted(
        path.relative_to(staging).as_posix()
        for path in staging.rglob("*")
        if path.is_file() or path.is_symlink()
    )


def _write_normalized_tar_gz(staging: Path, archive_path: Path) -> None:
    temporary = archive_path.with_name(f".{archive_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
                with tarfile.open(
                    fileobj=compressed,
                    mode="w",
                    format=tarfile.GNU_FORMAT,
                ) as archive:
                    for relative in _staging_member_names(staging):
                        path = staging / relative
                        if path.is_symlink():
                            info = tarfile.TarInfo(relative)
                            info.type = tarfile.SYMTYPE
                            info.linkname = os.readlink(path)
                            info.size = 0
                            info.mode = 0o777
                            _normalize_tar_info(info)
                            archive.addfile(info)
                        else:
                            data = path.read_bytes()
                            info = tarfile.TarInfo(relative)
                            info.size = len(data)
                            info.mode = 0o755 if path.stat().st_mode & stat.S_IXUSR else 0o644
                            _normalize_tar_info(info)
                            archive.addfile(info, io.BytesIO(data))
        os.replace(temporary, archive_path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalize_tar_info(info: tarfile.TarInfo) -> None:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.pax_headers = {}


def _archive_inventory(path: Path) -> list[str]:
    with tarfile.open(path, mode="r:gz") as archive:
        members = archive.getmembers()
    names = sorted(member.name for member in members)
    if len(names) != len(set(names)):
        raise HandoffManagementError("archive contains duplicate members")
    for member in members:
        if member.name.startswith("/") or ".." in Path(member.name).parts:
            raise HandoffManagementError("archive contains unsafe member path")
        if _bundle_path_forbidden(member.name):
            raise HandoffManagementError(f"archive contains forbidden member: {member.name}")
        if member.uid != 0 or member.gid != 0 or member.mtime != 0:
            raise HandoffManagementError("archive metadata is not normalized")
        if member.uname or member.gname or member.pax_headers:
            raise HandoffManagementError("archive ownership or PAX metadata is not normalized")
        if member.issym():
            resolved_link = posixpath.normpath(
                posixpath.join(posixpath.dirname(member.name), member.linkname)
            )
            top_level = member.name.split("/", 1)[0]
            if resolved_link != top_level and not resolved_link.startswith(f"{top_level}/"):
                raise HandoffManagementError("archive contains escaping symlink")
    return names


def _gzip_metadata_normalized(path: Path) -> bool:
    header = path.read_bytes()[:10]
    if len(header) != 10 or header[:3] != b"\x1f\x8b\x08":
        return False
    flags = header[3]
    mtime = int.from_bytes(header[4:8], "little")
    return flags & 0x1E == 0 and mtime == 0


def cleanup_workflow(
    repository_root: Path,
    *,
    workflow_id: str,
    dry_run: bool,
    include_repository_caches: bool = False,
    include_package_caches: bool = False,
    include_environments: bool = False,
    include_checkpoints: bool = False,
) -> dict[str, object]:
    """Plan or execute safe cleanup using only paths recorded by managed run manifests."""

    workflow_workspace = repository_root / WORKSPACE_ROOT / workflow_id
    manifests = sorted(workflow_workspace.glob("*/*/run-manifest.json"))
    if not manifests:
        raise HandoffManagementError(f"no managed run manifests found for {workflow_id}")
    records: list[tuple[Path, ManagedRunManifest]] = []
    recorded_paths: set[Path] = set()
    explicitly_retained: dict[Path, tuple[str, str]] = {}
    external_paths: list[tuple[str, Path, Path]] = []
    for manifest_path in manifests:
        try:
            record = ManagedRunManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise HandoffManagementError(
                f"invalid managed run manifest: {manifest_path}: {exc}"
            ) from exc
        if record.workflow_id != workflow_id:
            raise HandoffManagementError("managed run manifest workflow identity mismatch")
        records.append((manifest_path, record))
        for raw in (*record.created_paths, *record.reused_paths):
            path = Path(raw)
            resolved = path.resolve() if path.is_absolute() else (repository_root / path).resolve()
            recorded_paths.add(resolved)
        for raw in record.external_paths:
            path = Path(raw)
            supplied = path if path.is_absolute() else repository_root / path
            supplied = Path(os.path.abspath(supplied))
            external_paths.append((raw, supplied, supplied.resolve()))
        for raw in record.retained_paths:
            path = Path(raw)
            resolved = path.resolve() if path.is_absolute() else (repository_root / path).resolve()
            explicitly_retained[resolved] = (
                raw,
                record.retained_reasons.get(
                    raw,
                    "explicitly retained by the managed run manifest",
                ),
            )

    run_roots = {manifest.parent.resolve() for manifest, _ in records}
    unrecorded_runs = sorted(run_roots - recorded_paths)
    if unrecorded_runs:
        raise HandoffManagementError(
            f"managed run root is not recorded in its manifest: {unrecorded_runs}"
        )
    selected: set[Path] = set(run_roots)
    explicit_categories = {
        "repository-cache": include_repository_caches,
        "package-cache": include_package_caches,
        "materialized-environment": include_environments,
        "checkpoint-cache": include_checkpoints,
    }
    retained_reasons_by_path: dict[Path, tuple[str, str, str]] = {}
    for path, (_, reason) in sorted(explicitly_retained.items()):
        category = _cleanup_category(repository_root, workflow_id, path)
        retained_reasons_by_path[path] = (
            str(path),
            category or "explicitly-retained",
            reason,
        )
    for path in sorted(recorded_paths):
        category = _cleanup_category(repository_root, workflow_id, path)
        if category in {"ephemeral-workspace", "trial-environment"}:
            selected.add(path)
        elif category is not None and explicit_categories.get(category, False):
            selected.add(path)
        elif category is not None:
            retained_reasons_by_path.setdefault(
                path,
                (
                    str(path),
                    category,
                    "destructive cache cleanup was not explicitly requested",
                ),
            )

    actions = _minimal_cleanup_roots(selected)
    for path in actions:
        if path not in recorded_paths:
            raise HandoffManagementError(f"refusing to delete unrecorded path: {path}")
        if _cleanup_category(repository_root, workflow_id, path) is None:
            raise HandoffManagementError(f"refusing to delete unclassified path: {path}")

    diagnostics_root = (repository_root / ".torch-dae/reports/onboarding" / workflow_id).resolve()
    conflicts: list[CleanupRetentionConflict] = []
    external_conflicts: list[CleanupExternalProtectionConflict] = []
    errors: list[str] = []
    for retained_path, (_, reason) in sorted(explicitly_retained.items()):
        conflicting_root = next(
            (
                deletion_root
                for deletion_root in actions
                if retained_path == deletion_root or deletion_root in retained_path.parents
            ),
            None,
        )
        if conflicting_root is not None:
            remediation = (
                f"move the retained diagnostic under {diagnostics_root} and update retained_paths "
                "in the run manifest before cleanup"
            )
            conflicts.append(
                CleanupRetentionConflict(
                    retained_path=str(retained_path),
                    deletion_root=str(conflicting_root),
                    reason=reason,
                    remediation=remediation,
                )
            )
            errors.append(
                f"retention conflict: {retained_path} is contained by planned deletion root "
                f"{conflicting_root}; {remediation}"
            )
        if retained_path != diagnostics_root and diagnostics_root not in retained_path.parents:
            errors.append(
                f"retained diagnostic must be moved under {diagnostics_root} and explicitly "
                f"recorded before cleanup: {retained_path}"
            )
        if not retained_path.exists() and not retained_path.is_symlink():
            errors.append(f"retained diagnostic is missing: {retained_path}")

    external_remediation = (
        "move the external audit output outside the managed deletion roots, update external_paths "
        "in the run manifest, and rerun cleanup"
    )
    for raw, supplied, resolved in sorted(external_paths, key=lambda item: item[0]):
        external_status = _cleanup_path_status(supplied)
        if external_status == "missing":
            continue
        for deletion_root in actions:
            supplied_overlap = _paths_overlap(supplied, deletion_root)
            resolved_overlap = _paths_overlap(resolved, deletion_root)
            if not supplied_overlap and not resolved_overlap:
                continue
            if supplied_overlap and resolved_overlap:
                overlap_source = "its supplied path and resolved target"
            elif supplied_overlap:
                overlap_source = "its supplied path"
            else:
                overlap_source = "its resolved target"
            reason = (
                f"existing external audit output overlaps the planned deletion root through "
                f"{overlap_source}"
            )
            external_conflicts.append(
                CleanupExternalProtectionConflict(
                    supplied_external_path=raw,
                    resolved_path=str(resolved),
                    deletion_root=str(deletion_root),
                    status=external_status,
                    reason=reason,
                    remediation=external_remediation,
                )
            )
            errors.append(
                f"external protection conflict: supplied path {raw}, resolved path {resolved}, "
                f"status {external_status}, planned deletion root {deletion_root}; "
                f"{external_remediation}"
            )

    retained_managed = _cleanup_path_records(retained_reasons_by_path)
    retained_external = tuple(
        _cleanup_path_record(
            raw,
            supplied,
            "external-audit-output",
            (
                "external audit output retained"
                if supplied.exists() or supplied.is_symlink()
                else "external audit output declared by the run manifest is missing"
            ),
        )
        for raw, supplied, _ in sorted(external_paths, key=lambda item: item[0])
    )
    repository_caches = _records_for_category(retained_managed, "repository-cache")
    package_caches = _records_for_category(retained_managed, "package-cache")
    materialized_environments = _records_for_category(
        retained_managed,
        "materialized-environment",
    )
    checkpoint_caches = _records_for_category(retained_managed, "checkpoint-cache")

    operation_id = f"cleanup-{uuid.uuid4().hex}"
    operation_time = datetime.now(UTC)
    mode: Literal["dry-run", "execute"] = "dry-run" if dry_run else "execute"
    receipt_path = diagnostics_root / "cleanup" / f"{operation_id}.json"
    planned_paths = tuple(str(path) for path in actions)
    retained_before = {path: path.exists() or path.is_symlink() for path in explicitly_retained}
    external_before = {
        (raw, supplied, resolved): (
            supplied.exists() or supplied.is_symlink(),
            resolved.exists() or resolved.is_symlink(),
        )
        for raw, supplied, resolved in external_paths
    }

    if dry_run or errors:
        status = "dry-run" if dry_run and not errors else "blocked"
        receipt = _cleanup_receipt(
            repository_root=repository_root,
            workflow_id=workflow_id,
            operation_id=operation_id,
            operation_time=operation_time,
            mode=mode,
            records=records,
            manifest_status=status,
            receipt_path=receipt_path,
            planned_paths=planned_paths,
            removed_paths=(),
            conflicts=tuple(conflicts),
            external_conflicts=tuple(external_conflicts),
            retained_managed=retained_managed,
            retained_external=retained_external,
            repository_caches=repository_caches,
            package_caches=package_caches,
            materialized_environments=materialized_environments,
            checkpoint_caches=checkpoint_caches,
            verified_removed=False,
            errors=tuple(errors),
        )
        write_json_atomic(receipt_path, receipt)
        return _cleanup_response(receipt_path, receipt)

    pending_receipt = _cleanup_receipt(
        repository_root=repository_root,
        workflow_id=workflow_id,
        operation_id=operation_id,
        operation_time=operation_time,
        mode=mode,
        records=records,
        manifest_status="execution-pending",
        receipt_path=receipt_path,
        planned_paths=planned_paths,
        removed_paths=(),
        conflicts=(),
        external_conflicts=(),
        retained_managed=retained_managed,
        retained_external=retained_external,
        repository_caches=repository_caches,
        package_caches=package_caches,
        materialized_environments=materialized_environments,
        checkpoint_caches=checkpoint_caches,
        verified_removed=False,
        errors=(),
    )
    write_json_atomic(receipt_path, pending_receipt)

    removed: list[str] = []
    execution_errors: list[str] = []
    for path in sorted(actions, key=lambda item: len(item.parts), reverse=True):
        existed = path.exists() or path.is_symlink()
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            elif existed:
                path.unlink()
        except OSError as exc:
            execution_errors.append(f"cleanup failed to remove path {path}: {exc}")
            break
        if path.exists() or path.is_symlink():
            execution_errors.append(f"cleanup failed to remove path: {path}")
            break
        if existed:
            removed.append(str(path))

    for retained_path, existed in retained_before.items():
        if existed and not (retained_path.exists() or retained_path.is_symlink()):
            execution_errors.append(f"declared retained path was deleted: {retained_path}")
    for (raw, supplied, resolved), (supplied_existed, resolved_existed) in external_before.items():
        if supplied_existed and not (supplied.exists() or supplied.is_symlink()):
            execution_errors.append(f"declared external output disappeared during cleanup: {raw}")
        if (
            resolved != supplied
            and resolved_existed
            and not (resolved.exists() or resolved.is_symlink())
        ):
            execution_errors.append(
                f"declared external output target disappeared during cleanup: {raw} -> {resolved}"
            )

    retained_managed = _cleanup_path_records(retained_reasons_by_path)
    retained_external = tuple(
        _cleanup_path_record(
            raw,
            supplied,
            "external-audit-output",
            (
                "external audit output retained"
                if supplied.exists() or supplied.is_symlink()
                else "external audit output declared by the run manifest is missing"
            ),
        )
        for raw, supplied, _ in sorted(external_paths, key=lambda item: item[0])
    )
    verified_removed = not execution_errors and all(
        not path.exists() and not path.is_symlink() for path in actions
    )
    receipt = _cleanup_receipt(
        repository_root=repository_root,
        workflow_id=workflow_id,
        operation_id=operation_id,
        operation_time=operation_time,
        mode=mode,
        records=records,
        manifest_status="complete" if verified_removed else "failed",
        receipt_path=receipt_path,
        planned_paths=planned_paths,
        removed_paths=tuple(removed),
        conflicts=(),
        external_conflicts=(),
        retained_managed=retained_managed,
        retained_external=retained_external,
        repository_caches=_records_for_category(retained_managed, "repository-cache"),
        package_caches=_records_for_category(retained_managed, "package-cache"),
        materialized_environments=_records_for_category(
            retained_managed,
            "materialized-environment",
        ),
        checkpoint_caches=_records_for_category(retained_managed, "checkpoint-cache"),
        verified_removed=verified_removed,
        errors=tuple(execution_errors),
    )
    write_json_atomic(receipt_path, receipt)
    return _cleanup_response(receipt_path, receipt)


def _cleanup_path_record(
    displayed_path: str,
    resolved_path: Path,
    category: str,
    reason: str,
) -> CleanupPathRecord:
    status = _cleanup_path_status(resolved_path)
    if status == "retained-symlink":
        digest = None
    elif status == "retained-existing-file":
        digest = sha256_file(resolved_path)
    else:
        digest = None
    return CleanupPathRecord(
        path=displayed_path,
        category=category,
        reason=reason,
        status=status,
        sha256=digest,
    )


def _cleanup_path_status(
    path: Path,
) -> Literal[
    "retained-existing-file",
    "retained-existing-directory",
    "retained-symlink",
    "missing",
]:
    if path.is_symlink():
        return "retained-symlink"
    if path.is_file():
        return "retained-existing-file"
    if path.is_dir():
        return "retained-existing-directory"
    return "missing"


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _cleanup_path_records(
    retained_reasons_by_path: dict[Path, tuple[str, str, str]],
) -> tuple[CleanupPathRecord, ...]:
    return tuple(
        _cleanup_path_record(displayed, path, category, reason)
        for path, (displayed, category, reason) in sorted(
            retained_reasons_by_path.items(),
            key=lambda item: item[1][0],
        )
    )


def _records_for_category(
    records: tuple[CleanupPathRecord, ...],
    category: str,
) -> tuple[CleanupPathRecord, ...]:
    return tuple(record for record in records if record.category == category)


def _cleanup_receipt(
    *,
    repository_root: Path,
    workflow_id: str,
    operation_id: str,
    operation_time: datetime,
    mode: Literal["dry-run", "execute"],
    records: Sequence[tuple[Path, ManagedRunManifest]],
    manifest_status: str,
    receipt_path: Path,
    planned_paths: tuple[str, ...],
    removed_paths: tuple[str, ...],
    conflicts: tuple[CleanupRetentionConflict, ...],
    external_conflicts: tuple[CleanupExternalProtectionConflict, ...],
    retained_managed: tuple[CleanupPathRecord, ...],
    retained_external: tuple[CleanupPathRecord, ...],
    repository_caches: tuple[CleanupPathRecord, ...],
    package_caches: tuple[CleanupPathRecord, ...],
    materialized_environments: tuple[CleanupPathRecord, ...],
    checkpoint_caches: tuple[CleanupPathRecord, ...],
    verified_removed: bool,
    errors: tuple[str, ...],
) -> CleanupReceipt:
    receipt_reference = _display_path(repository_root, receipt_path)
    cleanup_result = {
        "schema_version": "1.0.0",
        "cleanup_operation_id": operation_id,
        "time": operation_time.isoformat(),
        "mode": mode,
        "status": manifest_status,
        "receipt_path": receipt_reference,
        "planned_paths": list(planned_paths),
        "removed_paths": list(removed_paths),
        "retention_conflicts": [item.model_dump(mode="json") for item in conflicts],
        "external_protection_conflicts": [
            item.model_dump(mode="json") for item in external_conflicts
        ],
        "verified_removed": verified_removed,
        "errors": list(errors),
    }
    consumed: list[CleanupConsumedRunManifest] = []
    for manifest_path, manifest in records:
        finalized = manifest.model_copy(update={"cleanup_result": cleanup_result})
        serialized = canonical_json_text(finalized.model_dump(mode="json"))
        consumed.append(
            CleanupConsumedRunManifest(
                path=_display_path(repository_root, manifest_path),
                sha256=hashlib.sha256(serialized.encode()).hexdigest(),
                manifest=finalized,
            )
        )
    return CleanupReceipt(
        schema_version="1.0.0",
        workflow_id=workflow_id,
        cleanup_operation_id=operation_id,
        time=operation_time,
        mode=mode,
        run_manifests_consumed=tuple(consumed),
        planned_paths=planned_paths,
        removed_paths=removed_paths,
        retention_conflicts=conflicts,
        external_protection_conflicts=external_conflicts,
        retained_managed_paths=retained_managed,
        retained_external_paths=retained_external,
        repository_caches_retained=repository_caches,
        package_caches_retained=package_caches,
        materialized_environments_retained=materialized_environments,
        checkpoint_caches_retained=checkpoint_caches,
        verified_removed=verified_removed,
        errors=errors,
    )


def _display_path(repository_root: Path, path: Path) -> str:
    try:
        return path.relative_to(repository_root).as_posix()
    except ValueError:
        return str(path)


def _cleanup_response(receipt_path: Path, receipt: CleanupReceipt) -> dict[str, object]:
    payload = receipt.model_dump(mode="json")
    payload.update(
        {
            "dry_run": receipt.mode == "dry-run",
            "cleanup_succeeded": not receipt.errors
            and (receipt.mode == "dry-run" or receipt.verified_removed),
            "receipt_path": str(receipt_path),
            "receipt_sha256": sha256_file(receipt_path),
        }
    )
    return payload


def _cleanup_category(
    repository_root: Path,
    workflow_id: str,
    path: Path,
) -> str | None:
    roots = (
        ("ephemeral-workspace", repository_root / WORKSPACE_ROOT / workflow_id),
        ("trial-environment", repository_root / ".torch-dae/trials"),
        (
            "retained-diagnostic",
            repository_root / ".torch-dae/reports/onboarding" / workflow_id,
        ),
        ("repository-cache", repository_root / ".torch-dae/repositories"),
        ("package-cache", repository_root / ".torch-dae/source-builds"),
        ("materialized-environment", repository_root / ".torch-dae/environments"),
        ("checkpoint-cache", repository_root / ".torch-dae/checkpoints"),
    )
    resolved = path.resolve()
    for category, root in roots:
        root_resolved = root.resolve()
        if resolved == root_resolved or root_resolved in resolved.parents:
            return category
    return None


def _minimal_cleanup_roots(paths: Iterable[Path]) -> list[Path]:
    ordered = sorted(set(paths), key=lambda item: (len(item.parts), str(item)))
    result: list[Path] = []
    for path in ordered:
        if any(parent == path or parent in path.parents for parent in result):
            continue
        result.append(path)
    return result


def _git(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise HandoffManagementError(
            f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout


def _git_z(repository_root: Path, *arguments: str) -> list[str]:
    return [item for item in _git(repository_root, *arguments).split("\0") if item]


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: object) -> None:
    _write_text(path, canonical_json_text(value))
