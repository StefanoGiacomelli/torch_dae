from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

import torch_dae.onboarding.handoff as handoff_module
from torch_dae.onboarding.contracts import (
    CleanupReceipt,
    HandoffArtifactReference,
    HandoffStatus,
    HandoffValidationSummary,
    OnboardingPhase,
    PhaseHandoffManifest,
    RecommendedNextMode,
    WorkflowRecord,
)
from torch_dae.onboarding.handoff import (
    HandoffManagementError,
    bundle_workflow,
    canonical_json_text,
    cleanup_workflow,
    discover_handoff,
    promote_phase,
    sha256_file,
    skill_fingerprint,
    validate_workflow,
)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository(tmp_path: Path, repo_root: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    (root / "project_spec.md").write_text("synthetic handoff specification\n")
    (root / ".gitignore").write_text(".torch-dae/\n")
    (root / "README.md").write_text("synthetic handoff repository\n")
    shutil.copytree(
        repo_root / "skills/audio-model-onboarding",
        root / "skills/audio-model-onboarding",
    )
    _git(root, "init")
    _git(root, "config", "user.email", "tests@example.invalid")
    _git(root, "config", "user.name", "Handoff Tests")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "synthetic baseline")
    return root


def _write_cleanup_manifest(
    root: Path,
    run: Path,
    *,
    workflow_id: str,
    created_paths: list[Path],
    external_paths: list[str],
) -> Path:
    manifest = {
        "schema_version": "1.0.0",
        "run_id": "run-one",
        "workflow_id": workflow_id,
        "phase": "analyze",
        "started_at": "2026-07-30T00:00:00Z",
        "repository_commit": _git(root, "rev-parse", "HEAD"),
        "created_paths": [path.relative_to(root).as_posix() for path in created_paths],
        "reused_paths": [],
        "external_paths": external_paths,
        "retained_paths": [],
        "retained_reasons": {},
        "cleanup_result": None,
    }
    manifest_path = run / "run-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


def _source(
    root: Path,
    workflow_id: str,
    phase: OnboardingPhase,
    *,
    accepted_phases: tuple[OnboardingPhase, ...] | None = None,
    content: str = '{"result": "accepted"}\n',
    artifact_sha256: str | None = None,
    superseded_handoff_sha256: str | None = None,
) -> Path:
    phases = accepted_phases or (phase,)
    run_root = root / ".torch-dae/workspaces" / workflow_id / phase.value / "run-one"
    if run_root.exists():
        shutil.rmtree(run_root)
    source = run_root / "promotion"
    phase_root = source / phase.value
    phase_root.mkdir(parents=True)
    output = phase_root / "report.json"
    output.write_text(content)
    head = _git(root, "rev-parse", "HEAD")
    workflow = WorkflowRecord.model_validate(
        {
            "schema_version": "1.0.0",
            "workflow_id": workflow_id,
            "model_family": "Synthetic",
            "target_variant_ids": ["variant-one"],
            "target_checkpoint_ids": ["checkpoint-one"],
            "target_card_ids": ["card-one"],
            "created_repository_commit": head,
            "current_accepted_phase": phases[-1].value,
            "accepted_phase_paths": [
                {
                    "phase": item.value,
                    "handoff_path": (f"onboarding_reports/{workflow_id}/{item.value}/handoff.json"),
                }
                for item in phases
            ],
            "status": "active",
        }
    )
    handoff = PhaseHandoffManifest(
        schema_version="1.0.0",
        workflow_id=workflow_id,
        phase=phase,
        handoff_status=HandoffStatus.ACCEPTED,
        repository_commit=head,
        project_spec_sha256=sha256_file(root / "project_spec.md"),
        canonical_skill_fingerprint=skill_fingerprint(root),
        input_artifacts=(),
        output_artifacts=(
            HandoffArtifactReference(
                path=f"onboarding_reports/{workflow_id}/{phase.value}/report.json",
                sha256=artifact_sha256 or sha256_file(output),
                media_type="application/json",
                originating_phase=phase.value,
                canonical_role="phase-report",
            ),
        ),
        target_variant_ids=("variant-one",),
        target_checkpoint_ids=("checkpoint-one",),
        target_card_ids=("card-one",),
        validation_summary=HandoffValidationSummary(
            passed=True,
            checks=("synthetic validation",),
        ),
        allowed_next_modes=(RecommendedNextMode.RESOLVE_ENVIRONMENT,),
        superseded_handoff_sha256=superseded_handoff_sha256,
    )
    (source / "workflow.json").write_text(canonical_json_text(workflow.model_dump(mode="json")))
    (phase_root / "handoff.json").write_text(canonical_json_text(handoff.model_dump(mode="json")))
    return source


def test_discover_validates_hashes_missing_phases_and_duplicate_attachments(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    root = _repository(tmp_path, repo_root)
    source = _source(root, "workflow-one", OnboardingPhase.ANALYZE)
    promote_phase(
        root,
        source_dir=source,
        workflow_id="workflow-one",
        phase=OnboardingPhase.ANALYZE,
    )
    result = discover_handoff(
        root,
        workflow_id="workflow-one",
        required_phase=OnboardingPhase.ANALYZE,
    )
    assert result["valid"] is True
    assert result["output_artifact_paths"] == [
        "onboarding_reports/workflow-one/analyze/report.json"
    ]

    duplicate = tmp_path / "report.json"
    duplicate.write_text('{"result": "different"}\n')
    with pytest.raises(HandoffManagementError, match="attachment hash mismatch"):
        discover_handoff(
            root,
            workflow_id="workflow-one",
            required_phase=OnboardingPhase.ANALYZE,
            attachments=(duplicate,),
        )
    with pytest.raises(HandoffManagementError, match="required phase is missing"):
        discover_handoff(
            root,
            workflow_id="workflow-one",
            required_phase=OnboardingPhase.RESOLVE_ENVIRONMENT,
        )

    report = root / "onboarding_reports/workflow-one/analyze/report.json"
    report.write_text('{"result": "tampered"}\n')
    with pytest.raises(HandoffManagementError, match="artifact hash mismatch"):
        validate_workflow(root, "workflow-one")


def test_discovery_rejects_ambiguous_active_workflows(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    root = _repository(tmp_path, repo_root)
    for workflow_id in ("workflow-one", "workflow-two"):
        promote_phase(
            root,
            source_dir=_source(root, workflow_id, OnboardingPhase.ANALYZE),
            workflow_id=workflow_id,
            phase=OnboardingPhase.ANALYZE,
        )
    with pytest.raises(HandoffManagementError, match="ambiguous active workflows"):
        discover_handoff(
            root,
            workflow_id=None,
            required_phase=OnboardingPhase.ANALYZE,
        )


def test_promotion_is_atomic_and_supersession_is_explicit(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    root = _repository(tmp_path, repo_root)
    first_source = _source(root, "workflow-one", OnboardingPhase.ANALYZE)
    promote_phase(
        root,
        source_dir=first_source,
        workflow_id="workflow-one",
        phase=OnboardingPhase.ANALYZE,
    )
    canonical_handoff = root / "onboarding_reports/workflow-one/analyze/handoff.json"
    previous_hash = sha256_file(canonical_handoff)
    unchanged = canonical_handoff.read_bytes()

    invalid = _source(
        root,
        "workflow-one",
        OnboardingPhase.RESOLVE_ENVIRONMENT,
        accepted_phases=(
            OnboardingPhase.ANALYZE,
            OnboardingPhase.RESOLVE_ENVIRONMENT,
        ),
        artifact_sha256="0" * 64,
    )
    with pytest.raises(HandoffManagementError, match="artifact hash mismatch"):
        promote_phase(
            root,
            source_dir=invalid,
            workflow_id="workflow-one",
            phase=OnboardingPhase.RESOLVE_ENVIRONMENT,
        )
    assert canonical_handoff.read_bytes() == unchanged
    assert not (root / "onboarding_reports/workflow-one/resolve-environment").exists()

    destination = root / "onboarding_reports/workflow-one"
    destination_snapshot = {
        path.relative_to(destination).as_posix(): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file()
    }
    unreferenced = _source(
        root,
        "workflow-one",
        OnboardingPhase.ANALYZE,
        content='{"result": "unreferenced rejection"}\n',
        superseded_handoff_sha256=previous_hash,
    )
    (unreferenced / "analyze/unreferenced.json").write_text('{"not": "declared"}\n')
    with pytest.raises(HandoffManagementError, match="unreferenced promotion source artifact"):
        promote_phase(
            root,
            source_dir=unreferenced,
            workflow_id="workflow-one",
            phase=OnboardingPhase.ANALYZE,
            supersede=True,
        )
    assert {
        path.relative_to(destination).as_posix(): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file()
    } == destination_snapshot
    symlink_source = _source(
        root,
        "workflow-one",
        OnboardingPhase.ANALYZE,
        content='{"result": "symlink rejection"}\n',
        superseded_handoff_sha256=previous_hash,
    )
    (symlink_source / "analyze/reference.json").symlink_to("report.json")
    with pytest.raises(HandoffManagementError, match="must not contain symlinks"):
        promote_phase(
            root,
            source_dir=symlink_source,
            workflow_id="workflow-one",
            phase=OnboardingPhase.ANALYZE,
            supersede=True,
        )
    assert {
        path.relative_to(destination).as_posix(): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file()
    } == destination_snapshot

    replacement = _source(
        root,
        "workflow-one",
        OnboardingPhase.ANALYZE,
        content='{"result": "superseding"}\n',
        superseded_handoff_sha256=previous_hash,
    )
    with pytest.raises(HandoffManagementError, match="explicit supersession"):
        promote_phase(
            root,
            source_dir=replacement,
            workflow_id="workflow-one",
            phase=OnboardingPhase.ANALYZE,
        )
    promoted = promote_phase(
        root,
        source_dir=replacement,
        workflow_id="workflow-one",
        phase=OnboardingPhase.ANALYZE,
        supersede=True,
    )
    assert promoted["superseded_handoff_sha256"] == previous_hash
    assert (
        root
        / "onboarding_reports/workflow-one/analyze"
        / f"handoff.{previous_hash}.superseded.json"
    ).is_file()
    validate_workflow(root, "workflow-one")


def test_bundle_is_deterministic_normalized_and_inventory_exact(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    root = _repository(tmp_path, repo_root)
    promote_phase(
        root,
        source_dir=_source(root, "workflow-one", OnboardingPhase.ANALYZE),
        workflow_id="workflow-one",
        phase=OnboardingPhase.ANALYZE,
    )
    first = bundle_workflow(
        root,
        workflow_id="workflow-one",
        through_phase=OnboardingPhase.ANALYZE,
        output_dir=tmp_path / "review-one",
        include_working_tree=True,
    )
    second = bundle_workflow(
        root,
        workflow_id="workflow-one",
        through_phase=OnboardingPhase.ANALYZE,
        output_dir=tmp_path / "review-two",
        include_working_tree=True,
    )
    assert first["sha256"] == second["sha256"]
    assert first["declared_actual_inventory_equal"] is True
    assert first["declared_actual_inventory_match"] is True
    assert first["archive_clean"] is True
    assert first["archive_metadata_normalized"] is True
    assert first["gzip_metadata_normalized"] is True
    assert first["archive_sha256"] == first["sha256"]
    checksum_sidecar = Path(str(first["checksum_sidecar_path"]))
    assert checksum_sidecar.read_text() == (
        f"{first['sha256']}  {Path(str(first['archive_path'])).name}\n"
    )
    with tarfile.open(str(first["archive_path"]), "r:gz") as archive:
        members = archive.getmembers()
        declared_file = archive.extractfile("metadata/declared-archive-inventory.json")
        actual_file = archive.extractfile("metadata/actual-archive-inventory.json")
        artifact_manifest_file = archive.extractfile("metadata/artifact-manifest.json")
        bundle_result_file = archive.extractfile("metadata/bundle-result.json")
        assert declared_file is not None
        assert actual_file is not None
        assert artifact_manifest_file is not None
        assert bundle_result_file is not None
        declared = json.loads(declared_file.read())
        actual = json.loads(actual_file.read())
        artifact_manifest = json.loads(artifact_manifest_file.read())
        bundle_result = json.loads(bundle_result_file.read())
    assert declared == actual == sorted(member.name for member in members)
    manifested_paths = {item["path"] for item in artifact_manifest}
    assert set(
        bundle_result["artifact_manifest_scope"]["self_excluded_metadata_records"]
    ).isdisjoint(manifested_paths)
    assert all(member.uid == member.gid == member.mtime == 0 for member in members)
    assert all(
        not member.uname and not member.gname and not member.pax_headers for member in members
    )
    forbidden = {".git", ".torch-dae", ".venv", ".DS_Store", "__MACOSX"}
    assert all(forbidden.isdisjoint(Path(member.name).parts) for member in members)


def test_cleanup_uses_recorded_paths_and_retains_caches(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    root = _repository(tmp_path, repo_root)
    workflow_id = "workflow-one"
    run = root / f".torch-dae/workspaces/{workflow_id}/analyze/run-one"
    run.mkdir(parents=True)
    trial = root / ".torch-dae/trials/legacy-trial-one"
    trial.mkdir(parents=True)
    cache = root / ".torch-dae/source-builds/example/cache"
    cache.mkdir(parents=True)
    manifest = {
        "schema_version": "1.0.0",
        "run_id": "run-one",
        "workflow_id": workflow_id,
        "phase": "analyze",
        "started_at": "2026-07-30T00:00:00Z",
        "repository_commit": _git(root, "rev-parse", "HEAD"),
        "created_paths": [
            run.relative_to(root).as_posix(),
            trial.relative_to(root).as_posix(),
            cache.relative_to(root).as_posix(),
        ],
        "reused_paths": [],
        "external_paths": [],
        "retained_paths": [],
        "retained_reasons": {},
        "cleanup_result": None,
    }
    (run / "run-manifest.json").write_text(json.dumps(manifest))
    dry_run = cleanup_workflow(root, workflow_id=workflow_id, dry_run=True)
    assert dry_run["planned_paths"] == [str(trial), str(run)]
    assert dry_run["retained_managed_paths"] == [
        {
            "path": str(cache),
            "category": "package-cache",
            "reason": "destructive cache cleanup was not explicitly requested",
            "status": "retained-existing-directory",
            "sha256": None,
        }
    ]
    assert dry_run["removed_paths"] == []
    assert dry_run["verified_removed"] is False
    assert run.exists()
    assert trial.exists()
    dry_receipt = CleanupReceipt.model_validate_json(Path(str(dry_run["receipt_path"])).read_text())
    assert dry_receipt.mode == "dry-run"
    assert dry_run["receipt_sha256"] == sha256_file(Path(str(dry_run["receipt_path"])))
    executed = cleanup_workflow(root, workflow_id=workflow_id, dry_run=False)
    assert executed["verified_removed"] is True
    assert not run.exists()
    assert not trial.exists()
    assert cache.exists()
    receipt_path = Path(str(executed["receipt_path"]))
    receipt = CleanupReceipt.model_validate_json(receipt_path.read_text())
    assert receipt.run_manifests_consumed[0].manifest.cleanup_result is not None
    assert receipt.run_manifests_consumed[0].manifest.cleanup_result["status"] == "complete"
    assert not list(receipt_path.parent.glob(".*.tmp"))


def test_cleanup_retention_conflict_blocks_all_deletion(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    root = _repository(tmp_path, repo_root)
    workflow_id = "workflow-one"
    run = root / f".torch-dae/workspaces/{workflow_id}/analyze/run-one"
    run.mkdir(parents=True)
    retained = run / "diagnostics/failure.log"
    retained.parent.mkdir()
    retained.write_text("bounded failure evidence\n")
    trial = root / ".torch-dae/trials/legacy-trial-one"
    trial.mkdir(parents=True)
    manifest = {
        "schema_version": "1.0.0",
        "run_id": "run-one",
        "workflow_id": workflow_id,
        "phase": "analyze",
        "started_at": "2026-07-30T00:00:00Z",
        "repository_commit": _git(root, "rev-parse", "HEAD"),
        "created_paths": [
            run.relative_to(root).as_posix(),
            trial.relative_to(root).as_posix(),
        ],
        "reused_paths": [],
        "external_paths": [],
        "retained_paths": [retained.relative_to(root).as_posix()],
        "retained_reasons": {
            retained.relative_to(root).as_posix(): "preserve the failure classification"
        },
        "cleanup_result": None,
    }
    (run / "run-manifest.json").write_text(json.dumps(manifest))

    dry_run = cleanup_workflow(root, workflow_id=workflow_id, dry_run=True)
    assert dry_run["retention_conflicts"] == [
        {
            "retained_path": str(retained),
            "deletion_root": str(run),
            "reason": "preserve the failure classification",
            "remediation": (
                f"move the retained diagnostic under "
                f"{root / '.torch-dae/reports/onboarding' / workflow_id} and update "
                "retained_paths in the run manifest before cleanup"
            ),
        }
    ]
    assert dry_run["removed_paths"] == []
    assert run.exists()
    assert trial.exists()

    executed = cleanup_workflow(root, workflow_id=workflow_id, dry_run=False)
    assert executed["cleanup_succeeded"] is False
    assert executed["verified_removed"] is False
    assert executed["removed_paths"] == []
    assert any("move the retained diagnostic" in error for error in executed["errors"])
    assert run.exists()
    assert trial.exists()
    assert retained.exists()


def test_cleanup_external_file_below_run_root_blocks_before_mutation(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    root = _repository(tmp_path, repo_root)
    workflow_id = "workflow-one"
    run = root / f".torch-dae/workspaces/{workflow_id}/analyze/run-one"
    run.mkdir(parents=True)
    audit = run / "audit.json"
    audit.write_bytes(b'{"audit": "must survive"}\n')
    trial = root / ".torch-dae/trials/trial-one"
    trial.mkdir(parents=True)
    marker = trial / "marker.txt"
    marker.write_bytes(b"no partial cleanup\n")
    manifest_path = _write_cleanup_manifest(
        root,
        run,
        workflow_id=workflow_id,
        created_paths=[run, trial],
        external_paths=[audit.relative_to(root).as_posix()],
    )
    manifest_before = manifest_path.read_bytes()
    audit_before = audit.read_bytes()
    marker_before = marker.read_bytes()

    dry_run = cleanup_workflow(root, workflow_id=workflow_id, dry_run=True)
    assert dry_run["cleanup_succeeded"] is False
    assert dry_run["removed_paths"] == []
    assert dry_run["verified_removed"] is False
    assert dry_run["external_protection_conflicts"] == [
        {
            "supplied_external_path": audit.relative_to(root).as_posix(),
            "resolved_path": str(audit),
            "deletion_root": str(run),
            "status": "retained-existing-file",
            "reason": (
                "existing external audit output overlaps the planned deletion root through "
                "its supplied path and resolved target"
            ),
            "remediation": (
                "move the external audit output outside the managed deletion roots, update "
                "external_paths in the run manifest, and rerun cleanup"
            ),
        }
    ]
    dry_receipt = CleanupReceipt.model_validate_json(Path(str(dry_run["receipt_path"])).read_text())
    assert dry_receipt.external_protection_conflicts
    assert manifest_path.read_bytes() == manifest_before
    assert audit.read_bytes() == audit_before
    assert marker.read_bytes() == marker_before

    executed = cleanup_workflow(root, workflow_id=workflow_id, dry_run=False)
    assert executed["cleanup_succeeded"] is False
    assert executed["removed_paths"] == []
    assert executed["verified_removed"] is False
    assert executed["external_protection_conflicts"]
    execution_receipt = CleanupReceipt.model_validate_json(
        Path(str(executed["receipt_path"])).read_text()
    )
    assert execution_receipt.external_protection_conflicts
    assert manifest_path.read_bytes() == manifest_before
    assert audit.read_bytes() == audit_before
    assert marker.read_bytes() == marker_before


def test_cleanup_external_directory_containing_deletion_root_blocks(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    root = _repository(tmp_path, repo_root)
    workflow_id = "workflow-one"
    phase_root = root / f".torch-dae/workspaces/{workflow_id}/analyze"
    run = phase_root / "run-one"
    run.mkdir(parents=True)
    manifest_path = _write_cleanup_manifest(
        root,
        run,
        workflow_id=workflow_id,
        created_paths=[run],
        external_paths=[phase_root.relative_to(root).as_posix()],
    )
    manifest_before = manifest_path.read_bytes()

    result = cleanup_workflow(root, workflow_id=workflow_id, dry_run=False)

    assert result["cleanup_succeeded"] is False
    assert result["removed_paths"] == []
    assert result["verified_removed"] is False
    assert result["external_protection_conflicts"][0]["status"] == ("retained-existing-directory")
    assert result["external_protection_conflicts"][0]["deletion_root"] == str(run)
    assert manifest_path.read_bytes() == manifest_before


def test_cleanup_external_symlink_resolving_into_deletion_root_blocks(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    root = _repository(tmp_path, repo_root)
    workflow_id = "workflow-one"
    run = root / f".torch-dae/workspaces/{workflow_id}/analyze/run-one"
    run.mkdir(parents=True)
    target = run / "audit.json"
    target.write_bytes(b'{"audit": "target"}\n')
    external_link = tmp_path / "external-review/audit-link.json"
    external_link.parent.mkdir()
    external_link.symlink_to(target)
    _write_cleanup_manifest(
        root,
        run,
        workflow_id=workflow_id,
        created_paths=[run],
        external_paths=[str(external_link)],
    )
    target_before = target.read_bytes()

    result = cleanup_workflow(root, workflow_id=workflow_id, dry_run=False)

    assert result["cleanup_succeeded"] is False
    assert result["removed_paths"] == []
    assert result["verified_removed"] is False
    assert result["external_protection_conflicts"][0]["supplied_external_path"] == str(
        external_link
    )
    assert result["external_protection_conflicts"][0]["resolved_path"] == str(target)
    assert result["external_protection_conflicts"][0]["status"] == "retained-symlink"
    assert external_link.is_symlink()
    assert external_link.resolve() == target
    assert target.read_bytes() == target_before


def test_cleanup_external_symlink_inside_deletion_root_blocks(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    root = _repository(tmp_path, repo_root)
    workflow_id = "workflow-one"
    run = root / f".torch-dae/workspaces/{workflow_id}/analyze/run-one"
    run.mkdir(parents=True)
    target = tmp_path / "external-review/audit.json"
    target.parent.mkdir()
    target.write_bytes(b'{"audit": "outside"}\n')
    internal_link = run / "audit-link.json"
    internal_link.symlink_to(target)
    _write_cleanup_manifest(
        root,
        run,
        workflow_id=workflow_id,
        created_paths=[run],
        external_paths=[internal_link.relative_to(root).as_posix()],
    )
    target_before = target.read_bytes()

    result = cleanup_workflow(root, workflow_id=workflow_id, dry_run=False)

    assert result["cleanup_succeeded"] is False
    assert result["removed_paths"] == []
    assert result["verified_removed"] is False
    assert result["external_protection_conflicts"][0]["resolved_path"] == str(target)
    assert result["external_protection_conflicts"][0]["status"] == "retained-symlink"
    assert internal_link.is_symlink()
    assert target.read_bytes() == target_before


def test_cleanup_retains_external_symlink_resolving_outside_deletion_roots(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    root = _repository(tmp_path, repo_root)
    workflow_id = "workflow-one"
    run = root / f".torch-dae/workspaces/{workflow_id}/analyze/run-one"
    run.mkdir(parents=True)
    target = tmp_path / "external-review/audit.json"
    target.parent.mkdir()
    target.write_bytes(b'{"audit": "outside"}\n')
    external_link = tmp_path / "external-review/audit-link.json"
    external_link.symlink_to(target)
    _write_cleanup_manifest(
        root,
        run,
        workflow_id=workflow_id,
        created_paths=[run],
        external_paths=[str(external_link)],
    )

    result = cleanup_workflow(root, workflow_id=workflow_id, dry_run=False)

    assert result["cleanup_succeeded"] is True
    assert result["verified_removed"] is True
    assert result["external_protection_conflicts"] == []
    assert not run.exists()
    assert external_link.is_symlink()
    assert external_link.resolve() == target
    assert target.read_bytes() == b'{"audit": "outside"}\n'
    assert result["retained_external_paths"] == [
        {
            "path": str(external_link),
            "category": "external-audit-output",
            "reason": "external audit output retained",
            "status": "retained-symlink",
            "sha256": None,
        }
    ]


def test_cleanup_retains_moved_diagnostics_and_external_outputs(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    root = _repository(tmp_path, repo_root)
    workflow_id = "workflow-one"
    run = root / f".torch-dae/workspaces/{workflow_id}/analyze/run-one"
    run.mkdir(parents=True)
    diagnostics = root / f".torch-dae/reports/onboarding/{workflow_id}/diagnostics/failure.json"
    diagnostics.parent.mkdir(parents=True)
    diagnostics.write_text('{"failure": "classified"}\n')
    external = tmp_path / "external-review/audit.json"
    external.parent.mkdir()
    external.write_text('{"audit": true}\n')
    missing_external = tmp_path / "external-review/missing.json"
    manifest = {
        "schema_version": "1.0.0",
        "run_id": "run-one",
        "workflow_id": workflow_id,
        "phase": "analyze",
        "started_at": "2026-07-30T00:00:00Z",
        "repository_commit": _git(root, "rev-parse", "HEAD"),
        "created_paths": [run.relative_to(root).as_posix()],
        "reused_paths": [],
        "external_paths": [str(external), str(missing_external)],
        "retained_paths": [diagnostics.relative_to(root).as_posix()],
        "retained_reasons": {
            diagnostics.relative_to(root).as_posix(): "bounded retained diagnostic"
        },
        "cleanup_result": None,
    }
    (run / "run-manifest.json").write_text(json.dumps(manifest))

    result = cleanup_workflow(
        root,
        workflow_id=workflow_id,
        dry_run=False,
        include_repository_caches=True,
        include_package_caches=True,
        include_environments=True,
        include_checkpoints=True,
    )
    assert result["verified_removed"] is True
    assert diagnostics.exists()
    assert external.exists()
    assert result["retained_external_paths"] == [
        {
            "path": str(external),
            "category": "external-audit-output",
            "reason": "external audit output retained",
            "status": "retained-existing-file",
            "sha256": sha256_file(external),
        },
        {
            "path": str(missing_external),
            "category": "external-audit-output",
            "reason": "external audit output declared by the run manifest is missing",
            "status": "missing",
            "sha256": None,
        },
    ]
    retained_diagnostics = [
        item
        for item in result["retained_managed_paths"]
        if item["category"] == "retained-diagnostic"
    ]
    assert retained_diagnostics[0]["sha256"] == sha256_file(diagnostics)
    receipt = CleanupReceipt.model_validate_json(Path(str(result["receipt_path"])).read_text())
    assert receipt.retained_external_paths[0].path == str(external)
    assert receipt.retained_external_paths[1].status == "missing"
    assert receipt.run_manifests_consumed[0].manifest.cleanup_result is not None
    assert receipt.run_manifests_consumed[0].manifest.cleanup_result["verified_removed"] is True


def test_cleanup_cannot_verify_success_after_external_symlink_target_disappears(
    tmp_path: Path,
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repository(tmp_path, repo_root)
    workflow_id = "workflow-one"
    run = root / f".torch-dae/workspaces/{workflow_id}/analyze/run-one"
    run.mkdir(parents=True)
    target = tmp_path / "external-review/audit.json"
    target.parent.mkdir()
    target.write_bytes(b'{"audit": "outside"}\n')
    external_link = tmp_path / "external-review/audit-link.json"
    external_link.symlink_to(target)
    _write_cleanup_manifest(
        root,
        run,
        workflow_id=workflow_id,
        created_paths=[run],
        external_paths=[str(external_link)],
    )
    original_rmtree = shutil.rmtree

    def remove_with_external_target_loss(
        path: str | Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        original_rmtree(path, *args, **kwargs)
        target.unlink(missing_ok=True)

    monkeypatch.setattr(handoff_module.shutil, "rmtree", remove_with_external_target_loss)
    result = cleanup_workflow(root, workflow_id=workflow_id, dry_run=False)

    assert result["verified_removed"] is False
    assert result["cleanup_succeeded"] is False
    assert external_link.is_symlink()
    assert any(
        "declared external output target disappeared during cleanup" in error
        for error in result["errors"]
    )


def test_cleanup_cannot_verify_success_after_retained_content_disappears(
    tmp_path: Path,
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repository(tmp_path, repo_root)
    workflow_id = "workflow-one"
    run = root / f".torch-dae/workspaces/{workflow_id}/analyze/run-one"
    run.mkdir(parents=True)
    retained = root / f".torch-dae/reports/onboarding/{workflow_id}/diagnostics/failure.log"
    retained.parent.mkdir(parents=True)
    retained.write_text("must survive\n")
    manifest = {
        "schema_version": "1.0.0",
        "run_id": "run-one",
        "workflow_id": workflow_id,
        "phase": "analyze",
        "started_at": "2026-07-30T00:00:00Z",
        "repository_commit": _git(root, "rev-parse", "HEAD"),
        "created_paths": [run.relative_to(root).as_posix()],
        "reused_paths": [],
        "external_paths": [],
        "retained_paths": [retained.relative_to(root).as_posix()],
        "retained_reasons": {retained.relative_to(root).as_posix(): "must survive cleanup"},
        "cleanup_result": None,
    }
    (run / "run-manifest.json").write_text(json.dumps(manifest))
    original_rmtree = shutil.rmtree

    def remove_with_retention_loss(path: str | Path, *args: object, **kwargs: object) -> None:
        original_rmtree(path, *args, **kwargs)
        retained.unlink(missing_ok=True)

    monkeypatch.setattr(handoff_module.shutil, "rmtree", remove_with_retention_loss)
    result = cleanup_workflow(root, workflow_id=workflow_id, dry_run=False)
    assert result["verified_removed"] is False
    assert result["cleanup_succeeded"] is False
    assert any("declared retained path was deleted" in error for error in result["errors"])


def test_cleanup_refuses_an_unrecorded_run_root(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    root = _repository(tmp_path, repo_root)
    run = root / ".torch-dae/workspaces/workflow-one/analyze/run-one"
    run.mkdir(parents=True)
    manifest = {
        "schema_version": "1.0.0",
        "run_id": "run-one",
        "workflow_id": "workflow-one",
        "phase": "analyze",
        "started_at": "2026-07-30T00:00:00Z",
        "repository_commit": _git(root, "rev-parse", "HEAD"),
        "created_paths": [],
        "reused_paths": [],
        "external_paths": [],
        "retained_paths": [],
        "retained_reasons": {},
        "cleanup_result": None,
    }
    (run / "run-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(HandoffManagementError, match="run root is not recorded"):
        cleanup_workflow(root, workflow_id="workflow-one", dry_run=True)


def test_panns_migration_is_valid_and_preserves_accepted_artifacts(repo_root: Path) -> None:
    result = validate_workflow(repo_root, "panns-audioset-three-tuple")
    assert result["validated_phases"] == ["analyze", "resolve-environment"]
    root = repo_root / "onboarding_reports/panns-audioset-three-tuple"
    expected_hashes = {
        "workflow.json": "ad5fea06d032d2a20ca422b807333674f1419bd65cc4c901505a3b8c8d1ba39a",
        "analyze/handoff.json": (
            "dee4f8fcfbc20a47d5c86ebce71e2c25a745b6f19faf5a1b4efc510cf29cece2"
        ),
        "analyze/manual-supplements.json": (
            "9d96ebe1b54674d72bf3b0b997fa4661cfcc960417060c12cd2a4a1ef5dfd7ef"
        ),
        "analyze/panns-technical-analysis.json": (
            "afc32f4652686b0e6dc39d771d325691e614b4568096a67c9b9f627cdec554d3"
        ),
        "analyze/panns-technical-analysis.md": (
            "1777ee28788d2834b70db9f9b0b3468ff2cd9277134b6b82805f119de3d930d5"
        ),
        "resolve-environment/constructor-trial-result.json": (
            "5c0499d473d88f446b826052651956af5c127ed7686f41c9cebed67fd9ede266"
        ),
        "resolve-environment/environment-resolution-report.json": (
            "a6d57bfb7ea855c94c226584ca69fa05fa5f72cfa453083910770e2a8f1218c4"
        ),
        "resolve-environment/handoff.json": (
            "54ff5a6dcd6b034319624499f4e1bdbd1a49b9ff895bd5fcad52a59f5e73a92c"
        ),
        "resolve-environment/source-revision-comparison.json": (
            "04f419093df438cff3d1bd1e8629d1b099cb48495d03c6f6b035943146fd2938"
        ),
    }
    assert {path: sha256_file(root / path) for path in expected_hashes} == expected_hashes
    report = json.loads(
        (root / "resolve-environment/environment-resolution-report.json").read_text()
    )
    assert report["environment_materialization_succeeded"] is False
    assert report["environment_verification_succeeded"] is False
    assert report["environment_fingerprint"] is None
    assert report["next_lifecycle_status"] is None


def test_workflow_and_handoff_schemas_are_strict_and_accept_the_migration(
    repo_root: Path,
) -> None:
    workflow = json.loads(
        (repo_root / "onboarding_reports/panns-audioset-three-tuple/workflow.json").read_text()
    )
    handoff = json.loads(
        (
            repo_root
            / "onboarding_reports/panns-audioset-three-tuple/resolve-environment/handoff.json"
        ).read_text()
    )
    workflow_validator = Draft202012Validator(
        json.loads((repo_root / "schemas/workflow-record.schema.json").read_text())
    )
    handoff_validator = Draft202012Validator(
        json.loads((repo_root / "schemas/phase-handoff.schema.json").read_text())
    )
    workflow_validator.validate(workflow)
    handoff_validator.validate(handoff)

    invalid_workflow = {**workflow, "unexpected": True}
    invalid_handoff = {**handoff, "unexpected": True}
    with pytest.raises(JsonSchemaValidationError):
        workflow_validator.validate(invalid_workflow)
    with pytest.raises(JsonSchemaValidationError):
        handoff_validator.validate(invalid_handoff)


def test_hardening_guidance_preserves_environment_and_verification_boundaries(
    repo_root: Path,
) -> None:
    environment = (
        repo_root / "skills/audio-model-onboarding/references/environment-resolution.md"
    ).read_text()
    workflow = (
        repo_root / "skills/audio-model-onboarding/references/workflow-overview.md"
    ).read_text()
    specification = (repo_root / "project_spec.md").read_text()
    assert "direct dependency" in environment.lower()
    assert "constructor trial" in environment.lower()
    assert "draft resolution complete" in environment.lower()
    assert "WORKFLOW_ID" in workflow
    assert "verification_reports/" in specification
    assert "checkpoint-specific runtime evidence only" in specification
