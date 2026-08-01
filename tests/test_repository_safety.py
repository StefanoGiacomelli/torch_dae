from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

from scripts.validate_repository import validate_integration_artifacts
from torch_dae.onboarding.handoff import skill_fingerprint


def test_runtime_state_is_ignored(repo_root: Path) -> None:
    result = subprocess.run(
        ["git", "check-ignore", "-q", ".torch-dae/checkpoints/card/hash/file.pt"],
        cwd=repo_root,
        check=False,
    )
    assert result.returncode == 0


def test_no_runtime_files_staged(repo_root: Path) -> None:
    result = subprocess.run(
        ["git", "status", "--short", ".torch-dae"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout == ""


def test_no_legacy_backbone_json(repo_root: Path) -> None:
    assert not list(repo_root.glob("**/*backbone*.json"))


def test_repository_validation_is_checkout_name_independent(
    repo_root: Path,
) -> None:
    source = (repo_root / "scripts/validate_repository.py").read_text()
    assert 'ROOT.name != "torch-dae"' not in source
    assert "repository root basename is not torch-dae" not in source


def test_root_environment_has_no_torch() -> None:
    assert importlib.util.find_spec("torch") is None


def prepare_integration_root(repo_root: Path, root: Path) -> None:
    shutil.copytree(repo_root / "schemas", root / "schemas")
    (root / "model_cards").mkdir()
    (root / "environments").mkdir()
    (root / "verification_reports").mkdir()
    wrapper = root / "src/torch_dae/synthetic.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("class AudioModel:\n    pass\n")


def add_valid_synthetic_integration(repo_root: Path, root: Path) -> None:
    card = json.loads((repo_root / "tests/fixtures/valid/model-card.analyzed.json").read_text())
    card["card_status"] = "environment_resolved"
    card_id = card["card_id"]

    environment_dir = root / "environments" / card_id
    environment_dir.mkdir()
    shutil.copy(
        repo_root / "tests/fixtures/valid/environment.synthetic.json",
        environment_dir / "environment.json",
    )
    shutil.copy(
        repo_root / "tests/fixtures/valid/environment-sources.synthetic.json",
        environment_dir / "sources.json",
    )
    (environment_dir / "pyproject.toml").write_text(
        "[project]\nname = 'synthetic-model-environment'\nversion = '0.1.0'\n"
    )
    (environment_dir / "uv.lock").write_text("version = 1\n")
    (environment_dir / "verify_environment.py").write_text("raise SystemExit(0)\n")
    evidence_dir = root / "onboarding_reports/synthetic-workflow/verify/environment-results"
    evidence_dir.mkdir(parents=True)
    evidence_path = evidence_dir / f"{card_id}.json"
    environment_result = json.loads(
        (
            repo_root / "tests/fixtures/valid/environment-verification-result.synthetic.json"
        ).read_text()
    )
    environment_result["environment_id"] = card_id
    environment_result["environment_spec_sha256"] = hashlib.sha256(
        (environment_dir / "environment.json").read_bytes()
    ).hexdigest()
    environment_result["environment_fingerprint"] = "7" * 64
    evidence_path.write_text(json.dumps(environment_result))
    recommended = card["usage"]["recommended_environment"]
    recommended["verified"] = True
    recommended["fingerprint"] = "7" * 64
    recommended["verification_result"] = evidence_path.relative_to(root).as_posix()
    recommended["verification_result_sha256"] = hashlib.sha256(
        evidence_path.read_bytes()
    ).hexdigest()
    (root / "model_cards" / f"{card_id}.json").write_text(json.dumps(card))


def add_valid_synthetic_runtime_integration(repo_root: Path, root: Path) -> tuple[Path, Path]:
    add_valid_synthetic_integration(repo_root, root)
    card = json.loads((repo_root / "tests/fixtures/valid/model-card.runtime.json").read_text())
    card_id = card["card_id"]
    environment_id = card["usage"]["recommended_environment"]["environment_id"]
    environment_dir = root / "environments" / environment_id
    (environment_dir / "sources.json").write_text(
        json.dumps({"schema_version": "1.0.0", "environment_id": environment_id, "sources": []})
    )
    (environment_dir / "pyproject.toml").write_text(
        "[project]\nname='synthetic-environment'\nversion='0.0.0'\n"
        "requires-python='==3.12.13'\ndependencies=[]\n"
    )
    (environment_dir / "uv.lock").write_text("version = 1\nrevision = 3\n")

    shutil.copy(repo_root / "project_spec.md", root / "project_spec.md")
    shutil.copytree(
        repo_root / "skills/audio-model-onboarding",
        root / "skills/audio-model-onboarding",
    )
    workflow_root = root / "onboarding_reports/synthetic-workflow"
    integrate_dir = workflow_root / "integrate"
    integrate_dir.mkdir(parents=True)
    integration_report = integrate_dir / "integration-report.json"
    integration_report.write_text("{}\n")
    handoff_path = integrate_dir / "handoff.json"
    handoff = {
        "schema_version": "1.0.0",
        "workflow_id": "synthetic-workflow",
        "phase": "integrate",
        "handoff_status": "accepted",
        "repository_commit": "a" * 40,
        "project_spec_sha256": hashlib.sha256((root / "project_spec.md").read_bytes()).hexdigest(),
        "canonical_skill_fingerprint": skill_fingerprint(root),
        "input_artifacts": [],
        "output_artifacts": [
            {
                "path": "onboarding_reports/synthetic-workflow/integrate/integration-report.json",
                "sha256": hashlib.sha256(integration_report.read_bytes()).hexdigest(),
                "media_type": "application/json",
                "originating_phase": "integrate",
                "canonical_role": "integration-report",
                "external_label": None,
            }
        ],
        "artifact_supersessions": [],
        "target_variant_ids": ["synthetic-variant"],
        "target_checkpoint_ids": [card["checkpoint"]["checkpoint_id"]],
        "target_card_ids": [card_id],
        "user_decisions_consumed": [],
        "unresolved_items_carried_forward": [],
        "validation_summary": {"passed": True, "checks": ["synthetic"], "errors": []},
        "allowed_next_modes": ["verify"],
        "lifecycle_promotion": None,
        "superseded_handoff_sha256": None,
    }
    handoff_path.write_text(json.dumps(handoff, indent=2))
    (workflow_root / "workflow.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "workflow_id": "synthetic-workflow",
                "model_family": "Synthetic",
                "target_variant_ids": ["synthetic-variant"],
                "target_checkpoint_ids": [card["checkpoint"]["checkpoint_id"]],
                "target_card_ids": [card_id],
                "created_repository_commit": "a" * 40,
                "current_accepted_phase": "integrate",
                "accepted_phase_paths": [
                    {
                        "phase": "integrate",
                        "handoff_path": (
                            "onboarding_reports/synthetic-workflow/integrate/handoff.json"
                        ),
                    }
                ],
                "status": "active",
            },
            indent=2,
        )
    )

    target = json.loads(
        (repo_root / "tests/fixtures/valid/runtime-verification-target.synthetic.json").read_text()
    )
    target["future_card_id"] = card_id
    target["registry_identity"] = card_id
    target["public_model_entry_point"] = card["identity"]["wrapper_entry_point"]
    target["checkpoint"] = dict(card["checkpoint"])
    target["checkpoint"]["observed_sha256"] = None
    target["environment_id"] = environment_id
    target["accepted_integration_handoff"]["sha256"] = hashlib.sha256(
        handoff_path.read_bytes()
    ).hexdigest()
    target["environment_spec_sha256"] = hashlib.sha256(
        (environment_dir / "environment.json").read_bytes()
    ).hexdigest()
    target["source_manifest"] = {
        "path": f"environments/{environment_id}/sources.json",
        "sha256": hashlib.sha256((environment_dir / "sources.json").read_bytes()).hexdigest(),
    }
    target_path = workflow_root / "verify/runtime-targets/synthetic-runtime-target.json"
    target_path.parent.mkdir(parents=True)
    target_path.write_text(json.dumps(target))

    report = json.loads(
        (repo_root / "tests/fixtures/valid/verification-report.synthetic.json").read_text()
    )
    report.update(
        {
            "schema_version": "2.0.0",
            "runtime_target_id": target["target_id"],
            "workflow_id": target["workflow_id"],
            "integrated_variant_id": target["integrated_variant_id"],
            "checkpoint_id": target["checkpoint"]["checkpoint_id"],
            "public_model_entry_point": target["public_model_entry_point"],
            "integration_handoff_sha256": target["accepted_integration_handoff"]["sha256"],
            "environment_spec_sha256": target["environment_spec_sha256"],
            "source_manifest_sha256": target["source_manifest"]["sha256"],
            "required_check_ids": target["required_check_ids"],
            "optional_check_ids": target["optional_check_ids"],
            "model_card_id": card_id,
            "environment_id": environment_id,
            "environment_fingerprint": "7" * 64,
            "checkpoint_sha256": card["checkpoint"]["observed_sha256"],
            "unsupported_capabilities": [],
            "verification_status": "passed",
        }
    )
    report["checks"] = [
        {"name": check_id, "status": "passed", "details": None}
        for check_id in target["required_check_ids"]
    ]
    report_path = root / f"verification_reports/{card_id}/report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps(report))

    evidence_path = (
        root
        / "onboarding_reports/synthetic-workflow/verify/environment-results"
        / f"{environment_id}.json"
    )
    recommended = card["usage"]["recommended_environment"]
    recommended["fingerprint"] = "7" * 64
    recommended["verification_result"] = evidence_path.relative_to(root).as_posix()
    recommended["verification_result_sha256"] = hashlib.sha256(
        evidence_path.read_bytes()
    ).hexdigest()
    card["runtime_verification_target"] = target_path.relative_to(root).as_posix()
    card["runtime_verification_target_sha256"] = hashlib.sha256(
        target_path.read_bytes()
    ).hexdigest()
    card["verification_report"] = report_path.relative_to(root).as_posix()
    card["verification_report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    (root / "model_cards/synthetic-family-variant-checkpoint.json").write_text(json.dumps(card))
    return evidence_path, report_path


def test_current_empty_integration_layout_passes(repo_root: Path) -> None:
    failures: list[str] = []
    validate_integration_artifacts(repo_root, failures)
    assert failures == []


def test_structurally_valid_synthetic_integration_passes(repo_root: Path, tmp_path: Path) -> None:
    prepare_integration_root(repo_root, tmp_path)
    add_valid_synthetic_integration(repo_root, tmp_path)
    failures: list[str] = []
    validate_integration_artifacts(tmp_path, failures)
    assert failures == []


def test_verified_card_rejects_hash_correct_failed_environment_result(
    repo_root: Path, tmp_path: Path
) -> None:
    prepare_integration_root(repo_root, tmp_path)
    add_valid_synthetic_integration(repo_root, tmp_path)
    evidence_path = next((tmp_path / "onboarding_reports").glob("**/environment-results/*.json"))
    evidence = json.loads(evidence_path.read_text())
    evidence["verification_status"] = "failed"
    evidence["lifecycle_state"] = "materialized"
    evidence["failure_classification"] = "verification_script"
    evidence_path.write_text(json.dumps(evidence))
    card_path = next((tmp_path / "model_cards").glob("*.json"))
    card = json.loads(card_path.read_text())
    card["usage"]["recommended_environment"]["verification_result_sha256"] = hashlib.sha256(
        evidence_path.read_bytes()
    ).hexdigest()
    card_path.write_text(json.dumps(card))

    failures: list[str] = []
    validate_integration_artifacts(tmp_path, failures)

    assert any("environment verification did not succeed" in failure for failure in failures)


def test_verified_card_requires_promoted_environment_result_path(
    repo_root: Path, tmp_path: Path
) -> None:
    prepare_integration_root(repo_root, tmp_path)
    add_valid_synthetic_integration(repo_root, tmp_path)
    evidence_path = next((tmp_path / "onboarding_reports").glob("**/environment-results/*.json"))
    unmanaged_path = tmp_path / "diagnostics/environment-result.json"
    unmanaged_path.parent.mkdir()
    shutil.move(evidence_path, unmanaged_path)
    card_path = next((tmp_path / "model_cards").glob("*.json"))
    card = json.loads(card_path.read_text())
    recommended = card["usage"]["recommended_environment"]
    recommended["verification_result"] = unmanaged_path.relative_to(tmp_path).as_posix()
    recommended["verification_result_sha256"] = hashlib.sha256(
        unmanaged_path.read_bytes()
    ).hexdigest()
    card_path.write_text(json.dumps(card))
    failures: list[str] = []

    validate_integration_artifacts(tmp_path, failures)

    assert any("result path is not canonical" in failure for failure in failures)


def test_runtime_verified_card_accepts_passed_matching_evidence(
    repo_root: Path, tmp_path: Path
) -> None:
    prepare_integration_root(repo_root, tmp_path)
    add_valid_synthetic_runtime_integration(repo_root, tmp_path)
    failures: list[str] = []

    validate_integration_artifacts(tmp_path, failures)

    assert failures == []


def test_runtime_verified_card_rejects_incomplete_required_report(
    repo_root: Path, tmp_path: Path
) -> None:
    prepare_integration_root(repo_root, tmp_path)
    _, report_path = add_valid_synthetic_runtime_integration(repo_root, tmp_path)
    report = json.loads(report_path.read_text())
    report["checks"].pop()
    report_path.write_text(json.dumps(report))
    card_path = next((tmp_path / "model_cards").glob("*.json"))
    card = json.loads(card_path.read_text())
    card["verification_report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    card_path.write_text(json.dumps(card))
    failures: list[str] = []

    validate_integration_artifacts(tmp_path, failures)

    assert any("invalid verification report" in failure for failure in failures)


def test_runtime_verified_card_rejects_target_report_check_contract_drift(
    repo_root: Path, tmp_path: Path
) -> None:
    prepare_integration_root(repo_root, tmp_path)
    _, report_path = add_valid_synthetic_runtime_integration(repo_root, tmp_path)
    report = json.loads(report_path.read_text())
    report["optional_check_ids"] = ["different-optional-check"]
    report_path.write_text(json.dumps(report))
    card_path = next((tmp_path / "model_cards").glob("*.json"))
    card = json.loads(card_path.read_text())
    card["verification_report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    card_path.write_text(json.dumps(card))
    failures: list[str] = []

    validate_integration_artifacts(tmp_path, failures)

    assert any("optional-check contract disagrees" in failure for failure in failures)


def test_runtime_verified_card_rejects_failed_report(repo_root: Path, tmp_path: Path) -> None:
    prepare_integration_root(repo_root, tmp_path)
    _, report_path = add_valid_synthetic_runtime_integration(repo_root, tmp_path)
    report = json.loads(report_path.read_text())
    report["verification_status"] = "failed"
    report["checks"][0]["status"] = "failed"
    report_path.write_text(json.dumps(report))
    card_path = next((tmp_path / "model_cards").glob("*.json"))
    card = json.loads(card_path.read_text())
    card["verification_report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    card_path.write_text(json.dumps(card))
    failures: list[str] = []

    validate_integration_artifacts(tmp_path, failures)

    assert any("runtime verification did not succeed" in failure for failure in failures)


def test_runtime_verified_card_rejects_report_fingerprint_drift(
    repo_root: Path, tmp_path: Path
) -> None:
    prepare_integration_root(repo_root, tmp_path)
    _, report_path = add_valid_synthetic_runtime_integration(repo_root, tmp_path)
    report = json.loads(report_path.read_text())
    report["environment_fingerprint"] = "f" * 64
    report_path.write_text(json.dumps(report))
    card_path = next((tmp_path / "model_cards").glob("*.json"))
    card = json.loads(card_path.read_text())
    card["verification_report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    card_path.write_text(json.dumps(card))
    failures: list[str] = []

    validate_integration_artifacts(tmp_path, failures)

    assert any("verification report fingerprint disagrees" in failure for failure in failures)


def test_runtime_verified_card_rejects_target_report_association_drift(
    repo_root: Path, tmp_path: Path
) -> None:
    prepare_integration_root(repo_root, tmp_path)
    _, report_path = add_valid_synthetic_runtime_integration(repo_root, tmp_path)
    report = json.loads(report_path.read_text())
    report["public_model_entry_point"] = "torch_dae.synthetic:OtherModel"
    report_path.write_text(json.dumps(report))
    card_path = next((tmp_path / "model_cards").glob("*.json"))
    card = json.loads(card_path.read_text())
    card["verification_report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    card_path.write_text(json.dumps(card))
    failures: list[str] = []

    validate_integration_artifacts(tmp_path, failures)

    assert any("public model entry point disagrees" in failure for failure in failures)


def test_invalid_model_card_fails(repo_root: Path, tmp_path: Path) -> None:
    prepare_integration_root(repo_root, tmp_path)
    (tmp_path / "model_cards/invalid.json").write_text("{}")
    failures: list[str] = []
    validate_integration_artifacts(tmp_path, failures)
    assert any("invalid model card" in failure for failure in failures)


def test_missing_wrapper_symbol_fails(repo_root: Path, tmp_path: Path) -> None:
    prepare_integration_root(repo_root, tmp_path)
    card = repo_root / "tests/fixtures/valid/model-card.analyzed.json"
    shutil.copy(card, tmp_path / "model_cards/card.json")
    (tmp_path / "src/torch_dae/synthetic.py").write_text("class OtherModel:\n    pass\n")
    failures: list[str] = []
    validate_integration_artifacts(tmp_path, failures)
    assert any("wrapper symbol" in failure for failure in failures)


def test_checkpoint_binary_fails(repo_root: Path, tmp_path: Path) -> None:
    prepare_integration_root(repo_root, tmp_path)
    binary = tmp_path / "tests/fixture.ckpt"
    binary.parent.mkdir()
    binary.write_bytes(b"\x00\x01checkpoint")
    failures: list[str] = []
    validate_integration_artifacts(tmp_path, failures)
    assert any("binary is forbidden" in failure for failure in failures)


def test_runtime_verified_card_without_report_fails(repo_root: Path, tmp_path: Path) -> None:
    prepare_integration_root(repo_root, tmp_path)
    card = json.loads((repo_root / "tests/fixtures/valid/model-card.runtime.json").read_text())
    (tmp_path / "model_cards/runtime.json").write_text(json.dumps(card))
    failures: list[str] = []
    validate_integration_artifacts(tmp_path, failures)
    assert any("lacks its verification report" in failure for failure in failures)
