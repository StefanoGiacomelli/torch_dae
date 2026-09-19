"""Repository validation for public safety and integrated-model structure."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable
from copy import deepcopy
from importlib.metadata import distributions
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from pydantic import BaseModel

from torch_dae.cards.models import ModelCard, ModelCardLifecycle
from torch_dae.cards.validation import load_json, validate_model_card_path
from torch_dae.core.checkpoint import (
    CheckpointMaterializationRecord,
    CheckpointSpec,
    checkpoint_specification_fingerprint,
)
from torch_dae.core.embeddings import EmbeddingSpec
from torch_dae.core.registry import ModelCardRegistry
from torch_dae.environment.results import (
    ArtifactEvidence,
    EnvironmentDependencyClosureResult,
    EnvironmentLifecycleState,
    EnvironmentMaterializationResult,
    EnvironmentVerificationResult,
)
from torch_dae.environment.runtime import EnvironmentMaterializationRecord, RuntimeReportSink
from torch_dae.environment.specification import EnvironmentSourcesManifest, EnvironmentSpecification
from torch_dae.environment.verification import VerificationReport
from torch_dae.onboarding.contracts import (
    AnalysisReport,
    CleanupReceipt,
    DependencyEvidenceRecord,
    EnvironmentResolutionReport,
    EvidenceItem,
    PhaseHandoffManifest,
    SkillEvaluationScenario,
    WorkflowRecord,
)
from torch_dae.onboarding.evaluation import evaluate_analysis_report
from torch_dae.onboarding.handoff import validate_workflow
from torch_dae.onboarding.inspection import (
    InspectionBudget,
    generate_environment_candidates,
    inspect_dependencies,
    inspect_scenario_repository,
)
from torch_dae.runtime_verification import (
    RuntimeVerificationTarget,
    validate_runtime_verification_target,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DEPS = {"torch", "torchaudio", "torchvision", "transformers", "tensorflow", "jax", "librosa"}
BINARY_MODEL_SUFFIXES = {".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".onnx"}
REQUIRED = [
    ".gitattributes",
    "project_spec.md",
    "pyproject.toml",
    "uv.lock",
    "scripts/onboarding_handoff.py",
    "scripts/check_worktree_patch.py",
    "skills/audio-model-onboarding/SKILL.md",
    ".agents/skills/audio-model-onboarding",
    ".claude/skills/audio-model-onboarding",
    "schemas/model-card.schema.json",
    "schemas/checkpoint.schema.json",
    "schemas/checkpoint-authority-resolution.schema.json",
    "schemas/checkpoint-materialization.schema.json",
    "schemas/environment.schema.json",
    "schemas/environment-sources.schema.json",
    "schemas/environment-materialization-result.schema.json",
    "schemas/environment-dependency-closure-result.schema.json",
    "schemas/environment-verification-result.schema.json",
    "schemas/embedding.schema.json",
    "schemas/verification-report.schema.json",
    "schemas/runtime-verification-target.schema.json",
    "schemas/analysis-report.schema.json",
    "schemas/environment-resolution-report.schema.json",
    "schemas/workflow-record.schema.json",
    "schemas/phase-handoff.schema.json",
    "schemas/cleanup-receipt.schema.json",
    "src/torch_dae",
    "tests/fixtures",
]


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_IGNORED_RUNTIME_CACHE_PREFIX = ".torch-dae/"


def _resolve_checkpoint_materialization_evidence(
    root: Path, evidence: ArtifactEvidence
) -> Path | None:
    """Resolve the file whose bytes must match ``evidence.sha256``.

    ``evidence.path`` is recorded, at runtime-verification time, as wherever the checkpoint
    materialization record happened to live -- which is legitimately the ignored, mutable
    ``.torch-dae/`` runtime cache (Section: "Treat `.torch-dae/` as ignored runtime state").
    That cache is explicitly allowed to be refreshed (e.g. by a later, unrelated checkpoint
    acquisition for the same checkpoint id/hash) without invalidating already-accepted evidence;
    only the *checkpoint payload* SHA-256 is the durable contract, not the cache-record's
    incidental fields (acquisition timestamp, per-request log correlation ids, etc.).

    So: first try the literal ``evidence.path`` (fast path, exact match). If that path is under
    the ignored runtime-cache prefix and no longer matches, fall back to any committed, tracked
    "checkpoint-materializations" evidence copy (written under `onboarding_reports/**/` at
    acceptance time) whose bytes match ``evidence.sha256`` -- these are durable, hash-addressed,
    and never expected to change. This keeps `.torch-dae/` cache churn from being able to fail
    repository validation, without weakening the guarantee: either way, the resolved file's bytes
    must still hash to the pinned ``evidence.sha256``.
    """

    literal_path = (root / evidence.path).resolve()
    try:
        literal_path.relative_to(root)
    except ValueError:
        return None
    if literal_path.is_file() and _sha256(literal_path) == evidence.sha256:
        return literal_path

    if not evidence.path.startswith(_IGNORED_RUNTIME_CACHE_PREFIX):
        return None

    onboarding_reports_root = root / "onboarding_reports"
    if not onboarding_reports_root.is_dir():
        return None
    for candidate in sorted(onboarding_reports_root.glob("**/checkpoint-materializations/*.json")):
        if candidate.is_file() and _sha256(candidate) == evidence.sha256:
            return candidate
    return None


def git_ignored(path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", path],
        cwd=ROOT,
        check=False,
    )
    return result.returncode == 0


def schema_validate(fixture: Path, schema: Path) -> None:
    data = load_json(fixture)
    Draft202012Validator(load_json(schema), format_checker=FormatChecker()).validate(data)


def pydantic_validate(fixture: Path) -> None:
    kind = fixture.name.split(".")[0]
    if kind == "model-card":
        validate_model_card_path(fixture)
        return
    model_by_kind: dict[str, type[BaseModel]] = {
        "checkpoint": CheckpointSpec,
        "embedding": EmbeddingSpec,
        "environment": EnvironmentSpecification,
        "environment-sources": EnvironmentSourcesManifest,
        "environment-materialization-result": EnvironmentMaterializationResult,
        "environment-dependency-closure-result": EnvironmentDependencyClosureResult,
        "environment-verification-result": EnvironmentVerificationResult,
        "verification-report": VerificationReport,
        "runtime-verification-target": RuntimeVerificationTarget,
        "analysis-report": AnalysisReport,
        "environment-resolution-report": EnvironmentResolutionReport,
        "workflow-record": WorkflowRecord,
        "phase-handoff": PhaseHandoffManifest,
        "cleanup-receipt": CleanupReceipt,
    }
    model_by_kind[kind].model_validate(load_json(fixture))


def validate_fixture(path: Path, schema: Path) -> None:
    kind = path.name.split(".")[0]
    if kind == "model-card":
        validate_model_card_path(path, schema)
        return
    schema_validate(path, schema)
    pydantic_validate(path)


def semantic_invalid_fixture_fails(path: Path) -> bool:
    """Return whether a semantic-cross-reference fixture fails repository validation."""

    try:
        pydantic_validate(path)
    except Exception:
        return True
    name = path.name
    if name == "environment.model-card-id-mismatch.json":
        specification = EnvironmentSpecification.model_validate_json(path.read_text())
        return specification.model_card_id != "synthetic-family-variant-checkpoint"
    if name == "environment-sources.environment-id-mismatch.json":
        valid_spec = EnvironmentSpecification.model_validate_json(
            (ROOT / "tests/fixtures/valid/environment.synthetic.json").read_text()
        )
        manifest = EnvironmentSourcesManifest.model_validate_json(path.read_text())
        return manifest.environment_id != valid_spec.environment_id
    return False


def _tracked_paths(root: Path) -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return ()
    return tuple(root / item.decode() for item in result.stdout.split(b"\0") if item)


def numbered_stage_errors(root: Path) -> list[str]:
    """Reject numbered internal-stage labels in tracked paths and textual contents."""

    stage_word = "".join(("ph", "ase"))
    pattern = re.compile(rf"{stage_word}[ _-]*0[0-9]", re.IGNORECASE)
    errors: list[str] = []
    for path in _tracked_paths(root):
        relative = path.relative_to(root)
        if relative.as_posix() == "project_spec.md":
            continue
        if pattern.search(relative.as_posix()):
            errors.append(f"numbered internal-stage label in tracked path: {relative}")
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if pattern.search(text):
            errors.append(f"numbered internal-stage label in tracked text: {relative}")
    return errors


def _validate_with_schema(path: Path, schema: Path, model: type[BaseModel]) -> BaseModel:
    payload = load_json(path)
    Draft202012Validator(load_json(schema), format_checker=FormatChecker()).validate(payload)
    return model.model_validate(payload)


def _wrapper_symbol_exists(root: Path, entry_point: str) -> bool:
    module_name, separator, symbol = entry_point.partition(":")
    if not separator or not module_name or not symbol:
        return False
    module_path = root / "src" / Path(*module_name.split("."))
    candidates = (module_path.with_suffix(".py"), module_path / "__init__.py")
    source_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if source_path is None:
        return False
    try:
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return False

    definitions = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    if any(isinstance(node, definitions) and node.name == symbol for node in tree.body):
        return True

    # Packages may intentionally expose runtime-heavy public symbols lazily
    # through module-level __getattr__, while keeping root imports lightweight.
    has_module_getattr = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__getattr__"
        for node in tree.body
    )
    if not has_module_getattr:
        return False

    lazy_exports: set[str] = set()

    for node in tree.body:
        name: str | None = None
        value: ast.expr | None = None

        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                name = target.id
                value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            value = node.value

        if name is None or value is None:
            continue
        if name != "__all__" and not name.endswith("_EXPORTS"):
            continue
        if not isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            continue

        for element in value.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                lazy_exports.add(element.value)

    return symbol in lazy_exports


def _is_binary_asset(path: Path) -> bool:
    try:
        data = path.read_bytes()
    except OSError:
        return True
    if b"\0" in data:
        return True
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


def validate_integration_artifacts(root: Path, failures: list[str]) -> None:
    """Validate empty or structurally populated public integration directories."""

    schema_root = root / "schemas"
    cards: dict[str, tuple[Path, ModelCard]] = {}
    for path in sorted((root / "model_cards").glob("**/*.json")):
        try:
            card = validate_model_card_path(path, schema_root / "model-card.schema.json")
        except Exception as exc:
            fail(f"invalid model card {path.relative_to(root)}: {exc}", failures)
            continue
        if card.card_id in cards:
            fail(f"duplicate model card id: {card.card_id}", failures)
            continue
        cards[card.card_id] = (path, card)
        if not _wrapper_symbol_exists(root, card.identity.wrapper_entry_point):
            fail(
                f"wrapper symbol does not resolve statically: {card.identity.wrapper_entry_point}",
                failures,
            )

        if card.card_status in {
            ModelCardLifecycle.ENVIRONMENT_RESOLVED,
            ModelCardLifecycle.CHECKPOINT_VERIFIED,
            ModelCardLifecycle.RUNTIME_VERIFIED,
            ModelCardLifecycle.PROFILED,
        }:
            recommended = card.usage.recommended_environment
            specification_path = root / recommended.specification
            environment_dir = specification_path.parent
            required = {
                "environment.json",
                "pyproject.toml",
                "uv.lock",
                "sources.json",
                "verify_environment.py",
            }
            missing = sorted(name for name in required if not (environment_dir / name).is_file())
            if missing:
                fail(f"missing environment artifacts for {card.card_id}: {missing}", failures)
                continue
            try:
                specification = _validate_with_schema(
                    environment_dir / "environment.json",
                    schema_root / "environment.schema.json",
                    EnvironmentSpecification,
                )
                sources = _validate_with_schema(
                    environment_dir / "sources.json",
                    schema_root / "environment-sources.schema.json",
                    EnvironmentSourcesManifest,
                )
            except Exception as exc:
                fail(f"invalid environment artifacts for {card.card_id}: {exc}", failures)
                continue
            assert isinstance(specification, EnvironmentSpecification)
            assert isinstance(sources, EnvironmentSourcesManifest)
            expected_prefix = environment_dir.relative_to(root).as_posix()
            if not (
                recommended.environment_id == specification.environment_id == sources.environment_id
            ):
                fail(f"environment IDs disagree for {card.card_id}", failures)
            expected_paths = {
                "specification": f"{expected_prefix}/environment.json",
                "lockfile": f"{expected_prefix}/uv.lock",
                "project_file": f"{expected_prefix}/pyproject.toml",
                "sources_file": f"{expected_prefix}/sources.json",
                "verification": f"{expected_prefix}/verify_environment.py",
            }
            observed_paths = {
                "specification": recommended.specification,
                "lockfile": specification.lockfile,
                "project_file": specification.project_file,
                "sources_file": specification.sources_file,
                "verification": specification.verification.script,
            }
            if recommended.lockfile != expected_paths["lockfile"]:
                fail(f"model-card lockfile path disagrees for {card.card_id}", failures)
            for label, expected in expected_paths.items():
                if observed_paths[label] != expected:
                    fail(f"{label} path disagrees for {card.card_id}", failures)

            assert recommended.verification_result is not None
            assert recommended.verification_result_sha256 is not None
            assert recommended.fingerprint is not None
            environment_result_path = root / recommended.verification_result
            result_parts = Path(recommended.verification_result).parts
            if not (
                len(result_parts) >= 4
                and result_parts[0] == "onboarding_reports"
                and result_parts[2] == "verify"
                and environment_result_path.suffix == ".json"
            ):
                fail(
                    f"environment verification result path is not canonical for {card.card_id}",
                    failures,
                )
            try:
                environment_result = _validate_with_schema(
                    environment_result_path,
                    schema_root / "environment-verification-result.schema.json",
                    EnvironmentVerificationResult,
                )
            except Exception as exc:
                fail(f"invalid environment verification result for {card.card_id}: {exc}", failures)
            else:
                assert isinstance(environment_result, EnvironmentVerificationResult)
                if _sha256(environment_result_path) != recommended.verification_result_sha256:
                    fail(
                        f"environment verification result hash disagrees for {card.card_id}",
                        failures,
                    )
                if environment_result.environment_id != recommended.environment_id:
                    fail(
                        f"environment verification identity disagrees for {card.card_id}", failures
                    )
                if environment_result.environment_fingerprint != recommended.fingerprint:
                    fail(f"environment fingerprint disagrees for {card.card_id}", failures)
                if environment_result.environment_spec_sha256 != _sha256(specification_path):
                    fail(
                        f"environment specification evidence disagrees for {card.card_id}", failures
                    )
                if not (
                    environment_result.verification_status == "passed"
                    and environment_result.lifecycle_state == EnvironmentLifecycleState.VERIFIED
                    and environment_result.failure_classification is None
                ):
                    fail(f"environment verification did not succeed for {card.card_id}", failures)

    targets: dict[str, tuple[Path, RuntimeVerificationTarget]] = {}
    target_candidates = sorted(
        set((root / "onboarding_reports").glob("**/runtime-verification-target*.json"))
        | set((root / "onboarding_reports").glob("**/runtime-targets/*.json"))
    )
    for path in target_candidates:
        try:
            target = _validate_with_schema(
                path,
                schema_root / "runtime-verification-target.schema.json",
                RuntimeVerificationTarget,
            )
            assert isinstance(target, RuntimeVerificationTarget)
            validate_runtime_verification_target(target, root)
        except Exception as exc:
            fail(f"invalid runtime verification target {path.relative_to(root)}: {exc}", failures)
            continue
        if target.target_id in targets:
            fail(f"duplicate runtime verification target ID: {target.target_id}", failures)
        targets[target.target_id] = (path, target)

    reports: dict[Path, VerificationReport] = {}
    for path in sorted((root / "verification_reports").glob("**/*.json")):
        try:
            verified_report = _validate_with_schema(
                path,
                schema_root / "verification-report.schema.json",
                VerificationReport,
            )
        except Exception as exc:
            fail(f"invalid verification report {path.relative_to(root)}: {exc}", failures)
            continue
        assert isinstance(verified_report, VerificationReport)
        reports[path.resolve()] = verified_report
        card_item = cards.get(verified_report.model_card_id)
        if verified_report.schema_version == "2.0.0":
            target_item = targets.get(verified_report.runtime_target_id or "")
            if target_item is None:
                fail(f"verification report references missing runtime target: {path}", failures)
                continue
            target = target_item[1]
            if target.schema_version != "2.0.0":
                fail(
                    f"target-aware verification report requires a completeness-aware "
                    f"runtime target: {path}",
                    failures,
                )
            associations = {
                "workflow": (verified_report.workflow_id, target.workflow_id),
                "variant": (verified_report.integrated_variant_id, target.integrated_variant_id),
                "checkpoint": (verified_report.checkpoint_id, target.checkpoint.checkpoint_id),
                "environment": (verified_report.environment_id, target.environment_id),
                "integration handoff": (
                    verified_report.integration_handoff_sha256,
                    target.accepted_integration_handoff.sha256,
                ),
                "environment specification": (
                    verified_report.environment_spec_sha256,
                    target.environment_spec_sha256,
                ),
                "source manifest": (
                    verified_report.source_manifest_sha256,
                    target.source_manifest.sha256,
                ),
                "public model entry point": (
                    verified_report.public_model_entry_point,
                    target.public_model_entry_point,
                ),
            }
            for label, values in associations.items():
                if values[0] != values[1]:
                    fail(f"verification report {label} disagrees for {path}", failures)
            if target.checkpoint_acquisition_policy.require_authority:
                if (
                    verified_report.checkpoint_specification_fingerprint
                    != checkpoint_specification_fingerprint(target.checkpoint)
                ):
                    fail(
                        f"verification report checkpoint specification fingerprint disagrees "
                        f"for {path}",
                        failures,
                    )
                evidence = verified_report.checkpoint_materialization
                if evidence is None:
                    fail(
                        f"authoritative verification report lacks checkpoint materialization "
                        f"provenance: {path}",
                        failures,
                    )
                else:
                    materialization_path = _resolve_checkpoint_materialization_evidence(
                        root, evidence
                    )
                    try:
                        if materialization_path is None:
                            raise ValueError("materialization evidence SHA-256 mismatch")
                        materialization = CheckpointMaterializationRecord.model_validate_json(
                            materialization_path.read_text()
                        )
                    except Exception as exc:
                        fail(
                            f"invalid authoritative checkpoint materialization for {path}: {exc}",
                            failures,
                        )
                    else:
                        authority = target.checkpoint.authority
                        if (
                            authority is None
                            or materialization.authority != authority
                            or materialization.expected_size_bytes != authority.expected_size_bytes
                            or materialization.observed_size_bytes != authority.expected_size_bytes
                            or materialization.published_checksums != authority.published_checksums
                            or materialization.observed_sha256 != verified_report.checkpoint_sha256
                            or materialization.specification_fingerprint
                            != checkpoint_specification_fingerprint(target.checkpoint)
                        ):
                            fail(
                                f"authoritative checkpoint provenance disagrees for {path}",
                                failures,
                            )
            if verified_report.required_check_ids != target.required_check_ids:
                fail(f"verification report required-check contract disagrees for {path}", failures)
            if verified_report.optional_check_ids != target.optional_check_ids:
                fail(f"verification report optional-check contract disagrees for {path}", failures)
            check_names = [check.name for check in verified_report.checks]
            if len(check_names) != len(set(check_names)):
                fail(f"verification report contains duplicate check names: {path}", failures)
            missing_required = sorted(set(target.required_check_ids) - set(check_names))
            if missing_required:
                fail(
                    f"verification report required-check coverage is incomplete for {path}: "
                    f"{missing_required}",
                    failures,
                )
            undeclared_checks = sorted(
                set(check_names) - set(target.required_check_ids) - set(target.optional_check_ids)
            )
            if undeclared_checks:
                fail(
                    f"verification report contains checks undeclared by its target for {path}: "
                    f"{undeclared_checks}",
                    failures,
                )
            status_by_name = {check.name: check.status for check in verified_report.checks}
            unsupported_required = sorted(
                name
                for name in target.required_check_ids
                if status_by_name.get(name) == "unsupported"
            )
            if unsupported_required:
                fail(
                    f"verification report has unsupported required checks for {path}: "
                    f"{unsupported_required}",
                    failures,
                )
            if verified_report.verification_status == "passed":
                unpassed_required = sorted(
                    name
                    for name in target.required_check_ids
                    if status_by_name.get(name) != "passed"
                )
                if unpassed_required:
                    fail(
                        f"verification report required checks did not all pass for {path}: "
                        f"{unpassed_required}",
                        failures,
                    )
            if (
                target.future_card_id is not None
                and target.future_card_id != verified_report.model_card_id
            ):
                fail(f"verification report future card identity disagrees for {path}", failures)
        elif card_item is None:
            fail(f"legacy verification report references missing card: {path}", failures)
        if card_item is not None:
            card = card_item[1]
            if verified_report.environment_id != card.usage.recommended_environment.environment_id:
                fail(f"verification report environment disagrees for {card.card_id}", failures)
            if verified_report.checkpoint_sha256 != card.checkpoint.observed_sha256:
                fail(f"verification report checkpoint disagrees for {card.card_id}", failures)
            if (
                verified_report.environment_fingerprint
                != card.usage.recommended_environment.fingerprint
            ):
                fail(f"verification report fingerprint disagrees for {card.card_id}", failures)

    for _, card in cards.values():
        if (
            card.card_status
            in {
                ModelCardLifecycle.CHECKPOINT_VERIFIED,
                ModelCardLifecycle.RUNTIME_VERIFIED,
                ModelCardLifecycle.PROFILED,
            }
            and card.checkpoint.schema_version == "2.0.0"
        ):
            expected_fingerprint = checkpoint_specification_fingerprint(card.checkpoint)
            if card.checkpoint_specification_fingerprint != expected_fingerprint:
                fail(
                    f"authority-complete card checkpoint fingerprint disagrees for {card.card_id}",
                    failures,
                )
            evidence = card.checkpoint_materialization
            if evidence is None:
                fail(
                    f"authority-complete card lacks checkpoint materialization for {card.card_id}",
                    failures,
                )
            else:
                materialization_path = _resolve_checkpoint_materialization_evidence(root, evidence)
                try:
                    if materialization_path is None:
                        raise ValueError("materialization evidence SHA-256 mismatch")
                    materialization = CheckpointMaterializationRecord.model_validate_json(
                        materialization_path.read_text()
                    )
                except Exception as exc:
                    fail(
                        f"invalid card checkpoint materialization for {card.card_id}: {exc}",
                        failures,
                    )
                else:
                    authority = card.checkpoint.authority
                    if (
                        authority is None
                        or materialization.authority != authority
                        or materialization.expected_size_bytes != authority.expected_size_bytes
                        or materialization.observed_size_bytes != authority.expected_size_bytes
                        or materialization.published_checksums != authority.published_checksums
                        or materialization.observed_sha256 != card.checkpoint.observed_sha256
                        or materialization.specification_fingerprint != expected_fingerprint
                    ):
                        fail(
                            f"card checkpoint materialization disagrees for {card.card_id}",
                            failures,
                        )
        if card.card_status not in {
            ModelCardLifecycle.RUNTIME_VERIFIED,
            ModelCardLifecycle.PROFILED,
        }:
            continue
        assert card.verification_report is not None
        assert card.verification_report_sha256 is not None
        assert card.runtime_verification_target is not None
        assert card.runtime_verification_target_sha256 is not None
        report_path = (root / card.verification_report).resolve()
        referenced_report = reports.get(report_path)
        if referenced_report is None:
            fail(f"runtime-verified card lacks its verification report: {card.card_id}", failures)
            continue
        if _sha256(report_path) != card.verification_report_sha256:
            fail(f"verification report hash disagrees for {card.card_id}", failures)
        if not referenced_report.successful:
            fail(f"runtime verification did not succeed for {card.card_id}", failures)
        if referenced_report.schema_version != "2.0.0":
            fail(
                f"runtime-verified card requires a target-aware verification report: "
                f"{card.card_id}",
                failures,
            )
        target_path = (root / card.runtime_verification_target).resolve()
        target_item = targets.get(referenced_report.runtime_target_id or "")
        if target_item is None or target_item[0].resolve() != target_path:
            fail(f"runtime target reference disagrees for {card.card_id}", failures)
            continue
        target = target_item[1]
        if target.schema_version != "2.0.0":
            fail(
                f"runtime-verified card requires a completeness-aware runtime target: "
                f"{card.card_id}",
                failures,
            )
        if _sha256(target_path) != card.runtime_verification_target_sha256:
            fail(f"runtime target hash disagrees for {card.card_id}", failures)
        if (
            referenced_report.required_check_ids != target.required_check_ids
            or referenced_report.optional_check_ids != target.optional_check_ids
        ):
            fail(f"runtime verification check contract disagrees for {card.card_id}", failures)
        report_checks = {check.name: check.status for check in referenced_report.checks}
        incomplete_required = sorted(
            name for name in target.required_check_ids if report_checks.get(name) != "passed"
        )
        if incomplete_required:
            fail(
                f"runtime-verified card has incomplete required runtime evidence for "
                f"{card.card_id}: {incomplete_required}",
                failures,
            )
        if (
            referenced_report.model_card_id != card.card_id
            or target.future_card_id != card.card_id
            or target.checkpoint.checkpoint_id != card.checkpoint.checkpoint_id
            or target.environment_id != card.usage.recommended_environment.environment_id
            or target.public_model_entry_point != card.identity.wrapper_entry_point
        ):
            fail(f"runtime verification evidence identity disagrees for {card.card_id}", failures)

    excluded_roots = {".git", ".torch-dae", ".venv", "build", "dist"}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in BINARY_MODEL_SUFFIXES:
            continue
        if any(part in excluded_roots for part in path.relative_to(root).parts):
            continue
        if _is_binary_asset(path):
            fail(
                f"committed model/checkpoint binary is forbidden: {path.relative_to(root)}",
                failures,
            )


def validate_onboarding_reports(
    root: Path,
    failures: list[str],
) -> list[dict[str, object]]:
    """Validate committed handoffs and return informational control-plane drift."""

    control_plane_drift: list[dict[str, object]] = []
    reports_root = root / "onboarding_reports"
    if not reports_root.exists():
        fail("committed onboarding_reports root is missing", failures)
        return control_plane_drift
    schema_root = root / "schemas"
    for workflow_path in sorted(reports_root.glob("*/workflow.json")):
        workflow_id = workflow_path.parent.name
        try:
            _validate_with_schema(
                workflow_path,
                schema_root / "workflow-record.schema.json",
                WorkflowRecord,
            )
            workflow = WorkflowRecord.model_validate_json(workflow_path.read_text())
            for reference in workflow.accepted_phase_paths:
                _validate_with_schema(
                    root / reference.handoff_path,
                    schema_root / "phase-handoff.schema.json",
                    PhaseHandoffManifest,
                )
            validation = validate_workflow(root, workflow_id)
            if validation["control_plane_drift"] is True:
                control_plane_drift.append(
                    {
                        "workflow_id": workflow_id,
                        "canonical_skill_drift": validation["canonical_skill_drift"],
                        "project_spec_drift": validation["project_spec_drift"],
                        "historical_control_planes": validation["historical_control_planes"],
                        "current_control_plane": validation["current_control_plane"],
                    }
                )
        except Exception as exc:
            fail(f"invalid onboarding workflow {workflow_id}: {exc}", failures)
    for path in reports_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in {".diff", ".json", ".md"}:
            fail(f"forbidden onboarding report artifact: {path.relative_to(root)}", failures)
    return control_plane_drift


def root_dependency_errors(root: Path) -> list[str]:
    try:
        project = tomllib.loads((root / "pyproject.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return [f"cannot inspect root dependencies: {exc}"]
    dependencies = project.get("project", {}).get("dependencies", ())
    names = {
        re.split(r"[^A-Za-z0-9_.-]", str(requirement), maxsplit=1)[0].lower()
        for requirement in dependencies
    }
    forbidden = sorted(names & MODEL_DEPS)
    return (
        [f"model-specific dependencies declared in root project: {forbidden}"] if forbidden else []
    )


def expect_model_failure(label: str, failures: list[str], func: Callable[[], object]) -> None:
    try:
        func()
    except Exception:
        return
    fail(f"behavioral smoke unexpectedly passed: {label}", failures)


def onboarding_behavioral_smoke(failures: list[str]) -> None:
    analysis_data = load_json(ROOT / "tests/fixtures/valid/analysis-report.synthetic.json")
    env_data = load_json(ROOT / "tests/fixtures/valid/environment-resolution-report.synthetic.json")

    verified_agent = deepcopy(analysis_data)
    verified_agent["source_strategy_candidates"][0]["status"] = "verified_upstream_fact"
    verified_agent["evidence_items"][0]["kind"] = "agent_inference"
    verified_agent["evidence_items"][0]["claim_status"] = "reasoned_inference"
    verified_agent["evidence_items"][0]["rationale"] = "Agent inference is not proof."
    expect_model_failure(
        "verified source strategy citing agent inference",
        failures,
        lambda: AnalysisReport.model_validate(verified_agent),
    )

    verified_runtime = deepcopy(analysis_data)
    verified_runtime["source_strategy_candidates"][0]["status"] = "verified_upstream_fact"
    verified_runtime["evidence_items"][0]["kind"] = "runtime_observation"
    verified_runtime["evidence_items"][0]["claim_status"] = "verified_upstream_fact"
    verified_runtime["evidence_items"][0]["source_file"] = None
    expect_model_failure(
        "verified source strategy citing runtime observation",
        failures,
        lambda: AnalysisReport.model_validate(verified_runtime),
    )

    generated_upstream = deepcopy(analysis_data)
    generated_upstream["evidence_items"].append(
        {
            "evidence_id": "ev-generated-upstream",
            "kind": "source_file",
            "claim_status": "verified_upstream_fact",
            "description": "Generated environment artifact.",
            "source_file": "environments/card/environment.json",
        }
    )
    generated_upstream["source_strategy_candidates"][0]["status"] = "verified_upstream_fact"
    generated_upstream["source_strategy_candidates"][0]["evidence_ids"] = ["ev-generated-upstream"]
    generated_upstream["confidence_summary"]["verified_fact_count"] += 2
    generated_upstream["confidence_summary"]["unresolved_count"] -= 1
    expect_model_failure(
        "generated environment artifact proving verified upstream fact",
        failures,
        lambda: AnalysisReport.model_validate(generated_upstream),
    )

    generic_package = deepcopy(env_data)
    for item in generic_package["evidence_items"]:
        if item["evidence_id"] == "ev-package":
            item["package_name"] = None
            item["package_version"] = None
    expect_model_failure(
        "official package selected from generic package evidence",
        failures,
        lambda: EnvironmentResolutionReport.model_validate(generic_package),
    )

    inferred_package = deepcopy(env_data)
    for item in inferred_package["evidence_items"]:
        if item["evidence_id"] == "ev-package":
            item["kind"] = "agent_inference"
            item["claim_status"] = "reasoned_inference"
            item["source_file"] = None
            item["rationale"] = "Package identity remains inferred."
    expect_model_failure(
        "agent inference proving official package identity",
        failures,
        lambda: EnvironmentResolutionReport.model_validate(inferred_package),
    )

    url_only_package = deepcopy(env_data)
    for item in url_only_package["evidence_items"]:
        if item["evidence_id"] == "ev-package":
            item["claim_status"] = "verified_upstream_fact"
            item["source_file"] = None
            item["url"] = "https://example.invalid/arbitrary-package-metadata"
    expect_model_failure(
        "arbitrary URL-only metadata proving official package identity",
        failures,
        lambda: EnvironmentResolutionReport.model_validate(url_only_package),
    )

    verification_script_package = deepcopy(env_data)
    for item in verification_script_package["evidence_items"]:
        if item["evidence_id"] == "ev-package":
            item["source_file"] = "environments/synthetic/verify_environment.py"
    expect_model_failure(
        "verification script proving official package identity",
        failures,
        lambda: EnvironmentResolutionReport.model_validate(verification_script_package),
    )

    unresolved_gate = deepcopy(env_data)
    unresolved_gate["source_strategy_decision_gates"] = [
        {
            "question_id": "q-arbitrary-source-choice",
            "classification": "needs_user_decision",
            "description": "Choose between source strategies.",
            "alternatives": ["official_package", "pinned_official_git_repository"],
            "evidence_ids": ["ev-package"],
            "default_if_deferred": "do not promote",
            "failure_classification": None,
        }
    ]
    expect_model_failure(
        "arbitrary source-strategy gate blocks promotion",
        failures,
        lambda: EnvironmentResolutionReport.model_validate(unresolved_gate),
    )

    fingerprint = str(env_data["environment_fingerprint"])
    sink = RuntimeReportSink(
        ROOT / ".torch-dae",
        "reports",
        "environments",
        "synthetic",
        fingerprint,
    )
    diagnostic = sink.record_event(operation="verification-script", status="success")
    diagnostic_report = deepcopy(env_data)
    diagnostic_report["verification_report_or_diagnostic_reference"] = diagnostic
    try:
        EnvironmentResolutionReport.model_validate(diagnostic_report)
    except Exception as exc:
        fail(f"actual environment diagnostic reference rejected: {exc}", failures)

    checkpoint_reference = deepcopy(env_data)
    checkpoint_reference["verification_report_or_diagnostic_reference"] = (
        "reports/checkpoints/synthetic/check.json"
    )
    expect_model_failure(
        "checkpoint report reference rejected for environment promotion",
        failures,
        lambda: EnvironmentResolutionReport.model_validate(checkpoint_reference),
    )

    try:
        EvidenceItem.model_validate(
            {
                "evidence_id": "ev-github",
                "kind": "source_file",
                "claim_status": "locally_observed_behavior",
                "description": "Observed CI workflow.",
                "source_file": ".github/workflows/ci.yml",
            }
        )
        DependencyEvidenceRecord.model_validate(
            {
                "normalized_name": "python",
                "raw_declaration": "python==3.11",
                "constraint": "==3.11",
                "exact_version": "3.11",
                "source_file": ".github/workflows/ci.yml",
                "source_section": "matrix.python-version",
                "dependency_kind": "locked",
                "valid": True,
                "evidence_id": "ev-ci-python",
            }
        )
    except Exception as exc:
        fail(f".github evidence path was not accepted: {exc}", failures)

    dependency_evidence = inspect_dependencies(
        ROOT / "tests/skills/fixtures/synthetic_onboarding/unpinned_dependencies",
        budget=InspectionBudget(),
    )
    records = dependency_evidence.get("dependency_records", ())
    if not any(
        record.get("raw_declaration") == "numpy<1.24"
        and record.get("constraint") == "<1.24"
        and record.get("dependency_kind") == "conda"
        for record in records
    ):
        fail("conda numpy range was not parsed as a version constraint", failures)
    if not any(
        record.get("source_file") == ".github/workflows/ci.yml"
        and record.get("source_section") == "matrix.python-version"
        and record.get("raw_declaration") == "python==3.10"
        for record in records
    ):
        fail("CI matrix list dependency was not parsed with preserved .github path", failures)

    hidden = load_json(ROOT / "tests/skills/golden/hidden-checkpoint-helper.analysis.json")
    candidates = hidden.get("checkpoint_candidates", ())
    if not candidates or candidates[0].get("helper_symbol") != "get_pretrained_checkpoint_url":
        fail("hidden checkpoint golden does not record helper symbol", failures)
    hidden_scenario = SkillEvaluationScenario.model_validate_json(
        (ROOT / "tests/skills/scenario_expectations/hidden-checkpoint-helper.json").read_text()
    )
    hidden_observation = inspect_scenario_repository(
        ROOT / "tests/skills/fixtures/synthetic_onboarding/hidden_checkpoint_helper",
        scenario_id="hidden-checkpoint-helper",
    )
    wrong_hash_report = deepcopy(hidden)
    wrong_hash_report["checkpoint_candidates"][0]["hash_evidence"] = "1" * 64
    checkpoint_failures = evaluate_analysis_report(
        hidden_scenario,
        AnalysisReport.model_validate(wrong_hash_report),
        hidden_observation,
    )
    if not any("hash was not associated" in failure for failure in checkpoint_failures):
        fail("checkpoint A accepted checkpoint B hash", failures)

    with TemporaryDirectory() as temporary:
        ci_root = Path(temporary)
        workflow = ci_root / ".github/workflows/ci.yml"
        workflow.parent.mkdir(parents=True)
        workflow.write_text(
            "jobs:\n"
            "  test:\n"
            "    strategy:\n"
            "      matrix:\n"
            '        python-version: ["3.11"]\n'
            "    steps:\n"
            "      - uses: actions/setup-python@v5\n"
            "        with:\n"
            "          python-version: ${{ matrix.python-version }}\n"
        )
        (ci_root / "pyproject.toml").write_text(
            "[project]\nname = 'validator-ci-fixture'\nversion = '0.1.0'\n"
        )
        (ci_root / "environment.yml").write_text("dependencies:\n  - python=>3.9\n")
        ci_result = generate_environment_candidates(ci_root)
        ci_records = ci_result["dependency_records"]
        if any("${{ matrix.python-version }}" in item["raw_declaration"] for item in ci_records):
            fail("GitHub Actions expression became a dependency record", failures)
        if ci_result["candidates"][0]["python_version"] != "3.11":
            fail("invalid dependency record erased valid exact CI Python version", failures)
        if any(item.startswith("python ") for item in ci_result["unresolved_constraints"]):
            fail("invalid dependency record created unresolved Python constraints", failures)
        if "dependency_conflict" in ci_result["candidates"][0]["predicted_failure_risks"]:
            fail("invalid dependency record created dependency conflict risk", failures)

    budget = InspectionBudget()
    inspect_scenario_repository(
        ROOT / "tests/skills/fixtures/synthetic_onboarding/official_package",
        scenario_id="official-package",
        budget=budget,
    )
    if budget.files_visited <= 0 or budget.bytes_read <= 0:
        fail("scenario inspection did not use the shared inspection budget", failures)


def main() -> int:
    failures: list[str] = []

    for relative in REQUIRED:
        if not (ROOT / relative).exists():
            fail(f"missing required path: {relative}", failures)

    worktree_patch = subprocess.run(
        [sys.executable, "scripts/check_worktree_patch.py", "--json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if worktree_patch.returncode != 0:
        try:
            detail = json.loads(worktree_patch.stdout)
        except json.JSONDecodeError:
            detail = worktree_patch.stderr.strip() or worktree_patch.stdout.strip()
        fail(f"staged-equivalent worktree validation failed: {detail}", failures)

    if list(ROOT.glob("**/*backbone*.json")):
        fail("legacy backbone JSON files are present", failures)
    failures.extend(numbered_stage_errors(ROOT))
    failures.extend(root_dependency_errors(ROOT))
    validate_integration_artifacts(ROOT, failures)
    control_plane_drift = validate_onboarding_reports(ROOT, failures)
    if subprocess.run(
        ["git", "ls-files", "._*"], cwd=ROOT, check=False, capture_output=True, text=True
    ).stdout:
        fail("tracked AppleDouble files are present", failures)
    if subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--", ".torch-dae"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    ).stdout:
        fail("runtime artifact is staged under .torch-dae", failures)
    if subprocess.run(
        ["git", "ls-files", ".torch-dae"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    ).stdout:
        fail("runtime artifact is tracked under .torch-dae", failures)
    if not git_ignored(".torch-dae/checkpoints/example/file.bin"):
        fail(".torch-dae/ paths are not ignored", failures)
    if list((ROOT / "environments").glob("*/.venv")):
        fail("committed environment directory contains .venv", failures)

    canonical = (ROOT / "skills/audio-model-onboarding").resolve()
    for relative in [
        ".agents/skills/audio-model-onboarding",
        ".claude/skills/audio-model-onboarding",
    ]:
        if (ROOT / relative).resolve() != canonical:
            fail(f"{relative} does not resolve to canonical skill", failures)

    for schema in (ROOT / "schemas").glob("*.json"):
        try:
            Draft202012Validator.check_schema(load_json(schema))
        except Exception as exc:
            fail(f"invalid schema {schema.name}: {exc}", failures)

    result = subprocess.run(
        [sys.executable, "scripts/generate_schemas.py", "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        fail("schemas are not synchronized with Pydantic generation", failures)

    valid_dir = ROOT / "tests/fixtures/valid"
    invalid_dir = ROOT / "tests/fixtures/invalid"
    manifest = ROOT / "tests/fixtures/invalid_manifest.json"
    classifications = load_json(manifest) if manifest.exists() else {}
    schema_map = {
        "model-card": ROOT / "schemas/model-card.schema.json",
        "environment": ROOT / "schemas/environment.schema.json",
        "environment-sources": ROOT / "schemas/environment-sources.schema.json",
        "environment-materialization-result": ROOT
        / "schemas/environment-materialization-result.schema.json",
        "environment-dependency-closure-result": ROOT
        / "schemas/environment-dependency-closure-result.schema.json",
        "environment-verification-result": ROOT
        / "schemas/environment-verification-result.schema.json",
        "checkpoint": ROOT / "schemas/checkpoint.schema.json",
        "embedding": ROOT / "schemas/embedding.schema.json",
        "verification-report": ROOT / "schemas/verification-report.schema.json",
        "runtime-verification-target": ROOT / "schemas/runtime-verification-target.schema.json",
        "analysis-report": ROOT / "schemas/analysis-report.schema.json",
        "environment-resolution-report": ROOT / "schemas/environment-resolution-report.schema.json",
        "workflow-record": ROOT / "schemas/workflow-record.schema.json",
        "phase-handoff": ROOT / "schemas/phase-handoff.schema.json",
        "cleanup-receipt": ROOT / "schemas/cleanup-receipt.schema.json",
    }
    for path in valid_dir.glob("*.json"):
        kind = path.name.split(".")[0]
        try:
            validate_fixture(path, schema_map[kind])
        except Exception as exc:
            fail(f"valid fixture failed {path.name}: {exc}", failures)
    for path in invalid_dir.glob("*.json"):
        relative = f"tests/fixtures/invalid/{path.name}"
        if classifications.get(relative) == "semantic_cross_reference":
            continue
        kind = path.name.split(".")[0]
        try:
            validate_fixture(path, schema_map[kind])
        except Exception:
            continue
        fail(f"invalid fixture passed: {path.name}", failures)

    if manifest.exists():
        for relative, classification in classifications.items():
            path = ROOT / relative
            kind = path.name.split(".")[0]
            schema = schema_map[kind]
            pydantic_failed = False
            schema_failed = False
            try:
                if kind == "model-card":
                    validate_model_card_path(path)
                else:
                    pydantic_validate(path)
            except Exception:
                pydantic_failed = True
            try:
                schema_validate(path, schema)
            except Exception:
                schema_failed = True
            if classification == "structural" and not (pydantic_failed and schema_failed):
                fail(
                    f"structural invalid fixture did not fail both validators: {relative}",
                    failures,
                )
            if classification == "semantic_cross_reference" and not pydantic_failed:
                if not semantic_invalid_fixture_fails(path):
                    fail(
                        f"semantic invalid fixture passed repository validation: {relative}",
                        failures,
                    )

    registry = ModelCardRegistry(ROOT)
    try:
        registry.list_cards()
    except Exception as exc:
        fail(f"empty production registry failed: {exc}", failures)

    installed = {dist.metadata["Name"].lower() for dist in distributions()}
    installed &= MODEL_DEPS
    if installed:
        fail(
            f"model-specific dependencies importable in root environment: {sorted(installed)}",
            failures,
        )
    for model in (EnvironmentMaterializationRecord, CheckpointMaterializationRecord):
        if not model.model_fields:
            fail(f"runtime metadata model did not import: {model.__name__}", failures)

    skill_root = ROOT / "skills/audio-model-onboarding"
    skill_text = (skill_root / "SKILL.md").read_text()
    required_modes = ("analyze", "resolve-environment", "integrate", "verify", "card", "profile")
    for mode in required_modes:
        if f"## `{mode}` Mode" not in skill_text:
            fail(f"onboarding skill mode is missing: {mode}", failures)
    if (
        "Profiling remains unavailable until a model is runtime_verified and a profiling workflow "
        "is\nexplicitly implemented and invoked." not in skill_text
    ):
        fail("profile mode is not truthfully reserved in the skill", failures)
    stale_license_access_phrases = (
        "license_or_" + "access_blocker",
        "licensing/" + "access constraint",
        "license/" + "access implications",
        "ambiguous license " + "evidence",
    )
    for phrase in stale_license_access_phrases:
        if phrase in skill_text:
            fail(
                f"stale combined {'license/' + 'access'} wording remains in SKILL.md: {phrase}",
                failures,
            )
    required_references = {
        "workflow-overview.md",
        "evidence-policy.md",
        "repository-analysis.md",
        "environment-resolution.md",
        "source-strategy.md",
        "checkpoint-discovery.md",
        "architecture-and-embeddings.md",
        "integration-planning.md",
        "runtime-verification.md",
        "model-card-authoring.md",
        "lifecycle-and-decision-gates.md",
        "failure-classification.md",
        "synthetic-evaluation.md",
    }
    required_templates = {
        "technical-analysis-report.json",
        "technical-analysis-report.md",
        "environment-resolution-report.json",
        "integration-plan.md",
        "verification-plan.md",
        "decision-request.md",
        "model-card-draft.json",
        "agent-request.md",
        "agent-response.md",
    }
    required_scripts = {
        "inspect_repository.py",
        "inspect_python_project.py",
        "inspect_dependencies.py",
        "inspect_checkpoints.py",
        "inspect_model_candidates.py",
        "inspect_output_candidates.py",
        "generate_environment_candidates.py",
        "validate_analysis_report.py",
        "validate_skill_artifacts.py",
        "render_analysis_report.py",
        "extract_pdf_text.py",
        "common.py",
    }
    for name in sorted(required_references):
        path = skill_root / "references" / name
        if not path.exists() or not path.read_text().strip():
            fail(f"required onboarding reference is missing or empty: {name}", failures)
        if f"references/{name}" not in skill_text:
            fail(f"SKILL.md does not link reference: {name}", failures)
    for name in sorted(required_templates):
        if not (skill_root / "templates" / name).exists():
            fail(f"required onboarding template is missing: {name}", failures)
    for name in sorted(required_scripts):
        path = skill_root / "scripts" / name
        if not path.exists():
            fail(f"required onboarding script is missing: {name}", failures)
        elif "import torch" in path.read_text() or "urllib.request.urlopen" in path.read_text():
            fail(
                f"onboarding script imports a model runtime or public network primitive: {name}",
                failures,
            )
    result = subprocess.run(
        [
            sys.executable,
            "skills/audio-model-onboarding/scripts/validate_skill_artifacts.py",
            str(ROOT),
            "--json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        fail("onboarding skill artifact validation script failed", failures)
    onboarding_behavioral_smoke(failures)
    synthetic_root = ROOT / "tests/skills/fixtures/synthetic_onboarding"
    required_scenarios = {
        "official_package": "official_package",
        "pinned_git": "pinned_official_git_repository",
        "minimal_vendoring": "minimal_vendored_adaptation",
        "ambiguous_embeddings": "official_package",
        "unpinned_dependencies": "official_package",
        "hidden_checkpoint_helper": "official_package",
        "non_pytorch_upstream": "external_pytorch_implementation",
        "unsupported": "unsupported_or_non_equivalent_implementation",
    }
    expectation_root = ROOT / "tests/skills/scenario_expectations"
    golden_root = ROOT / "tests/skills/golden"
    for scenario, expected_strategy in sorted(required_scenarios.items()):
        scenario_root = synthetic_root / scenario
        marker = scenario_root / "SCENARIO.json"
        readme = scenario_root / "README.md"
        if not marker.exists():
            fail(f"missing synthetic scenario marker: {scenario}", failures)
            continue
        if not readme.exists() or "synthetic" not in readme.read_text().lower():
            fail(f"synthetic fixture marker missing from README: {scenario}", failures)
        try:
            payload = json.loads(marker.read_text())
        except json.JSONDecodeError as exc:
            fail(f"invalid synthetic scenario JSON {scenario}: {exc}", failures)
            continue
        if payload.get("synthetic") is not True:
            fail(f"synthetic fixture is not explicitly marked synthetic: {scenario}", failures)
        stale_keys = sorted(key for key in payload if key.startswith("expected_"))
        if stale_keys:
            fail(f"synthetic fixture embeds oracle keys {scenario}: {stale_keys}", failures)
        expectation_id = payload.get("scenario_id")
        expectation_path = expectation_root / f"{expectation_id}.json"
        golden_path = golden_root / f"{expectation_id}.analysis.json"
        if not expectation_path.exists():
            fail(f"missing external scenario expectation: {expectation_id}", failures)
            continue
        expectation_payload = json.loads(expectation_path.read_text())
        if expectation_payload.get("expected_source_strategy") != expected_strategy:
            fail(f"external scenario expectation has wrong strategy: {expectation_id}", failures)
        if not golden_path.exists():
            fail(f"missing golden scenario analysis report: {expectation_id}", failures)
        else:
            try:
                scenario_contract = SkillEvaluationScenario.model_validate(expectation_payload)
                analysis_report = AnalysisReport.model_validate_json(golden_path.read_text())
                external_fixture = synthetic_root / "external_pytorch_implementation"
                observation = inspect_scenario_repository(
                    scenario_root,
                    scenario_id=scenario_contract.scenario_id,
                    external_pytorch_root=external_fixture
                    if scenario_contract.scenario_id == "non-pytorch-upstream"
                    else None,
                )
                failures_for_report = evaluate_analysis_report(
                    scenario_contract, analysis_report, observation
                )
                if failures_for_report:
                    fail(
                        f"golden scenario report failed {expectation_id}: {failures_for_report}",
                        failures,
                    )
            except Exception as exc:
                fail(f"golden scenario report is invalid {expectation_id}: {exc}", failures)
    inspection_text = (ROOT / "src/torch_dae/onboarding/inspection.py").read_text()
    if "expected_source_strategy" in inspection_text or "SCENARIO.json" in inspection_text:
        fail("production inspection code reads synthetic oracle fields", failures)
    for relative in (
        ".agents/skills/audio-model-onboarding",
        ".claude/skills/audio-model-onboarding",
    ):
        alias = ROOT / relative
        if not alias.is_symlink():
            fail(f"agent skill path is not a symlink: {relative}", failures)
        elif (alias.resolve() / "SKILL.md").read_bytes() != (canonical / "SKILL.md").read_bytes():
            fail(f"agent skill SKILL.md bytes diverge: {relative}", failures)
    stale_duplicates = [
        path
        for path in ROOT.glob("**/audio-model-onboarding/SKILL.md")
        if "skills/audio-model-onboarding/SKILL.md" not in path.as_posix()
    ]
    if stale_duplicates:
        fail(f"stale duplicate onboarding skills found: {stale_duplicates}", failures)

    env_manager = (ROOT / "src/torch_dae/environment/manager.py").read_text()
    source_manager = (ROOT / "src/torch_dae/environment/sources.py").read_text()
    runtime_module = (ROOT / "src/torch_dae/environment/runtime.py").read_text()
    if "_write_local_wheel" in env_manager or "wheel_record_hash" in env_manager:
        fail("handwritten local wheel implementation remains present", failures)
    if '"uv",\n                "build"' not in env_manager:
        fail("local torch-dae wheel is not built through uv/build backend", failures)
    if "source-builds/torch-dae/current" in source_manager:
        fail("stale hard-coded local wheel cache lookup remains", failures)
    if (
        "recommended.environment_id != model_card_id" in env_manager
        or "must match the requested card ID" in env_manager
    ):
        fail("environment manager still requires environment_id == card_id", failures)
    for required in (
        "recommended.lockfile != specification.lockfile",
        "_validate_environment_artifact_paths",
        "environments/{model_card_id}/uv.lock",
        "environments/{model_card_id}/pyproject.toml",
        "environments/{model_card_id}/sources.json",
        "environments/{model_card_id}/verify_environment.py",
    ):
        if required not in env_manager:
            fail(f"environment path/reference validation is missing: {required}", failures)
    fingerprint_module = (ROOT / "src/torch_dae/environment/fingerprint.py").read_text()
    for required in (
        "readme",
        "local_package_content_digest",
        "local_package_build_inputs",
        'src_root = repository_root / "src" / "torch_dae"',
        "content-sha256:",
        "local_package_provenance",
    ):
        if required not in fingerprint_module:
            fail(f"local package identity coverage is missing: {required}", failures)
    locking_module = (ROOT / "src/torch_dae/environment/locking.py").read_text()
    for required in (
        "ManagedDirectoryLock",
        "ManagedLockTimeoutError",
        "stale_after_seconds",
        "owner_token",
        "os.kill(pid, 0)",
    ):
        if required not in locking_module:
            fail(f"managed cache locking coverage is missing: {required}", failures)
    for required in (
        "local_wheel_cache_key",
        "ManagedDirectoryLock",
        "os.replace(build_dir, wheel_dir)",
        ".build-{os.getpid()}",
        "LOCAL_WHEEL_SOURCE_DATE_EPOCH",
    ):
        if required not in env_manager:
            fail(f"concurrency-safe local wheel caching is missing: {required}", failures)
    if '["git", "clone", source.url, str(checkout)]' in source_manager:
        fail("Git source acquisition still clones directly into the final cache path", failures)
    for required in (
        ".clone-",
        "_ensure_git_checkout",
        "except SourceMaterializationError",
        "OfflineResourceUnavailableError",
        "replace_tree(checkout)",
    ):
        if required not in source_manager:
            fail(f"online Git checkout recovery is missing: {required}", failures)
    if (
        "GitSourceWheelCacheRecord" not in runtime_module
        or "_valid_cached_git_wheel" not in source_manager
    ):
        fail("Git source wheel cache metadata validation is missing", failures)
    compact_env_manager = env_manager.replace(" ", "").replace("\n", "")
    if (
        "env_remove=python_env_remove()" not in compact_env_manager
        or "PYTHONPATH" not in env_manager
    ):
        fail("model-environment subprocess sanitation is missing", failures)
    subprocess_module = (ROOT / "src/torch_dae/environment/subprocess.py").read_text()
    if "RuntimeReportSink" not in runtime_module:
        fail("shared runtime report sink is missing", failures)
    if "with_report_sink" not in subprocess_module:
        fail("CommandExecutor report-sink integration is missing", failures)
    if (
        "command_log_references" not in env_manager
        or '"reports"' not in env_manager
        or '"environments"' not in env_manager
    ):
        fail("environment command diagnostics are not wired into metadata", failures)
    for operation in (
        "python-resolution",
        "uv-venv",
        "uv-sync",
        "local-wheel-build",
        "local-wheel-install",
        "dependency-check",
        "verification-script",
    ):
        if operation not in env_manager:
            fail(f"environment operation report label is missing: {operation}", failures)
    for operation in (
        "git-clone",
        "git-checkout",
        "git-revision-check",
        "git-remote-check",
        "git-cleanliness-check",
        "git-archive",
        "git-wheel-build",
        "git-wheel-install",
    ):
        if operation not in source_manager:
            fail(f"source operation report label is missing: {operation}", failures)
    try:
        sink = RuntimeReportSink(ROOT / ".torch-dae", "reports", "validation-smoke")
        ref = sink.record_event(
            operation="validation",
            status="failed",
            arguments=("https://user:secret@example.invalid/file?token=secret",),
            stderr="Authorization: Bearer secret",
        )
        payload = json.loads((ROOT / ".torch-dae" / ref).read_text())
        serialized = json.dumps(payload)
        if (
            payload["status"] != "failed"
            or "user:secret" in serialized
            or "Bearer secret" in serialized
            or "token=secret" in serialized
        ):
            fail("runtime report sink did not redact or persist expected fields", failures)
    except Exception as exc:
        fail(f"runtime report sink behavioral smoke failed: {exc}", failures)
    if not git_ignored(".torch-dae/reports/environments/card/hash/log.json"):
        fail(".torch-dae diagnostic reports are not ignored", failures)
    checkpoint_module = (ROOT / "src/torch_dae/core/checkpoint.py").read_text()
    for required in (
        "reports",
        "checkpoints",
        "remote-open",
        "remote-stream",
        "remote-finalize",
        "hash-validation",
        "offline-cache-lookup",
        "local-path-copy",
        "package-bundle-lookup",
        "package-bundle-copy",
        "cache-finalize",
        "metadata-write",
        "response-close",
        "failure-cleanup",
        "command_log_references",
    ):
        if required not in checkpoint_module:
            fail(f"checkpoint diagnostic implementation is missing: {required}", failures)
    for required in (
        "checkpoint download stream failed",
        "checkpoint_failure_classification",
        "expected_hash_mismatch",
        "observed_hash_mismatch",
        "offline_cache_miss",
        "checkpoint metadata write failed",
        "checkpoint copy failed",
        "checkpoint cache finalization failed",
        "checkpoint response close failed",
        "HTTPError",
        "body.close()",
    ):
        if required not in checkpoint_module:
            fail(f"checkpoint failure normalization is missing: {required}", failures)
    report: dict[str, Any] = {
        "ok": not failures,
        "failures": failures,
        "historical_control_plane_drift": control_plane_drift,
    }
    report_dir = ROOT / ".torch-dae/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "repository-validation.json").write_text(json.dumps(report, indent=2))

    if failures:
        for item in failures:
            print(f"FAIL: {item}")
        return 1
    if control_plane_drift:
        print(
            "Repository validation passed; historical control-plane drift reported for: "
            + ", ".join(str(item["workflow_id"]) for item in control_plane_drift)
        )
    else:
        print("Repository validation passed; no historical control-plane drift")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
