"""Repository validation for Phase 01 repository safety."""

from __future__ import annotations

import json
import subprocess
import sys
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from pydantic import BaseModel

from torch_dae.cards.validation import load_json, validate_model_card_path
from torch_dae.core.checkpoint import CheckpointMaterializationRecord, CheckpointSpec
from torch_dae.core.embeddings import EmbeddingSpec
from torch_dae.core.registry import ModelCardRegistry
from torch_dae.environment.runtime import EnvironmentMaterializationRecord, RuntimeReportSink
from torch_dae.environment.specification import EnvironmentSourcesManifest, EnvironmentSpecification
from torch_dae.environment.verification import VerificationReport

ROOT = Path(__file__).resolve().parents[1]
MODEL_DEPS = {"torch", "torchaudio", "torchvision", "transformers", "tensorflow", "jax", "librosa"}
REQUIRED = [
    "project_spec.md",
    "pyproject.toml",
    "uv.lock",
    "skills/audio-model-onboarding/SKILL.md",
    ".agents/skills/audio-model-onboarding",
    ".claude/skills/audio-model-onboarding",
    "schemas/model-card.schema.json",
    "schemas/checkpoint.schema.json",
    "schemas/environment.schema.json",
    "schemas/environment-sources.schema.json",
    "schemas/embedding.schema.json",
    "schemas/verification-report.schema.json",
    "src/torch_dae",
    "tests/fixtures",
]


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)


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
        "verification-report": VerificationReport,
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


def main() -> int:
    failures: list[str] = []

    if ROOT.name != "torch-dae":
        fail("repository root basename is not torch-dae", failures)
    for relative in REQUIRED:
        if not (ROOT / relative).exists():
            fail(f"missing required path: {relative}", failures)

    if list(ROOT.glob("**/*backbone*.json")):
        fail("legacy backbone JSON files are present", failures)
    if list((ROOT / "model_cards").glob("**/*.json")):
        fail("real model cards are present during Phase 00", failures)
    if (ROOT / "src/torch_dae/models").exists():
        fail("pilot model modules exist during Phase 00", failures)
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
        "checkpoint": ROOT / "schemas/checkpoint.schema.json",
        "embedding": ROOT / "schemas/embedding.schema.json",
        "verification-report": ROOT / "schemas/verification-report.schema.json",
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

    env_cli = (ROOT / "src/torch_dae/cli/environment.py").read_text()
    checkpoint_cli = (ROOT / "src/torch_dae/cli/checkpoints.py").read_text()
    model_cli = (ROOT / "src/torch_dae/cli/models.py").read_text()
    env_manager = (ROOT / "src/torch_dae/environment/manager.py").read_text()
    source_manager = (ROOT / "src/torch_dae/environment/sources.py").read_text()
    runtime_module = (ROOT / "src/torch_dae/environment/runtime.py").read_text()
    if "belongs to Phase 01" in env_cli or "belongs to Phase 01" in checkpoint_cli:
        fail("environment or checkpoint CLI remains deferred to Phase 01", failures)
    if "belongs to Phase 03+" not in model_cli:
        fail("model CLI no longer truthfully defers model integration", failures)
    if "_write_local_wheel" in env_manager or "wheel_record_hash" in env_manager:
        fail("handwritten local wheel implementation remains present", failures)
    if '"uv",\n                    "build"' not in env_manager:
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
        "git:{head.stdout.strip()}:content:",
    ):
        if required not in fingerprint_module:
            fail(f"local package identity coverage is missing: {required}", failures)
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
    phase01_readme = ROOT / "tests/fixtures/phase01/README.md"
    if not phase01_readme.exists() or "synthetic" not in phase01_readme.read_text().lower():
        fail("Phase 01 fixture area is missing a synthetic marker", failures)
    checkpoint_tests = (ROOT / "tests/core/test_phase01_checkpoint.py").read_text()
    source_tests = (ROOT / "tests/environment/test_phase01_sources.py").read_text()
    materialization_tests = (ROOT / "tests/environment/test_phase01_materialization.py").read_text()
    if "example.invalid" not in checkpoint_tests:
        fail("checkpoint tests do not document fake public-network endpoints", failures)
    if (
        "test_phase01_package_bundle_checkpoint_uses_owned_distribution_file_offline"
        not in checkpoint_tests
    ):
        fail("Phase 01 package-bundle checkpoint integration coverage is missing", failures)
    if "test_phase01_git_source_build_metadata_workspace_and_offline_reuse" not in source_tests:
        fail("Phase 01 Git source cache/reuse coverage is missing", failures)
    if (
        "@pytest.mark.integration\n"
        "def test_phase01_git_source_build_metadata_workspace_and_offline_reuse" in source_tests
    ):
        fail("simulated Git source state-machine test is incorrectly marked integration", failures)
    if (
        "test_phase01_git_source_real_local_repository_installs_and_reuses_offline"
        not in source_tests
    ):
        fail("real local-Git source integration coverage is missing", failures)
    real_git_body = source_tests.split(
        "def test_phase01_git_source_real_local_repository_installs_and_reuses_offline",
        1,
    )[1].split("def run_git_command", 1)[0]
    if "GitSourceRunner" in real_git_body:
        fail("real local-Git integration test still uses the fake Git runner", failures)
    for required in (
        "test_phase01_git_source_online_recovers_invalid_checkout",
        "test_phase01_git_source_offline_invalid_checkout_fails_without_mutation",
        "dirty-offline",
        'remote", "set-url"',
        "other_revision",
        'metadata.write_text("{bad json")',
    ):
        if required not in source_tests:
            fail(f"Git invalid-cache recovery coverage is missing: {required}", failures)
    for required in (
        "phase01-synthetic-shared-environment",
        "test_phase01_environment_load_rejects_cross_document_path_mismatch",
        "test_phase01_failed_uv_sync_metadata_references_reports",
        "test_phase01_failed_git_clone_metadata_references_reports",
        "test_phase01_failed_git_wheel_build_metadata_references_reports",
        "failed_materialization_metadata",
        "command_log_references",
        "remove-wheel-json",
        "malformed-wheel-json",
    ):
        if required not in materialization_tests:
            fail(f"environment identity/cache regression coverage is missing: {required}", failures)
    environment_tests = (ROOT / "tests/environment/test_environment.py").read_text()
    for required in ("README.md", "package-data.json", "vendor/source.txt", "new_module.py"):
        if required not in environment_tests:
            fail(f"local package identity input coverage is missing: {required}", failures)
    if "test_phase01_local_wheel_backend_build_is_reproducible" not in materialization_tests:
        fail("Phase 01 reproducible local wheel coverage is missing", failures)
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
    for required in (
        "test_phase01_package_bundle_rejects_file_owned_by_other_distribution",
        "test_phase01_remote_response_is_closed_for_success_and_http_failure",
        "test_phase01_remote_response_is_closed_for_hash_mismatch",
        "test_phase01_interrupted_remote_read_cleans_partial_state_and_reports_failure",
        "test_phase01_transport_open_oserror_is_typed_reported_and_redacted",
        "test_phase01_urllib_transport_urlerror_is_typed_and_sanitized",
        "test_phase01_urllib_transport_httperror_closes_body",
        "test_phase01_local_copy_failure_is_typed_reported_and_cleans_tmp",
        "test_phase01_cache_finalize_failure_is_typed_reported_and_cleans_tmp",
        "test_phase01_metadata_write_failure_removes_incomplete_cache_entry_and_reports",
        "test_phase01_package_bundle_malformed_lookup_is_typed_and_reported",
        "test_phase01_successful_response_close_failure_is_reported_and_typed",
        "test_phase01_failed_response_close_does_not_mask_stream_error",
        "closed_observed",
        "hash-validation",
        "offline-cache-lookup",
        "metadata-write",
        "response-close",
        "failure-cleanup",
        "secret-token",
        "redacted secret",
    ):
        if required not in checkpoint_tests:
            fail(f"checkpoint evidence regression coverage is missing: {required}", failures)
    if "with pytest.raises(OSError)" in checkpoint_tests:
        fail("interrupted checkpoint stream coverage still expects raw OSError", failures)
    if (
        "test_phase01_checkpoint_cli_expected_acquisition_failures_are_concise"
        not in (ROOT / "tests/cli/test_phase01_cli.py").read_text()
    ):
        fail("real checkpoint CLI operational-failure coverage is missing", failures)

    report: dict[str, Any] = {"ok": not failures, "failures": failures}
    report_dir = ROOT / ".torch-dae/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "phase01-validation.json").write_text(json.dumps(report, indent=2))

    if failures:
        for item in failures:
            print(f"FAIL: {item}")
        return 1
    print("Phase 01 repository validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
