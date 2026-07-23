"""Repository validation for Phase 00."""

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
from torch_dae.core.checkpoint import CheckpointSpec
from torch_dae.core.embeddings import EmbeddingSpec
from torch_dae.core.registry import ModelCardRegistry
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
    if not git_ignored(".torch-dae/checkpoints/example/file.bin"):
        fail(".torch-dae/ paths are not ignored", failures)

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

    report: dict[str, Any] = {"ok": not failures, "failures": failures}
    report_dir = ROOT / ".torch-dae/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "phase00-validation.json").write_text(json.dumps(report, indent=2))

    if failures:
        for item in failures:
            print(f"FAIL: {item}")
        return 1
    print("Phase 00 repository validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
