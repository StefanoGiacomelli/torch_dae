"""Deterministic static inspection helpers for Phase 02 synthetic onboarding."""

from __future__ import annotations

import ast
import configparser
import json
import os
import re
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from torch_dae.contracts import contained_path
from torch_dae.onboarding.contracts import (
    CandidateTrialStatus,
    ClaimStatus,
    DependencyEvidenceRecord,
    DependencyKind,
    EnvironmentCandidate,
    EnvironmentCandidateGenerationResult,
    EvidenceItem,
    EvidenceItemKind,
    FailureClassification,
    OpenQuestion,
    OpenQuestionClassification,
    ScenarioInspectionResult,
    SourceStrategy,
    SourceStrategyCandidate,
    ensure_onboarding_evidence_path,
)

IGNORED_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".torch-dae",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}
CHECKPOINT_SUFFIXES = {".ckpt", ".h5", ".onnx", ".params", ".pt", ".pth", ".safetensors"}
ARCHIVE_SUFFIXES = {".gz", ".tar", ".tgz", ".zip"}
MODEL_RUNTIME_IMPORTS = {"torch", "torchaudio", "torchvision", "transformers", "tensorflow", "jax"}
REMOVED_API_PATTERNS = {"np.float": "numpy_compatibility", "np.int": "numpy_compatibility"}
DEFAULT_MAX_FILE_SIZE_BYTES = 512_000
DEFAULT_MAX_TOTAL_FILES = 10_000
DEFAULT_MAX_TOTAL_BYTES = 5_000_000
TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".lock",
    ".md",
    ".py",
    ".rst",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SUPPORTED_SYMLINK_ARTIFACTS = {
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "environment.yml",
    "environment.yaml",
    "Pipfile",
    "Pipfile.lock",
    "poetry.lock",
    "uv.lock",
    "Dockerfile",
}


class OnboardingInspectionError(ValueError):
    """Expected static-inspection failure with a user-safe message."""


@dataclass(frozen=True)
class StaticFile:
    """Safe repository-relative file entry."""

    path: str
    size_bytes: int
    kind: str
    skipped_reason: str | None = None


@dataclass
class InspectionBudget:
    """Per-operation inspection budget and file-content cache."""

    maximum_total_files: int = DEFAULT_MAX_TOTAL_FILES
    maximum_total_inspected_bytes: int = DEFAULT_MAX_TOTAL_BYTES
    maximum_single_file_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES
    files_visited: int = 0
    bytes_read: int = 0
    _visited_paths: set[str] = field(default_factory=set)
    _charged_paths: set[str] = field(default_factory=set)
    _text_cache: dict[str, str] = field(default_factory=dict)

    def visit_file(self, key: str, size_bytes: int) -> None:
        if key not in self._visited_paths:
            if self.files_visited + 1 > self.maximum_total_files:
                raise OnboardingInspectionError("inspection limit exceeded: maximum total files")
            self._visited_paths.add(key)
            self.files_visited += 1
        if size_bytes > self.maximum_single_file_bytes:
            raise OnboardingInspectionError(f"file exceeds inspection size limit: {key}")
        if key not in self._charged_paths:
            if self.bytes_read + size_bytes > self.maximum_total_inspected_bytes:
                raise OnboardingInspectionError(
                    "inspection limit exceeded: maximum total inspected bytes"
                )
            self._charged_paths.add(key)
            self.bytes_read += size_bytes

    def cached_text(self, key: str) -> str | None:
        return self._text_cache.get(key)

    def store_text(self, key: str, text: str) -> None:
        self._text_cache[key] = text


def repository_root(path: Path) -> Path:
    """Resolve and validate a static-inspection root."""

    root = path.resolve()
    if not root.exists() or not root.is_dir():
        raise OnboardingInspectionError(f"repository root does not exist: {path}")
    return root


def safe_child(root: Path, relative: str) -> Path:
    """Return a contained child path."""

    try:
        return contained_path(root, relative)
    except ValueError as exc:
        raise OnboardingInspectionError(str(exc)) from exc


def _is_supported_inspection_artifact(relative: str) -> bool:
    path = Path(relative)
    name = path.name
    return bool(
        name in SUPPORTED_SYMLINK_ARTIFACTS
        or (name.startswith("requirements") and path.suffix == ".txt")
        or path.suffix == ".py"
        or (path.suffix in {".yml", ".yaml"} and ".github/workflows" in relative)
        or path.suffix in CHECKPOINT_SUFFIXES
    )


def iter_static_files(
    root: Path,
    *,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    max_total_files: int = DEFAULT_MAX_TOTAL_FILES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    budget: InspectionBudget | None = None,
) -> tuple[StaticFile, ...]:
    """Inventory files without following symlinks outside the inspected root."""

    root = repository_root(root)
    budget = budget or InspectionBudget(
        maximum_total_files=max_total_files,
        maximum_total_inspected_bytes=max_total_bytes,
        maximum_single_file_bytes=max_file_size_bytes,
    )
    entries: list[StaticFile] = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if any(part in IGNORED_DIRS for part in rel.split("/")):
            continue
        if path.is_symlink():
            if _is_supported_inspection_artifact(rel):
                raise OnboardingInspectionError(
                    f"refusing to inspect symlinked supported artifact: {rel}"
                )
            target = path.resolve()
            try:
                target.relative_to(root)
            except ValueError:
                entries.append(StaticFile(rel, 0, "external_symlink"))
                continue
            entries.append(StaticFile(rel, 0, "symlink"))
            continue
        if path.is_dir():
            continue
        try:
            stat = path.stat()
        except (OSError, PermissionError) as exc:
            raise OnboardingInspectionError(f"failed to stat {rel}: {exc}") from exc
        ensure_onboarding_evidence_path(rel)
        suffixes = set(path.suffixes)
        if path.suffix in CHECKPOINT_SUFFIXES:
            kind = "checkpoint_like"
        elif suffixes & ARCHIVE_SUFFIXES:
            kind = "archive"
        elif path.name.lower().startswith("license") or path.name.lower() in {"copying", "notice"}:
            kind = "license"
        elif path.name in {"pyproject.toml", "setup.py", "setup.cfg"}:
            kind = "packaging"
        elif path.name.startswith("requirements") and path.suffix == ".txt":
            kind = "requirements"
        elif path.suffix in {".yml", ".yaml"} and (
            "environment" in path.name or ".github/workflows" in rel
        ):
            kind = "environment_or_ci"
        elif path.suffix == ".py":
            kind = "python_source"
        elif path.suffix.lower() in {".md", ".rst", ".txt"}:
            kind = "documentation"
        else:
            kind = "file"
        budget.visit_file(_budget_key(root, path), stat.st_size)
        entries.append(StaticFile(rel, stat.st_size, kind))
    return tuple(entries)


def inspect_repository(root: Path, *, budget: InspectionBudget | None = None) -> dict[str, Any]:
    """Return a deterministic repository inventory."""

    budget = budget or InspectionBudget()
    files = iter_static_files(root, budget=budget)
    directories = sorted(
        {Path(item.path).parent.as_posix() for item in files if str(Path(item.path).parent) != "."}
    )
    return {
        "root": ".",
        "synthetic_marker": _synthetic_marker(root, budget),
        "files": [item.__dict__ for item in files],
        "skipped_files": [item.__dict__ for item in files if item.skipped_reason is not None],
        "directories": directories,
        "documentation": [item.path for item in files if item.kind == "documentation"],
        "licenses": [item.path for item in files if item.kind == "license"],
        "requirement_files": [item.path for item in files if item.kind == "requirements"],
        "environment_files": [item.path for item in files if item.kind == "environment_or_ci"],
        "checkpoint_like_files": [item.path for item in files if item.kind == "checkpoint_like"],
        "archives": [item.path for item in files if item.kind == "archive"],
        "test_directories": [item for item in directories if "test" in item],
    }


def inspect_python_project(root: Path, *, budget: InspectionBudget | None = None) -> dict[str, Any]:
    """Extract packaging metadata without executing project code."""

    root = repository_root(root)
    budget = budget or InspectionBudget()
    pyproject = _read_pyproject(root / "pyproject.toml", root, budget)
    setup_py = _inspect_setup_py(root / "setup.py", root, budget)
    setup_cfg = _read_setup_cfg(root / "setup.cfg", root, budget)
    inventory = iter_static_files(root, budget=budget)
    requirements = {
        item.path: _parse_requirements(safe_child(root, item.path), item.path, root, budget)
        for item in inventory
        if item.kind == "requirements"
    }
    environment_files = {
        item.path: _parse_environment_file(safe_child(root, item.path), root, budget)
        for item in inventory
        if item.kind == "environment_or_ci"
    }
    pipfile = _read_toml(root / "Pipfile", root, budget)
    pipfile_lock = _read_json(root / "Pipfile.lock", root, budget)
    poetry_lock = _read_poetry_lock(root / "poetry.lock", root, budget)
    uv_lock = _read_uv_lock(root / "uv.lock", root, budget)
    return {
        "pyproject": pyproject,
        "setup_py": setup_py,
        "setup_cfg": setup_cfg,
        "requirements": requirements,
        "environment_files": environment_files,
        "pipfile": pipfile,
        "pipfile_lock": pipfile_lock,
        "poetry_lock": poetry_lock,
        "uv_lock": uv_lock,
        "source_roots": _source_roots(root),
        "source_strategy_assessment": classify_source_strategy(root, budget=budget),
    }


def inspect_dependencies(root: Path, *, budget: InspectionBudget | None = None) -> dict[str, Any]:
    """Inspect dependency declarations and AST imports."""

    root = repository_root(root)
    budget = budget or InspectionBudget()
    project = inspect_python_project(root, budget=budget)
    imports = inspect_imports(root, budget=budget)
    records: list[DependencyEvidenceRecord] = []
    for raw in project["pyproject"].get("dependencies", []):
        records.append(_dependency_record(raw, "pyproject.toml", "project.dependencies"))
    for section, values in sorted(project["pyproject"].get("optional_dependencies", {}).items()):
        for raw in values:
            records.append(
                _dependency_record(
                    raw, "pyproject.toml", f"project.optional-dependencies.{section}"
                )
            )
    for section, values in sorted(project["pyproject"].get("dependency_groups", {}).items()):
        for raw in values:
            records.append(
                _dependency_record(raw, "pyproject.toml", f"dependency-groups.{section}")
            )
    setup_values = project["setup_py"].get("values", {})
    for raw in setup_values.get("install_requires", ()) or ():
        records.append(_dependency_record(raw, "setup.py", "setup.install_requires"))
    if setup_values.get("python_requires"):
        records.append(
            _dependency_record(
                f"python{setup_values['python_requires']}",
                "setup.py",
                "setup.python_requires",
            )
        )
    for raw in project["setup_cfg"].get("install_requires", ()):
        records.append(_dependency_record(raw, "setup.cfg", "options.install_requires"))
    if project["setup_cfg"].get("python_requires"):
        records.append(
            _dependency_record(
                f"python{project['setup_cfg']['python_requires']}",
                "setup.cfg",
                "options.python_requires",
            )
        )
    for section in ("dependencies", "dev-dependencies", "packages"):
        values = project.get("pipfile", {}).get(section, {})
        if isinstance(values, dict):
            for name, specifier in sorted(values.items()):
                raw = f"{name}{specifier if isinstance(specifier, str) else ''}"
                records.append(_dependency_record(raw, "Pipfile", section))
    for section in ("default", "develop"):
        for package_name, package in sorted(
            project.get("pipfile_lock", {}).get(section, {}).items()
        ):
            version = package.get("version") if isinstance(package, dict) else None
            records.append(
                _dependency_record(f"{package_name}{version or ''}", "Pipfile.lock", section)
            )
    for requirements in project["requirements"].values():
        for item in requirements:
            records.append(
                _dependency_record(
                    item["raw"],
                    item["source_file"],
                    item["source_section"],
                    dependency_kind=item["dependency_kind"],
                    editable=item["editable"],
                    direct_url=item["direct_url"],
                    vcs=item["vcs"],
                    local_path=item["local_path"],
                    valid=item["valid"],
                )
            )
    for source_file, environment_file in project["environment_files"].items():
        for item in environment_file.get("dependencies", ()):
            records.append(
                _dependency_record(
                    item, source_file, "dependencies", dependency_kind=DependencyKind.CONDA
                )
            )
    for raw in project["poetry_lock"].get("dependencies", ()):
        records.append(
            _dependency_record(raw, "poetry.lock", "package", dependency_kind=DependencyKind.LOCKED)
        )
    for raw in project["uv_lock"].get("dependencies", ()):
        records.append(
            _dependency_record(raw, "uv.lock", "package", dependency_kind=DependencyKind.LOCKED)
        )
    for item in iter_static_files(root, budget=budget):
        if item.skipped_reason:
            continue
        if item.path == "Dockerfile" or item.path.endswith("/Dockerfile"):
            records.extend(_parse_dockerfile_dependencies(root, item.path, budget))
        elif ".github/workflows/" in item.path:
            records.extend(_parse_ci_dependencies(root, item.path, budget))

    valid_records = [record for record in records if record.valid]
    declared = [record.raw_declaration for record in valid_records]
    unpinned = [
        record.normalized_name or record.raw_declaration
        for record in valid_records
        if record.normalized_name
        and record.exact_version is None
        and record.constraint != "unconstrained"
    ]
    api_risks = []
    for file_result in imports["files"]:
        text = _read_text_file(safe_child(root, file_result["path"]), root, budget)
        for token, classification in REMOVED_API_PATTERNS.items():
            if token in text:
                api_risks.append(
                    {
                        "path": file_result["path"],
                        "api": token,
                        "failure_classification": classification,
                    }
                )
    return {
        "python_constraint": project["pyproject"].get("requires_python"),
        "declared_dependencies": sorted(declared),
        "dependency_records": [
            record.model_dump(mode="json") for record in sorted(records, key=_record_sort_key)
        ],
        "unpinned_dependencies": sorted(set(unpinned)),
        "imports": imports,
        "api_risks": sorted(api_risks, key=lambda item: (item["path"], item["api"])),
        "runtime_frameworks": sorted(imports["frameworks"]),
    }


def inspect_imports(root: Path, *, budget: InspectionBudget | None = None) -> dict[str, Any]:
    """Parse Python sources for imports and framework API references."""

    file_results: list[dict[str, Any]] = []
    all_imports: set[str] = set()
    frameworks: set[str] = set()
    root = repository_root(root)
    budget = budget or InspectionBudget()
    for item in iter_static_files(root, budget=budget):
        if item.kind != "python_source":
            continue
        if item.skipped_reason:
            continue
        path = safe_child(root, item.path)
        tree = _parse_ast(path, root, budget)
        imports: set[str] = set()
        api_refs: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
            elif isinstance(node, ast.Attribute):
                api_refs.add(_attribute_name(node))
        imports.discard("")
        all_imports.update(imports)
        frameworks.update(imports & MODEL_RUNTIME_IMPORTS)
        file_results.append(
            {
                "path": item.path,
                "imports": sorted(imports),
                "framework_api_references": sorted(ref for ref in api_refs if ref),
            }
        )
    return {
        "packages": sorted(all_imports),
        "frameworks": sorted(frameworks),
        "files": file_results,
    }


def inspect_model_candidates(
    root: Path, *, budget: InspectionBudget | None = None
) -> dict[str, Any]:
    """Identify static model, factory, loader, and preprocessing candidates."""

    candidates: list[dict[str, Any]] = []
    root = repository_root(root)
    budget = budget or InspectionBudget()
    for item in iter_static_files(root, budget=budget):
        if item.kind != "python_source":
            continue
        if item.skipped_reason:
            continue
        tree = _parse_ast(safe_child(root, item.path), root, budget)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = [_attribute_name(base) for base in node.bases]
                if any(base.endswith("nn.Module") or base == "Module" for base in bases):
                    candidates.append(
                        {
                            "kind": "torch_nn_module_subclass",
                            "path": item.path,
                            "symbol": node.name,
                            "bases": bases,
                        }
                    )
            elif isinstance(node, ast.FunctionDef):
                lower = node.name.lower()
                if any(token in lower for token in ("model", "checkpoint", "preprocess", "load")):
                    candidates.append(
                        {
                            "kind": "function_candidate",
                            "path": item.path,
                            "symbol": node.name,
                            "arguments": [arg.arg for arg in node.args.args],
                        }
                    )
    return {"candidates": sorted(candidates, key=lambda item: (item["path"], item["symbol"]))}


def inspect_output_candidates(
    root: Path, *, budget: InspectionBudget | None = None
) -> dict[str, Any]:
    """Identify forward returns and potential tensor names without validating semantics."""

    outputs: list[dict[str, Any]] = []
    embedding_words = ("embedding", "feature", "latent", "pooled", "logit", "classifier")
    root = repository_root(root)
    budget = budget or InspectionBudget()
    for item in iter_static_files(root, budget=budget):
        if item.kind != "python_source":
            continue
        if item.skipped_reason:
            continue
        tree = _parse_ast(safe_child(root, item.path), root, budget)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in {"forward", "__call__"}:
                returns: list[str] = []
                for child in ast.walk(node):
                    if isinstance(child, ast.Return):
                        returns.extend(_return_keys(child.value))
                outputs.append(
                    {
                        "kind": "forward_return",
                        "path": item.path,
                        "symbol": node.name,
                        "candidate_keys": sorted(set(returns)),
                    }
                )
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and any(
                        word in target.id for word in embedding_words
                    ):
                        outputs.append(
                            {
                                "kind": "tensor_name_candidate",
                                "path": item.path,
                                "symbol": target.id,
                                "semantic_status": "candidate_only",
                            }
                        )
    return {
        "candidates": sorted(outputs, key=lambda item: (item["path"], item["kind"], item["symbol"]))
    }


def inspect_checkpoints(root: Path, *, budget: InspectionBudget | None = None) -> dict[str, Any]:
    """Identify checkpoint-like paths, URLs, hashes, and helper functions."""

    candidates: list[dict[str, Any]] = []
    root = repository_root(root)
    budget = budget or InspectionBudget()
    for item in iter_static_files(root, budget=budget):
        path = safe_child(root, item.path)
        if item.skipped_reason:
            continue
        if item.kind == "checkpoint_like":
            candidates.append({"kind": "local_checkpoint_like_file", "path": item.path})
        if item.kind not in {"python_source", "documentation", "file"}:
            continue
        if not _looks_textual(path):
            continue
        text = _read_text_file(path, root, budget)
        for url in sorted(set(re.findall(r"https?://[^\s'\"<>]+", text))):
            if any(suffix in url for suffix in CHECKPOINT_SUFFIXES):
                candidates.append(
                    {
                        "kind": "checkpoint_url",
                        "path": item.path,
                        "source_file": item.path,
                        "url": url,
                        "complete_url": url,
                        "filename": Path(url.split("?", 1)[0]).name,
                        "helper_symbol": None,
                        "associated_hashes": [],
                        "hashes": [],
                        "expression_status": "resolved",
                        "unresolved_components": ["hash association"],
                    }
                )
        for sha in sorted(set(re.findall(r"\b[0-9a-f]{64}\b", text))):
            candidates.append({"kind": "sha256", "path": item.path, "sha256": sha})
        if item.kind == "python_source":
            candidates.extend(_checkpoint_helpers(item.path, text))
    return {"candidates": sorted(candidates, key=lambda item: json.dumps(item, sort_keys=True))}


def generate_environment_candidates(
    root: Path,
    *,
    target_platform: str | None = None,
    external_pytorch_root: Path | None = None,
    budget: InspectionBudget | None = None,
) -> dict[str, Any]:
    """Convert dependency evidence into ordered, unverified compatibility candidates."""

    root = repository_root(root)
    budget = budget or InspectionBudget()
    deps = inspect_dependencies(root, budget=budget)
    source_assessment = classify_source_strategy(
        root, external_pytorch_root=external_pytorch_root, budget=budget
    )
    source_context = tuple(_source_strategy_contracts(source_assessment))
    candidates: list[EnvironmentCandidate] = []
    python_constraint = deps.get("python_constraint") or "unresolved"
    records = [
        DependencyEvidenceRecord.model_validate(item) for item in deps.get("dependency_records", ())
    ]
    valid_records = [record for record in records if record.valid]
    evidence_items = tuple(
        sorted(
            {
                item.evidence_id: item
                for item in (
                    [_evidence_item_for_dependency(record) for record in records]
                    + _source_strategy_evidence_items(source_assessment)
                )
            }.values(),
            key=lambda item: item.evidence_id,
        )
    )
    dependency_evidence_ids = tuple(
        record.evidence_id for record in sorted(valid_records, key=_record_sort_key)
    )
    invalid_dependency_evidence_ids = {record.evidence_id for record in records if not record.valid}
    fallback_evidence_ids = tuple(
        item.evidence_id
        for item in evidence_items
        if item.evidence_id not in invalid_dependency_evidence_ids
    )
    by_name = _principal_dependency_versions(valid_records)
    dependency_conflicts = _dependency_conflicts(valid_records)
    python_version = (
        by_name["python"]["version"]
        if "python" in by_name
        else _select_python_version(str(python_constraint), valid_records)
    )
    risks: list[FailureClassification] = []
    if deps["unpinned_dependencies"] or dependency_conflicts:
        risks.append(FailureClassification.DEPENDENCY_CONFLICT)
    for api_risk in deps["api_risks"]:
        risks.append(FailureClassification(api_risk["failure_classification"]))
    if _incompatible_torch_audio(by_name):
        risks.append(FailureClassification.TORCH_TORCHAUDIO_MISMATCH)
    viable_source_context = [
        item
        for item in source_context
        if item.strategy != SourceStrategy.UNSUPPORTED_OR_NON_EQUIVALENT_IMPLEMENTATION
    ]
    observed_revision = source_assessment.get("observed_repository_revision") or {}
    decision_gates: tuple[OpenQuestion, ...] = ()
    if len(viable_source_context) > 1:
        decision_gates = (
            OpenQuestion(
                question_id="q-source-strategy",
                classification=OpenQuestionClassification.NEEDS_USER_DECISION,
                description=(
                    "Select one unresolved source strategy before environment materialization."
                ),
                alternatives=tuple(item.strategy.value for item in viable_source_context),
                evidence_ids=tuple(
                    sorted(
                        {
                            evidence_id
                            for item in viable_source_context
                            for evidence_id in item.evidence_ids
                        }
                    )
                ),
                default_if_deferred="do not materialize an environment",
            ),
        )
    for index, source_candidate in enumerate(viable_source_context or source_context, start=1):
        candidates.append(
            EnvironmentCandidate(
                candidate_id=f"evidence-ranked-{index}",
                reason_for_selection=(
                    "Derived from declared Python constraints, dependency files, static imports, "
                    "and source strategy evidence; not a verified environment."
                ),
                python_constraint=None
                if python_constraint == "unresolved"
                else str(python_constraint),
                python_version=python_version,
                pytorch_constraint=by_name.get("torch", {}).get("constraint"),
                pytorch_version=by_name.get("torch", {}).get("version"),
                torchaudio_constraint=by_name.get("torchaudio", {}).get("constraint"),
                torchaudio_version=by_name.get("torchaudio", {}).get("version"),
                numpy_constraint=by_name.get("numpy", {}).get("constraint"),
                numpy_version=by_name.get("numpy", {}).get("version"),
                other_principal_dependencies={
                    record.normalized_name: record.raw_declaration
                    for record in records
                    if record.valid
                    and record.normalized_name
                    and record.normalized_name not in {"python", "torch", "torchaudio", "numpy"}
                },
                source_revision=(
                    observed_revision.get("revision")
                    if source_candidate.strategy
                    in {
                        SourceStrategy.PINNED_OFFICIAL_GIT_REPOSITORY,
                        SourceStrategy.MINIMAL_VENDORED_ADAPTATION,
                    }
                    else None
                ),
                source_package_name=(
                    _normalize_name(source_assessment["observed_packaging"]["package_name"])
                    if source_candidate.strategy == SourceStrategy.OFFICIAL_PACKAGE
                    else None
                ),
                source_package_version=(
                    source_assessment["observed_packaging"]["package_version"]
                    if source_candidate.strategy == SourceStrategy.OFFICIAL_PACKAGE
                    else None
                ),
                installation_strategy=source_candidate.strategy,
                expected_compatibility_evidence=tuple(
                    dict.fromkeys(
                        dependency_evidence_ids
                        + source_candidate.evidence_ids
                        + (() if dependency_evidence_ids else fallback_evidence_ids)
                    )
                ),
                trial_status=CandidateTrialStatus.NOT_ATTEMPTED,
                failure_classification=None,
                failure_diagnostics=None,
                uncertainty=tuple(
                    sorted(set(deps["unpinned_dependencies"]) | set(dependency_conflicts))
                ),
                predicted_failure_risks=tuple(sorted(set(risks), key=lambda item: item.value)),
                trial_command_plan=(
                    "Prepare environments/<card-id>/pyproject.toml using selected pinned versions.",
                    "Run uv lock inside the model-specific environment artifact directory.",
                    "Use torch-dae env ensure <card-id> after Phase 01 artifacts are committed.",
                ),
            )
        )
    if not deps["declared_dependencies"]:
        candidates = [
            candidate.model_copy(
                update={"predicted_failure_risks": (FailureClassification.INSUFFICIENT_EVIDENCE,)}
            )
            for candidate in candidates
        ]
    result = EnvironmentCandidateGenerationResult(
        schema_version="1.0.0",
        evidence_items=evidence_items,
        dependency_records=tuple(sorted(records, key=_record_sort_key)),
        candidates=tuple(candidates),
        unresolved_constraints=tuple(
            sorted(set(deps["unpinned_dependencies"]) | set(dependency_conflicts))
        ),
        source_strategy_context=source_context,
        decision_gates=decision_gates,
        target_platform=target_platform,
    )
    return result.model_dump(mode="json")


def classify_source_strategy(
    root: Path,
    *,
    external_pytorch_root: Path | None = None,
    budget: InspectionBudget | None = None,
) -> dict[str, Any]:
    """Assess source-strategy evidence without selecting an official strategy."""

    root = repository_root(root)
    budget = budget or InspectionBudget()
    project = _read_pyproject(root / "pyproject.toml", root, budget)
    import_result = inspect_imports(root, budget=budget)
    imports = set(import_result["frameworks"])
    observed_packaging = {
        "package_metadata_exists": bool(project.get("name") and project.get("version")),
        "package_name": project.get("name"),
        "package_version": project.get("version"),
        "broken_packaging_marker": (root / "BROKEN_PACKAGING").exists(),
    }
    observed_repository_revision = _observed_revision(root, budget)
    officiality = _officiality_assessment(root, budget)
    observed_framework = (
        "non_pytorch"
        if {"tensorflow", "jax"} & imports
        else ("pytorch" if imports & {"torch", "torchaudio"} else "unresolved")
    )
    candidates: list[dict[str, Any]] = []
    if observed_packaging["package_metadata_exists"]:
        candidates.append(
            {
                "strategy": SourceStrategy.OFFICIAL_PACKAGE.value,
                "status": "unresolved_ambiguity",
                "evidence": ["pyproject.toml"],
                "rationale": (
                    "Package metadata exists, but officiality and semantic equivalence require "
                    "separate upstream evidence."
                ),
                "decision_required": True,
            }
        )
    if observed_repository_revision:
        candidates.append(
            {
                "strategy": SourceStrategy.PINNED_OFFICIAL_GIT_REPOSITORY.value,
                "status": "unresolved_ambiguity",
                "evidence": [observed_repository_revision["source"]],
                "rationale": (
                    "An immutable revision may be available, but repository officiality is not "
                    "established by revision evidence alone."
                ),
                "decision_required": True,
            }
        )
    vendoring = _vendoring_assessment(root, observed_repository_revision, budget)
    if vendoring["candidate_supported"]:
        candidates.append(
            {
                "strategy": SourceStrategy.MINIMAL_VENDORED_ADAPTATION.value,
                "status": "unresolved_ambiguity",
                "evidence": vendoring["evidence"],
                "rationale": (
                    "Packaging is marked unsuitable, but vendoring also requires failed package "
                    "or Git strategies and a known minimal copied subset."
                ),
                "decision_required": True,
            }
        )
    if {"tensorflow", "jax"} & imports:
        if external_pytorch_root is not None:
            external = repository_root(external_pytorch_root)
            external_import_result = inspect_imports(external, budget=budget)
            external_imports = set(external_import_result["frameworks"])
            if external_imports & {"torch", "torchaudio"}:
                candidates.append(
                    {
                        "strategy": SourceStrategy.EXTERNAL_PYTORCH_IMPLEMENTATION.value,
                        "status": "unresolved_ambiguity",
                        "evidence": tuple(
                            sorted(
                                {
                                    item["path"]
                                    for item in import_result["files"]
                                    if set(item.get("imports", ())) & {"tensorflow", "jax"}
                                }
                                | {
                                    item["path"]
                                    for item in external_import_result["files"]
                                    if set(item.get("imports", ())) & {"torch", "torchaudio"}
                                }
                            )
                        ),
                        "rationale": (
                            "A non-PyTorch upstream and an explicit PyTorch implementation were "
                            "both inspected; semantic equivalence still requires review."
                        ),
                        "decision_required": True,
                    }
                )
    if not candidates:
        candidates.append(
            {
                "strategy": SourceStrategy.UNSUPPORTED_OR_NON_EQUIVALENT_IMPLEMENTATION.value,
                "status": "unsupported_claim",
                "evidence": [
                    "README.md" if (root / "README.md").exists() else "repository_inventory"
                ],
                "rationale": (
                    "Insufficient technical evidence for a reliable PyTorch source strategy."
                ),
                "decision_required": True,
            }
        )
    return {
        "observed_packaging": observed_packaging,
        "observed_repository_revision": observed_repository_revision,
        "observed_framework": observed_framework,
        "external_wrapper_assessment": _external_wrapper_assessment(
            root, external_pytorch_root, budget
        ),
        "vendoring_assessment": vendoring,
        "officiality_status": officiality["status"],
        "officiality_evidence": officiality,
        "equivalence_status": "unresolved",
        "decision_required": any(candidate["decision_required"] for candidate in candidates),
        "source_strategy_candidates": candidates,
    }


def _synthetic_marker(root: Path, budget: InspectionBudget) -> bool:
    path = root / "README.md"
    return (
        path.exists()
        and _looks_textual(path)
        and "synthetic" in _read_text_file(path, root, budget).lower()
    )


def _officiality_assessment(root: Path, budget: InspectionBudget) -> dict[str, Any]:
    path = root / "OFFICIAL.md"
    if not path.exists():
        return {"status": "unresolved", "evidence": ()}
    text = _read_text_file(path, root, budget).lower()
    if "explicitly states" in text and "official" in text and "synthetic upstream" in text:
        return {"status": "locally_observed_behavior", "evidence": ("OFFICIAL.md",)}
    return {"status": "unresolved", "evidence": ("OFFICIAL.md",)}


def _source_roots(root: Path) -> list[str]:
    candidates = []
    for dirname in ("src", "."):
        path = root / dirname
        if path.exists() and any(child.suffix == ".py" for child in path.rglob("*.py")):
            candidates.append(dirname)
    return sorted(set(candidates))


def _read_pyproject(path: Path, root: Path, budget: InspectionBudget) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = _read_toml(path, root, budget)
    project = data.get("project", {})
    build = data.get("build-system", {})
    optional = project.get("optional-dependencies", {})
    return {
        "name": project.get("name"),
        "version": project.get("version"),
        "requires_python": project.get("requires-python"),
        "dependencies": tuple(project.get("dependencies", ())),
        "optional_dependencies": optional,
        "build_backend": build.get("build-backend"),
        "entry_points": project.get("scripts", {}),
        "dependency_groups": data.get("dependency-groups", {}),
    }


def _read_setup_cfg(path: Path, root: Path, budget: InspectionBudget) -> dict[str, Any]:
    if not path.exists():
        return {}
    parser = configparser.ConfigParser()
    try:
        parser.read_string(_read_text_file(path, root, budget))
    except configparser.Error as exc:
        raise OnboardingInspectionError(f"failed to parse setup.cfg {path}: {exc}") from exc
    metadata = dict(parser.items("metadata")) if parser.has_section("metadata") else {}
    options = dict(parser.items("options")) if parser.has_section("options") else {}
    return {
        "name": metadata.get("name"),
        "version": metadata.get("version"),
        "python_requires": options.get("python_requires"),
        "install_requires": _split_config_lines(options.get("install_requires")),
        "package_dir": options.get("package_dir"),
        "package_data": _split_config_lines(options.get("package_data")),
    }


def _inspect_setup_py(path: Path, root: Path, budget: InspectionBudget) -> dict[str, Any]:
    if not path.exists():
        return {}
    tree = _parse_ast(path, root, budget)
    result: dict[str, Any] = {"dynamic": [], "values": {}}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _attribute_name(node.func).endswith("setup"):
            for keyword in node.keywords:
                if keyword.arg is None:
                    continue
                literal = _literal(keyword.value)
                if literal is None:
                    result["dynamic"].append(keyword.arg)
                else:
                    result["values"][keyword.arg] = literal
    return result


def _parse_requirements(
    path: Path, source_file: str, root: Path, budget: InspectionBudget
) -> list[dict[str, Any]]:
    items = []
    for line in _read_text_file(path, root, budget).splitlines():
        raw = _strip_requirement_comment(line)
        if not raw:
            continue
        if raw.startswith("-e "):
            requirement_text = raw[3:].strip()
            parsed = _parse_editable_requirement(requirement_text)
            items.append(
                {
                    "raw": requirement_text,
                    "name": parsed["name"],
                    "pinned": False,
                    "valid": True,
                    "kind": "editable",
                    "dependency_kind": parsed["dependency_kind"],
                    "editable": True,
                    "direct_url": parsed["direct_url"],
                    "vcs": parsed["vcs"],
                    "local_path": parsed["local_path"],
                    "source_file": source_file,
                    "source_section": "requirements",
                }
            )
            continue
        if raw.startswith(("git+", "http://", "https://")):
            items.append(
                {
                    "raw": raw,
                    "name": None,
                    "pinned": False,
                    "valid": True,
                    "kind": "direct_url",
                    "dependency_kind": (
                        DependencyKind.VCS if raw.startswith("git+") else DependencyKind.DIRECT_URL
                    ),
                    "editable": False,
                    "direct_url": True,
                    "vcs": _vcs_name(raw),
                    "local_path": False,
                    "source_file": source_file,
                    "source_section": "requirements",
                }
            )
            continue
        if raw.startswith("-"):
            continue
        name = _dependency_name(raw)
        try:
            Requirement(raw)
            valid = True
        except InvalidRequirement:
            valid = False
        items.append(
            {
                "raw": raw,
                "name": name,
                "pinned": _is_pinned(raw),
                "valid": valid,
                "kind": "requirement",
                "dependency_kind": DependencyKind.REQUIREMENT,
                "editable": False,
                "direct_url": False,
                "vcs": None,
                "local_path": _looks_local_path(raw),
                "source_file": source_file,
                "source_section": "requirements",
            }
        )
    return items


def _strip_requirement_comment(line: str) -> str:
    stripped = line.strip()
    if "#egg=" in stripped:
        return stripped
    return re.split(r"\s+#", stripped, maxsplit=1)[0].strip()


def _dependency_name(requirement: str) -> str | None:
    try:
        return Requirement(requirement).name.lower()
    except InvalidRequirement:
        return None


def _is_pinned(requirement: str) -> bool:
    try:
        parsed = Requirement(requirement)
    except InvalidRequirement:
        return False
    return any(spec.operator == "==" for spec in parsed.specifier)


def _parse_ast(path: Path, root: Path, budget: InspectionBudget) -> ast.AST:
    try:
        return ast.parse(_read_text_file(path, root, budget), filename=str(path))
    except (SyntaxError, OSError, UnicodeError) as exc:
        raise OnboardingInspectionError(f"failed to parse Python source {path}: {exc}") from exc


def _attribute_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _attribute_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Call):
        return _attribute_name(node.func)
    return ""


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return None


def _return_keys(node: ast.AST | None) -> list[str]:
    if node is None:
        return []
    if isinstance(node, ast.Dict):
        values = []
        for key in node.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                values.append(key.value)
        return values
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Tuple | ast.List):
        sequence_values: list[str] = []
        for child in node.elts:
            sequence_values.extend(_return_keys(child))
        return sequence_values
    return []


def _checkpoint_helpers(path: str, text: str) -> list[dict[str, Any]]:
    tree = ast.parse(text)
    helpers: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        lower = node.name.lower()
        if not any(token in lower for token in ("download", "checkpoint", "pretrained", "weight")):
            continue
        environment, observed_values = _checkpoint_static_values(node)
        associations = [
            association
            for value in observed_values
            for association in _checkpoint_associations(value)
        ]
        associations.extend(_keyed_checkpoint_associations(environment))
        if not associations:
            strings = set(_iter_static_strings(observed_values))
            associations.extend(
                (url, _checkpoint_filename(url), ())
                for url in sorted(strings)
                if _complete_checkpoint_url(url)
            )
        merged: dict[tuple[str | None, str | None], set[str]] = {}
        for complete_url, filename, associated_hashes in associations:
            if complete_url is None and filename is None:
                continue
            key = (complete_url, filename or _checkpoint_filename(complete_url))
            merged.setdefault(key, set()).update(associated_hashes)
        if not merged:
            merged[(None, None)] = set()
        for (complete_url, filename), hash_set in sorted(
            merged.items(), key=lambda item: (item[0][0] or "", item[0][1] or "")
        ):
            unresolved_components = []
            if complete_url is None:
                unresolved_components.append("complete URL")
            if not hash_set:
                unresolved_components.append("hash association")
            helpers.append(
                {
                    "kind": "checkpoint_helper",
                    "path": path,
                    "source_file": path,
                    "symbol": node.name,
                    "helper_symbol": node.name,
                    "urls": [complete_url] if complete_url else [],
                    "complete_candidate_urls": [complete_url] if complete_url else [],
                    "complete_url": complete_url,
                    "filenames": [filename] if filename else [],
                    "filename": filename,
                    "associated_hashes": sorted(hash_set),
                    "hashes": sorted(hash_set),
                    "expression_status": "resolved" if complete_url else "unresolved",
                    "unresolved_components": unresolved_components,
                }
            )
    return helpers


def _checkpoint_static_values(
    node: ast.FunctionDef,
) -> tuple[dict[str, Any], list[Any]]:
    environment: dict[str, Any] = {}
    observed: list[Any] = []
    for child in node.body:
        value_node: ast.AST | None = None
        target_name: str | None = None
        if (
            isinstance(child, ast.Assign)
            and len(child.targets) == 1
            and isinstance(child.targets[0], ast.Name)
        ):
            target_name = child.targets[0].id
            value_node = child.value
        elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            target_name = child.target.id
            value_node = child.value
        elif isinstance(child, ast.Return):
            value_node = child.value
        if value_node is None:
            continue
        value = _safe_checkpoint_value(value_node, environment)
        if target_name is not None:
            environment[target_name] = value
        observed.append(value)
    return environment, observed


def _safe_checkpoint_value(node: ast.AST, environment: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return environment.get(node.id)
    if isinstance(node, ast.Dict):
        return {
            _safe_checkpoint_value(key, environment): _safe_checkpoint_value(value, environment)
            for key, value in zip(node.keys, node.values, strict=True)
            if key is not None
        }
    if isinstance(node, ast.Tuple | ast.List):
        return tuple(_safe_checkpoint_value(item, environment) for item in node.elts)
    if isinstance(node, ast.Subscript):
        container = _safe_checkpoint_value(node.value, environment)
        key = _safe_checkpoint_value(node.slice, environment)
        if isinstance(container, dict):
            return container.get(key)
        if isinstance(container, tuple) and isinstance(key, int) and 0 <= key < len(container):
            return container[key]
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _safe_checkpoint_value(node.left, environment)
        right = _safe_checkpoint_value(node.right, environment)
        return left + right if isinstance(left, str) and isinstance(right, str) else None
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                rendered = _safe_checkpoint_value(value.value, environment)
                if not isinstance(rendered, str):
                    return None
                parts.append(rendered)
            else:
                return None
        return "".join(parts)
    if isinstance(node, ast.Call):
        function = _attribute_name(node.func)
        if function in {"os.path.join", "pathlib.Path"}:
            parts = [_safe_checkpoint_value(argument, environment) for argument in node.args]
            if all(isinstance(part, str) for part in parts):
                return os.path.join(*(part for part in parts if isinstance(part, str)))
    return None


def _checkpoint_associations(
    value: Any,
) -> list[tuple[str | None, str | None, tuple[str, ...]]]:
    if isinstance(value, dict):
        lowered = {str(key).lower(): item for key, item in value.items()}
        direct_values = tuple(item for item in lowered.values() if isinstance(item, str))
        urls = tuple(item for item in direct_values if _complete_checkpoint_url(item))
        hashes = tuple(sorted(item for item in direct_values if _checkpoint_hash(item)))
        explicit_filename = next(
            (item for key, item in lowered.items() if "filename" in key and isinstance(item, str)),
            None,
        )
        if urls and (
            hashes or explicit_filename is not None or any("url" in key for key in lowered)
        ):
            return [(url, explicit_filename or _checkpoint_filename(url), hashes) for url in urls]
        return [
            association
            for child in value.values()
            for association in _checkpoint_associations(child)
        ]
    if isinstance(value, tuple):
        direct_values = tuple(item for item in value if isinstance(item, str))
        urls = tuple(item for item in direct_values if _complete_checkpoint_url(item))
        hashes = tuple(sorted(item for item in direct_values if _checkpoint_hash(item)))
        filenames = tuple(
            item
            for item in direct_values
            if not item.startswith(("http://", "https://"))
            and any(item.endswith(suffix) for suffix in CHECKPOINT_SUFFIXES)
        )
        if urls:
            filename = filenames[0] if len(filenames) == 1 else None
            return [(url, filename or _checkpoint_filename(url), hashes) for url in urls]
        return [association for child in value for association in _checkpoint_associations(child)]
    return []


def _keyed_checkpoint_associations(
    environment: dict[str, Any],
) -> list[tuple[str | None, str | None, tuple[str, ...]]]:
    by_key: dict[Any, dict[str, set[str]]] = {}
    for name, mapping in environment.items():
        if not isinstance(mapping, dict):
            continue
        semantic_name = name.lower()
        for key, value in mapping.items():
            if not isinstance(value, str):
                continue
            entry = by_key.setdefault(key, {"urls": set(), "filenames": set(), "hashes": set()})
            if _complete_checkpoint_url(value) and (
                "url" in semantic_name or "checkpoint" in semantic_name or "weight" in semantic_name
            ):
                entry["urls"].add(value)
            elif _checkpoint_hash(value) and ("hash" in semantic_name or "sha" in semantic_name):
                entry["hashes"].add(value)
            elif any(value.endswith(suffix) for suffix in CHECKPOINT_SUFFIXES) and (
                "file" in semantic_name or "name" in semantic_name
            ):
                entry["filenames"].add(value)
    associations: list[tuple[str | None, str | None, tuple[str, ...]]] = []
    for entry in by_key.values():
        filename = next(iter(entry["filenames"])) if len(entry["filenames"]) == 1 else None
        associations.extend(
            (
                url,
                filename or _checkpoint_filename(url),
                tuple(sorted(entry["hashes"])),
            )
            for url in sorted(entry["urls"])
        )
    return associations


def _iter_static_strings(values: Any) -> list[str]:
    if isinstance(values, str):
        return [values]
    if isinstance(values, dict):
        return [item for value in values.values() for item in _iter_static_strings(value)]
    if isinstance(values, (list, tuple)):
        return [item for value in values for item in _iter_static_strings(value)]
    return []


def _complete_checkpoint_url(value: str) -> bool:
    return value.startswith(("http://", "https://")) and any(
        suffix in value for suffix in CHECKPOINT_SUFFIXES
    )


def _checkpoint_filename(value: str | None) -> str | None:
    if value is None or not _complete_checkpoint_url(value):
        return None
    return Path(value.split("?", 1)[0]).name


def _checkpoint_hash(value: str) -> bool:
    return re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _select_python_version(
    constraint: str,
    records: list[DependencyEvidenceRecord] | None = None,
) -> str | None:
    for record in records or []:
        if record.normalized_name == "python" and record.exact_version:
            return record.exact_version
    if constraint == "unresolved":
        return None
    try:
        SpecifierSet(constraint)
    except InvalidSpecifier:
        return None
    return None


def inspect_scenario_repository(
    scenario_root: Path,
    *,
    scenario_id: str,
    external_pytorch_root: Path | None = None,
    budget: InspectionBudget | None = None,
) -> ScenarioInspectionResult:
    """Run production inspectors and package their observations for evaluation."""

    root = repository_root(scenario_root)
    budget = budget or InspectionBudget()
    source_strategy_assessment = classify_source_strategy(
        root, external_pytorch_root=external_pytorch_root, budget=budget
    )
    repository_inventory = inspect_repository(root, budget=budget)
    warnings = tuple(
        item["path"]
        for item in repository_inventory.get("skipped_files", ())
        if item.get("skipped_reason")
    )
    return ScenarioInspectionResult(
        schema_version="1.0.0",
        scenario_id=scenario_id,
        repository_inventory=repository_inventory,
        packaging_evidence=inspect_python_project(root, budget=budget),
        dependency_evidence=inspect_dependencies(root, budget=budget),
        import_evidence=inspect_imports(root, budget=budget),
        model_candidates=inspect_model_candidates(root, budget=budget),
        output_candidates=inspect_output_candidates(root, budget=budget),
        checkpoint_candidates=inspect_checkpoints(root, budget=budget),
        source_strategy_assessment=source_strategy_assessment,
        environment_candidates=EnvironmentCandidateGenerationResult.model_validate(
            generate_environment_candidates(
                root, external_pytorch_root=external_pytorch_root, budget=budget
            )
        ),
        inspection_warnings=warnings,
    )


def _dependency_record(
    raw: str,
    source_file: str,
    source_section: str,
    *,
    dependency_kind: DependencyKind = DependencyKind.REQUIREMENT,
    editable: bool = False,
    direct_url: bool = False,
    vcs: str | None = None,
    local_path: bool = False,
    valid: bool | None = None,
) -> DependencyEvidenceRecord:
    normalized_name, constraint, exact_version, parsed_valid = _parse_dependency(
        raw, dependency_kind
    )
    parsed_kind, parsed_direct_url, parsed_vcs, parsed_local_path = _dependency_transport(raw)
    if parsed_direct_url:
        dependency_kind = parsed_kind
        direct_url = True
        vcs = parsed_vcs
        local_path = parsed_local_path
    return DependencyEvidenceRecord(
        normalized_name=normalized_name,
        raw_declaration=raw,
        constraint=constraint,
        exact_version=exact_version,
        source_file=source_file,
        source_section=source_section,
        dependency_kind=dependency_kind,
        editable=editable,
        direct_url=direct_url,
        vcs=vcs,
        local_path=local_path,
        valid=parsed_valid if valid is None else valid,
        evidence_id=_evidence_id_for_dependency(source_file, source_section, raw),
    )


def _parse_dependency(
    raw: str, dependency_kind: DependencyKind
) -> tuple[str | None, str | None, str | None, bool]:
    if dependency_kind == DependencyKind.CONDA:
        return _parse_conda_dependency(raw)
    try:
        requirement = Requirement(raw)
    except InvalidRequirement:
        egg_match = re.search(r"[#&]egg=([A-Za-z0-9_.-]+)", raw)
        if egg_match:
            return (_normalize_name(egg_match.group(1)), None, None, True)
        return (
            _normalize_name(raw.split("=", 1)[0].split()[0] if raw.split() else None),
            None,
            None,
            False,
        )
    constraint = str(requirement.specifier) or None
    exact_versions = [
        specifier.version for specifier in requirement.specifier if specifier.operator == "=="
    ]
    return (
        _normalize_name(requirement.name),
        constraint,
        exact_versions[0] if exact_versions else None,
        True,
    )


def _dependency_transport(raw: str) -> tuple[DependencyKind, bool, str | None, bool]:
    try:
        requirement = Requirement(raw)
    except InvalidRequirement:
        if raw.startswith(("git+", "hg+", "svn+")):
            return (DependencyKind.VCS, True, _vcs_name(raw), False)
        return (
            (DependencyKind.LOCAL_PATH, False, None, True)
            if _looks_local_path(raw)
            else (
                DependencyKind.UNKNOWN,
                False,
                None,
                False,
            )
        )
    if requirement.url:
        vcs = _vcs_name(requirement.url)
        return (
            DependencyKind.VCS if vcs else DependencyKind.DIRECT_URL,
            True,
            vcs,
            _looks_local_path(requirement.url),
        )
    return (
        (
            DependencyKind.LOCAL_PATH,
            False,
            None,
            True,
        )
        if _looks_local_path(raw)
        else (DependencyKind.REQUIREMENT, False, None, False)
    )


def _parse_editable_requirement(raw: str) -> dict[str, Any]:
    vcs = _vcs_name(raw)
    egg_match = re.search(r"[#&]egg=([A-Za-z0-9_.-]+)", raw)
    name = _normalize_name(egg_match.group(1)) if egg_match else _dependency_name(raw)
    return {
        "name": name,
        "dependency_kind": DependencyKind.VCS if vcs else DependencyKind.LOCAL_PATH,
        "direct_url": bool(vcs or raw.startswith(("http://", "https://"))),
        "vcs": vcs,
        "local_path": _looks_local_path(raw),
    }


def _parse_conda_dependency(raw: str) -> tuple[str | None, str | None, str | None, bool]:
    text = raw.strip()
    match = re.fullmatch(
        r"([A-Za-z0-9_.-]+)\s*(?:(~=|!=|<=|>=|==|=|<|>)\s*([A-Za-z0-9_.!+*-]+)(?:=([A-Za-z0-9_.!+*-]+))?)?",
        text,
    )
    if not match:
        name = re.split(r"[<>=!~\s]", text, maxsplit=1)[0] or None
        return (_normalize_name(name), None, None, False)
    name = _normalize_name(match.group(1))
    operator = match.group(2)
    version = match.group(3)
    if version is None:
        return (name, None, None, True)
    constraint_operator = "==" if operator == "=" else operator
    constraint = f"{constraint_operator}{version}"
    exact_version = version if constraint_operator == "==" else None
    try:
        SpecifierSet(constraint)
        if exact_version is not None:
            Version(exact_version)
    except (InvalidSpecifier, InvalidVersion):
        return (name, None, None, False)
    return (name, constraint, exact_version, True)


def _normalize_name(name: str | None) -> str | None:
    if not name:
        return None
    normalized = name.lower().replace("_", "-")
    return {"pytorch": "torch"}.get(normalized, normalized)


def _evidence_id_for_dependency(source_file: str, source_section: str, raw: str) -> str:
    token = f"{source_file}-{source_section}-{raw}".lower()
    token = re.sub(r"[^a-z0-9]+", "-", token).strip("-")
    return f"ev-{token}"[:80].rstrip("-")


def _record_sort_key(record: DependencyEvidenceRecord) -> tuple[str, str, str]:
    return (record.source_file, record.source_section, record.raw_declaration)


def _first_dependency_source(deps: dict[str, Any]) -> str | None:
    records = deps.get("dependency_records", ())
    return str(records[0]["source_file"]) if records else None


def _principal_dependency_versions(
    records: list[DependencyEvidenceRecord],
) -> dict[str, dict[str, str | None]]:
    result: dict[str, dict[str, str | None]] = {}
    conflicts: dict[str, set[str]] = {}
    unpinned: set[str] = set()
    for record in records:
        if record.normalized_name is None or not record.valid:
            continue
        entry = result.setdefault(record.normalized_name, {"constraint": None, "version": None})
        if record.constraint and entry["constraint"] in {None, "unconstrained"}:
            entry["constraint"] = record.constraint
        if record.exact_version:
            conflicts.setdefault(record.normalized_name, set()).add(record.exact_version)
            entry["version"] = record.exact_version
        else:
            unpinned.add(record.normalized_name)
    for name, versions in conflicts.items():
        if len(versions) > 1 or name in unpinned:
            result[name]["version"] = None
            result[name]["constraint"] = None
    return result


def _dependency_conflicts(records: list[DependencyEvidenceRecord]) -> list[str]:
    exact_by_name: dict[str, set[str]] = {}
    ranged_by_name: set[str] = set()
    for record in records:
        if record.normalized_name is None or not record.valid:
            continue
        if record.exact_version:
            exact_by_name.setdefault(record.normalized_name, set()).add(record.exact_version)
        elif record.constraint and record.constraint != "unconstrained":
            ranged_by_name.add(record.normalized_name)
    conflicts = []
    for name, versions in sorted(exact_by_name.items()):
        if len(versions) > 1:
            conflicts.append(f"{name} exact version conflict: {', '.join(sorted(versions))}")
        elif name in ranged_by_name:
            conflicts.append(f"{name} exact and range constraint conflict")
    return conflicts


def _incompatible_torch_audio(by_name: dict[str, dict[str, str | None]]) -> bool:
    torch_version = by_name.get("torch", {}).get("version")
    audio_version = by_name.get("torchaudio", {}).get("version")
    if not torch_version or not audio_version:
        return False
    try:
        return Version(torch_version).release[:2] != Version(audio_version).release[:2]
    except InvalidVersion:
        return True


def _source_strategy_contracts(assessment: dict[str, Any]) -> list[SourceStrategyCandidate]:
    return [
        SourceStrategyCandidate(
            strategy=item["strategy"],
            status=item["status"],
            rationale=item["rationale"],
            evidence_ids=tuple(
                _evidence_id_for_source_evidence(evidence) for evidence in item.get("evidence", ())
            ),
            user_decision_required=item.get("decision_required", False),
            unresolved_reason=(
                item["rationale"] if item["status"] == "unresolved_ambiguity" else None
            ),
        )
        for item in assessment.get("source_strategy_candidates", ())
    ]


def _evidence_item_for_dependency(record: DependencyEvidenceRecord) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=record.evidence_id,
        kind=EvidenceItemKind.CONFIGURATION_FILE,
        claim_status=ClaimStatus.LOCALLY_OBSERVED_BEHAVIOR,
        description=(
            "Dependency declaration observed in "
            f"{record.source_file} [{record.source_section}]: {record.raw_declaration}"
        ),
        source_file=record.source_file,
    )


def _source_strategy_evidence_items(assessment: dict[str, Any]) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    observed_packaging = assessment.get("observed_packaging", {})
    for candidate in assessment.get("source_strategy_candidates", ()):
        for evidence in candidate.get("evidence", ()):
            evidence_id = _evidence_id_for_source_evidence(str(evidence))
            source_file = (
                str(evidence) if _looks_repository_relative_evidence(str(evidence)) else None
            )
            is_package_metadata = (
                source_file in {"pyproject.toml", "setup.py", "setup.cfg"}
                and observed_packaging.get("package_name")
                and observed_packaging.get("package_version")
            )
            items.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    kind=EvidenceItemKind.PACKAGE_METADATA
                    if is_package_metadata
                    else (
                        EvidenceItemKind.SOURCE_FILE
                        if source_file
                        else EvidenceItemKind.AGENT_INFERENCE
                    ),
                    claim_status=ClaimStatus.LOCALLY_OBSERVED_BEHAVIOR
                    if source_file
                    else ClaimStatus.REASONED_INFERENCE,
                    description=f"Source strategy evidence observed from {evidence}.",
                    source_file=source_file,
                    package_name=_normalize_name(observed_packaging.get("package_name"))
                    if is_package_metadata
                    else None,
                    package_version=observed_packaging.get("package_version")
                    if is_package_metadata
                    else None,
                    rationale=None if source_file else "Synthetic aggregate inspection evidence.",
                )
            )
    return items


def _evidence_id_for_source_evidence(evidence: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "-", evidence.lower()).strip("-") or "source-strategy"
    return f"ev-{token}"[:80].rstrip("-")


def _looks_repository_relative_evidence(evidence: str) -> bool:
    return (
        evidence not in {"python_imports", "external_python_imports", "repository_inventory"}
        and evidence != ".git"
        and not evidence.startswith("/")
        and ".." not in Path(evidence).parts
    )


def _vendoring_assessment(
    root: Path, observed_revision: dict[str, str] | None, budget: InspectionBudget
) -> dict[str, Any]:
    evidence = [
        name for name in ("BROKEN_PACKAGING", "VENDORING.md", "setup.py") if (root / name).exists()
    ]
    if observed_revision is not None:
        evidence.append(observed_revision["source"])
    copied_files: list[str] = []
    if (root / "VENDORING.md").exists():
        copied_files = sorted(
            re.findall(
                r"`([^`]+\.py)`",
                _read_text_file(root / "VENDORING.md", root, budget),
            )
        )
    return {
        "package_unsuitable": (root / "BROKEN_PACKAGING").exists(),
        "pinned_git_unsuitable": (root / "VENDORING.md").exists(),
        "minimal_subset_identified": bool(copied_files),
        "copied_files": copied_files,
        "upstream_revision_available": observed_revision is not None,
        "adaptation_notes_exist": (root / "VENDORING.md").exists(),
        "semantic_deviations_known": (root / "VENDORING.md").exists(),
        "evidence": evidence,
        "candidate_supported": bool(
            (root / "BROKEN_PACKAGING").exists()
            and (root / "VENDORING.md").exists()
            and copied_files
            and observed_revision is not None
        ),
    }


def _external_wrapper_assessment(
    root: Path, external_pytorch_root: Path | None, budget: InspectionBudget
) -> dict[str, Any]:
    upstream_imports = set(inspect_imports(root, budget=budget)["frameworks"])
    upstream_identity = _logical_repository_identity(root, budget)
    if external_pytorch_root is None:
        return {
            "upstream_repository_identity": upstream_identity,
            "upstream_framework": "non_pytorch"
            if {"tensorflow", "jax"} & upstream_imports
            else "unresolved",
            "external_framework": None,
            "external_repository_identity": None,
            "unresolved_equivalence_questions": ("external PyTorch implementation not identified",),
            "user_decision_required": bool({"tensorflow", "jax"} & upstream_imports),
        }
    external = repository_root(external_pytorch_root)
    external_imports = set(inspect_imports(external, budget=budget)["frameworks"])
    mapping = _read_json(external / "EQUIVALENCE.json", external, budget)
    return {
        "upstream_repository_identity": mapping.get(
            "upstream_repository_identity", upstream_identity
        ),
        "external_repository_identity": mapping.get(
            "external_repository_identity", _logical_repository_identity(external, budget)
        ),
        "upstream_framework": "non_pytorch",
        "external_framework": "pytorch"
        if {"torch", "torchaudio"} & external_imports
        else "unresolved",
        "architecture_mapping": mapping.get("architecture_mapping"),
        "input_mapping": mapping.get("input_mapping"),
        "output_mapping": mapping.get("output_mapping"),
        "checkpoint_provenance": mapping.get("checkpoint_provenance"),
        "known_differences": tuple(mapping.get("known_differences", ())),
        "evidence_files": tuple(
            file
            for file in ("EQUIVALENCE.json", "UPSTREAM.json")
            if (external / file).exists() or (root / file).exists()
        ),
        "unresolved_equivalence_questions": tuple(
            mapping.get("unresolved_equivalence_questions", ())
        )
        or ("semantic equivalence requires user review",),
        "user_decision_required": True,
    }


def _logical_repository_identity(root: Path, budget: InspectionBudget) -> str:
    metadata = _read_json(root / "UPSTREAM.json", root, budget)
    if identity := metadata.get("repository_identity"):
        return str(identity)
    project = _read_pyproject(root / "pyproject.toml", root, budget)
    if project.get("name"):
        return str(project["name"])
    return root.name


def _parse_dockerfile_dependencies(
    root: Path, relative: str, budget: InspectionBudget
) -> list[DependencyEvidenceRecord]:
    text = _read_text_file(safe_child(root, relative), root, budget)
    return [
        _dependency_record(raw, relative, "dockerfile.pip")
        for match in re.finditer(r"pip install ([^\n]+)", text)
        for raw in match.group(1).split()
        if not raw.startswith("-")
    ]


def _parse_ci_dependencies(
    root: Path, relative: str, budget: InspectionBudget
) -> list[DependencyEvidenceRecord]:
    text = _read_text_file(safe_child(root, relative), root, budget)
    records: list[DependencyEvidenceRecord] = []
    key_map = {
        "python-version": "python",
        "torch-version": "torch",
        "pytorch-version": "torch",
        "torchaudio-version": "torchaudio",
        "numpy-version": "numpy",
    }
    lines = text.splitlines()
    context: list[tuple[int, str]] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        match = re.match(r"(\s*)([A-Za-z0-9_-]+):\s*(.*)$", line)
        if not match:
            index += 1
            continue
        indent = len(match.group(1))
        key = match.group(2)
        tail = match.group(3).strip()
        while context and context[-1][0] >= indent:
            context.pop()
        in_static_matrix = (
            len(context) >= 2 and context[-2][1] == "strategy" and context[-1][1] == "matrix"
        )
        if key not in key_map or not in_static_matrix:
            if not tail:
                context.append((indent, key))
            index += 1
            continue
        package = key_map[key]
        values: list[str] = []
        if tail:
            values.extend(_ci_matrix_values(tail))
        else:
            cursor = index + 1
            while cursor < len(lines):
                item_match = re.match(r"(\s*)-\s*(.+?)\s*$", lines[cursor])
                if not item_match or len(item_match.group(1)) <= indent:
                    break
                values.extend(_ci_matrix_values(item_match.group(2).strip()))
                cursor += 1
            index = cursor - 1
        for value in sorted(set(values)):
            records.append(
                _dependency_record(
                    f"{package}=={value}",
                    relative,
                    f"matrix.{key}",
                    dependency_kind=DependencyKind.LOCKED,
                )
            )
        index += 1
    return records


def _ci_matrix_values(text: str) -> list[str]:
    cleaned = text.strip()
    if "${{" in cleaned:
        return []
    if cleaned.startswith("[") and cleaned.endswith("]"):
        return [
            item.strip().strip("'\"")
            for item in cleaned[1:-1].split(",")
            if item.strip().strip("'\"")
        ]
    return [cleaned.strip("'\"")] if cleaned.strip("'\"") else []


def _vcs_name(raw: str) -> str | None:
    for prefix in ("git+", "hg+", "svn+"):
        if raw.startswith(prefix) or f"+{prefix}" in raw:
            return prefix[:-1]
    return None


def _looks_local_path(raw: str) -> bool:
    return raw.startswith(("./", "../", "/", "file:"))


def _read_text_file(path: Path, root: Path, budget: InspectionBudget) -> str:
    if not path.exists():
        raise OnboardingInspectionError(f"file does not exist: {path}")
    if path.is_symlink():
        raise OnboardingInspectionError(f"refusing to inspect symlinked file: {path}")
    try:
        stat = path.stat()
    except (OSError, PermissionError) as exc:
        raise OnboardingInspectionError(f"failed to stat {path}: {exc}") from exc
    if not path.is_file():
        raise OnboardingInspectionError(f"path is not a regular file: {path}")
    key = _budget_key(root, path)
    budget.visit_file(key, stat.st_size)
    cached = budget.cached_text(key)
    if cached is not None:
        return cached
    if not _looks_textual(path):
        raise OnboardingInspectionError(f"unsupported file encoding or binary file: {path}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            text = stream.read(budget.maximum_single_file_bytes + 1)
            budget.store_text(key, text)
            return text
    except UnicodeDecodeError as exc:
        raise OnboardingInspectionError(f"unsupported file encoding: {path}") from exc
    except (OSError, PermissionError) as exc:
        raise OnboardingInspectionError(f"failed to read {path}: {exc}") from exc


def _looks_textual(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            sample = stream.read(1024)
    except (OSError, PermissionError) as exc:
        raise OnboardingInspectionError(f"failed to inspect {path}: {exc}") from exc
    return b"\x00" not in sample


def _budget_key(root: Path, path: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        relative = path.resolve().as_posix()
    return f"{root.resolve().as_posix()}::{relative}"


def _read_toml(path: Path, root: Path, budget: InspectionBudget) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return tomllib.loads(_read_text_file(path, root, budget))
    except tomllib.TOMLDecodeError as exc:
        raise OnboardingInspectionError(f"failed to parse TOML {path}: {exc}") from exc


def _read_json(path: Path, root: Path, budget: InspectionBudget) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(_read_text_file(path, root, budget))
    except json.JSONDecodeError as exc:
        raise OnboardingInspectionError(f"failed to parse JSON {path}: {exc}") from exc
    return value if isinstance(value, dict) else {}


def _read_text_dependencies(path: Path, root: Path, budget: InspectionBudget) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = _read_text_file(path, root, budget)
    return {"dependencies": sorted(set(re.findall(r'name\s*=\s*"([^"]+)"', text)))}


def _read_poetry_lock(path: Path, root: Path, budget: InspectionBudget) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = _read_toml(path, root, budget)
    dependencies = []
    for package in data.get("package", ()):
        if isinstance(package, dict) and package.get("name") and package.get("version"):
            dependencies.append(f"{package['name']}=={package['version']}")
    return {"dependencies": sorted(set(dependencies))}


def _read_uv_lock(path: Path, root: Path, budget: InspectionBudget) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = _read_toml(path, root, budget)
    dependencies = []
    for package in data.get("package", ()):
        if isinstance(package, dict) and package.get("name") and package.get("version"):
            dependencies.append(f"{package['name']}=={package['version']}")
    return {"dependencies": sorted(set(dependencies))}


def _parse_environment_file(path: Path, root: Path, budget: InspectionBudget) -> dict[str, Any]:
    dependencies: list[str] = []
    in_dependencies = False
    for line in _read_text_file(path, root, budget).splitlines():
        stripped = line.strip()
        if stripped == "dependencies:":
            in_dependencies = True
            continue
        if in_dependencies and stripped.startswith("- "):
            value = stripped[2:].strip()
            if value and not value.startswith("pip:"):
                dependencies.append(value)
        elif in_dependencies and stripped and not line.startswith((" ", "-")):
            in_dependencies = False
    return {"dependencies": dependencies}


def _split_config_lines(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(line.strip() for line in value.splitlines() if line.strip())


def _observed_revision(root: Path, budget: InspectionBudget) -> dict[str, str] | None:
    commit = root / "COMMIT"
    if commit.exists():
        value = _read_text_file(commit, root, budget).strip()
        if re.fullmatch(r"[0-9a-f]{40}", value):
            return {"source": "COMMIT", "revision": value}
    if (root / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        revision = result.stdout.strip()
        if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", revision):
            return {"source": ".git", "revision": revision}
        return {"source": ".git", "revision": "present-but-unresolved"}
    return None


def _requirement_versions(requirement: str) -> dict[str, str | None]:
    try:
        parsed = Requirement(requirement)
    except InvalidRequirement:
        return {"constraint": requirement, "version": None}
    exact_versions = [
        specifier.version for specifier in parsed.specifier if specifier.operator == "=="
    ]
    return {
        "constraint": str(parsed.specifier) or None,
        "version": exact_versions[0] if exact_versions else None,
    }


def _function_string_constants(node: ast.FunctionDef) -> dict[str, str | None]:
    constants: dict[str, str | None] = {}
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Assign)
            and len(child.targets) == 1
            and isinstance(child.targets[0], ast.Name)
        ):
            constants[child.targets[0].id] = _safe_string_expr(child.value, constants)
        elif isinstance(child, ast.Return) and child.value is not None:
            constants["return"] = _safe_string_expr(child.value, constants)
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            constants.setdefault(f"literal-{len(constants)}", child.value)
    return constants


def _safe_string_expr(node: ast.AST, constants: dict[str, str | None]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _safe_string_expr(node.left, constants)
        right = _safe_string_expr(node.right, constants)
        return left + right if left is not None and right is not None else None
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                rendered = _safe_string_expr(value.value, constants)
                if rendered is None:
                    return None
                parts.append(rendered)
            else:
                return None
        return "".join(parts)
    if isinstance(node, ast.Call):
        func = _attribute_name(node.func)
        if func in {"os.path.join", "pathlib.Path"}:
            call_parts: list[str | None] = [_safe_string_expr(arg, constants) for arg in node.args]
            if all(part is not None for part in call_parts):
                safe_parts = [part for part in call_parts if part is not None]
                return os.path.join(*safe_parts)
    return None
