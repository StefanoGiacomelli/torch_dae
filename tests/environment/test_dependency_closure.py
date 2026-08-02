from __future__ import annotations

import socket
import tomllib
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from torch_dae.core.errors import EnvironmentDependencyClosureError
from torch_dae.environment.dependency_closure import validate_wheel_dependency_closure
from torch_dae.environment.manager import EnvironmentManager
from torch_dae.environment.results import (
    EnvironmentDependencyClosureResult,
    EnvironmentFailureClassification,
    RuntimeRequirementEvidence,
)

ROOT = Path(__file__).resolve().parents[2]
PANN_ENVIRONMENTS = (
    "panns-cnn14-16k-map-0438",
    "panns-resnet38-map-0434",
    "panns-wavegram-logmel-cnn14-map-0439",
)
PACKAGE_RUNTIME_REQUIREMENTS = (
    "jsonschema>=4.25.0",
    "packaging>=26.0",
    "platformdirs>=4.3.0",
    "pydantic>=2.11.0",
    "pypdf>=6,<7",
    "typer>=0.16.0",
)


def test_complete_lock_passes_preflight(tmp_path: Path) -> None:
    wheel, project, lock = closure_fixture(tmp_path, ("example-pkg>=1",))
    write_lock(lock, root_dependencies=("example-pkg",), packages={"example-pkg": "1.2.0"})

    result = preflight(wheel, project, lock)

    assert result.status == "passed"
    assert result.missing_requirements == ()
    assert result.incompatible_requirements == ()


def test_missing_wheel_runtime_dependency_fails(tmp_path: Path) -> None:
    wheel, project, lock = closure_fixture(tmp_path, ("missing-pkg>=1",))
    write_lock(lock)

    result = preflight(wheel, project, lock)

    assert result.status == "failed"
    assert names(result.missing_requirements) == ["missing-pkg"]
    assert result.failure_classification == EnvironmentFailureClassification.DEPENDENCY_CLOSURE


def test_incompatible_wheel_runtime_dependency_fails(tmp_path: Path) -> None:
    wheel, project, lock = closure_fixture(tmp_path, ("example-pkg>=2",))
    write_lock(lock, root_dependencies=("example-pkg",), packages={"example-pkg": "1.9"})

    result = preflight(wheel, project, lock)

    assert names(result.incompatible_requirements) == ["example-pkg"]
    assert result.incompatible_requirements[0].reachable_versions == ("1.9",)


def test_normalized_distribution_names_match(tmp_path: Path) -> None:
    wheel, project, lock = closure_fixture(tmp_path, ("Example_Pkg>=1",))
    write_lock(lock, root_dependencies=("example.pkg",), packages={"example-pkg": "1.0"})

    result = preflight(wheel, project, lock)

    assert result.status == "passed"
    assert names(result.satisfied_requirements) == ["example-pkg"]


def test_inactive_environment_marker_and_optional_extra_are_ignored(tmp_path: Path) -> None:
    requirements = (
        "active-pkg>=1; python_version >= '3.12'",
        "inactive-pkg>=1; python_version < '3.12'",
        "docs-pkg>=1; extra == 'docs'",
    )
    wheel, project, lock = closure_fixture(tmp_path, requirements)
    write_lock(lock, root_dependencies=("active-pkg",), packages={"active-pkg": "1.0"})

    result = preflight(wheel, project, lock)

    assert result.active_runtime_requirements == ('active-pkg>=1; python_version >= "3.12"',)
    assert result.status == "passed"


def test_active_environment_marker_is_enforced(tmp_path: Path) -> None:
    wheel, project, lock = closure_fixture(
        tmp_path, ("required-on-macos>=1; sys_platform == 'darwin'",)
    )
    write_lock(lock)

    result = preflight(wheel, project, lock)

    assert names(result.missing_requirements) == ["required-on-macos"]


def test_reachable_transitive_lock_entry_is_accepted(tmp_path: Path) -> None:
    wheel, project, lock = closure_fixture(tmp_path, ("transitive-pkg>=2",))
    write_lock(
        lock,
        root_dependencies=("parent-pkg",),
        packages={"parent-pkg": "1.0", "transitive-pkg": "2.1"},
        dependencies={"parent-pkg": ("transitive-pkg",)},
    )

    result = preflight(wheel, project, lock)

    assert result.status == "passed"
    assert names(result.satisfied_requirements) == ["transitive-pkg"]


def test_orphan_transitive_lock_entry_is_rejected_under_no_deps_policy(tmp_path: Path) -> None:
    wheel, project, lock = closure_fixture(tmp_path, ("orphan-pkg>=1",))
    write_lock(lock, packages={"orphan-pkg": "1.0"})

    result = preflight(wheel, project, lock)

    assert names(result.missing_requirements) == ["orphan-pkg"]
    assert result.installation_policy == "locked-project-closure-then-local-wheel-no-deps"


def test_preflight_performs_no_network_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel, project, lock = closure_fixture(tmp_path, ("example-pkg",))
    write_lock(lock, root_dependencies=("example-pkg",), packages={"example-pkg": "1.0"})

    def reject_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", reject_network)
    assert preflight(wheel, project, lock).status == "passed"


def test_preflight_requires_no_model_card_and_is_deterministic(tmp_path: Path) -> None:
    wheel, project, lock = closure_fixture(tmp_path, ("example-pkg",))
    write_lock(lock, root_dependencies=("example-pkg",), packages={"example-pkg": "1.0"})

    first = preflight(wheel, project, lock)
    second = preflight(wheel, project, lock)

    assert first == second
    assert not (tmp_path / "model_cards").exists()


def test_failed_preflight_runs_before_environment_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = EnvironmentManager(tmp_path)
    failed = failed_result()
    definition = SimpleNamespace(
        environment_id="synthetic-env",
        environment_spec_sha256="a" * 64,
    )
    monkeypatch.setattr(manager, "resolve_environment", lambda _environment_id: definition)
    monkeypatch.setattr(manager, "_preflight_definition", lambda _definition: failed)

    with pytest.raises(EnvironmentDependencyClosureError, match="preflight failed"):
        manager.materialize_environment("synthetic-env")

    assert not (tmp_path / ".torch-dae/environments").exists()


def test_preflight_control_plane_does_not_import_model_runtimes() -> None:
    module = (ROOT / "src/torch_dae/environment/dependency_closure.py").read_text()
    root_dependencies = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "dependencies"
    ]

    assert "import torch" not in module
    assert all(
        not item.lower().startswith(("torch", "torchaudio", "librosa"))
        for item in root_dependencies
    )


def test_all_three_corrected_panns_environments_pass_preflight(tmp_path: Path) -> None:
    wheel = make_wheel(tmp_path / "torch_deepaudioembedding.whl", PACKAGE_RUNTIME_REQUIREMENTS)
    for environment_id in PANN_ENVIRONMENTS:
        environment = ROOT / "environments" / environment_id
        result = validate_wheel_dependency_closure(
            environment_id=environment_id,
            wheel_path=wheel,
            project_path=environment / "pyproject.toml",
            lock_path=environment / "uv.lock",
            python_version="3.12.13",
            platform="macos-arm64",
            result_path=f"reports/{environment_id}.json",
        )
        assert result.status == "passed"
        assert len(result.satisfied_requirements) == 6


def test_original_incomplete_panns_direct_dependency_fixture_reports_six_missing(
    tmp_path: Path,
) -> None:
    wheel, project, lock = closure_fixture(tmp_path, PACKAGE_RUNTIME_REQUIREMENTS)
    project.write_text(project_text(("numpy==2.4.6", "torch==2.13.0", "torchlibrosa==0.1.0")))
    write_lock(
        lock,
        root_dependencies=("numpy", "torch", "torchlibrosa"),
        packages={"numpy": "2.4.6", "torch": "2.13.0", "torchlibrosa": "0.1.0"},
    )

    result = preflight(wheel, project, lock)

    assert names(result.missing_requirements) == [
        "jsonschema",
        "packaging",
        "platformdirs",
        "pydantic",
        "pypdf",
        "typer",
    ]


def test_all_repository_environment_definitions_still_resolve() -> None:
    manager = EnvironmentManager(ROOT)
    for environment_file in sorted((ROOT / "environments").glob("*/environment.json")):
        environment_id = environment_file.parent.name
        assert manager.resolve_environment(environment_id).environment_id == environment_id


def closure_fixture(tmp_path: Path, requirements: tuple[str, ...]) -> tuple[Path, Path, Path]:
    wheel = make_wheel(tmp_path / "fixture.whl", requirements)
    project = tmp_path / "pyproject.toml"
    project.write_text(project_text(()))
    lock = tmp_path / "uv.lock"
    return wheel, project, lock


def project_text(dependencies: tuple[str, ...]) -> str:
    rendered = "\n".join(f'  "{item}",' for item in dependencies)
    return (
        '[project]\nname = "fixture-environment"\nversion = "0.0.0"\n'
        'requires-python = "==3.12.13"\ndependencies = [\n'
        f"{rendered}\n]\n"
    )


def write_lock(
    path: Path,
    *,
    root_dependencies: tuple[str, ...] = (),
    packages: dict[str, str] | None = None,
    dependencies: dict[str, tuple[str, ...]] | None = None,
) -> None:
    packages = packages or {}
    dependencies = dependencies or {}
    root_edges = "\n".join(f'    {{ name = "{name}" }},' for name in root_dependencies)
    blocks = [
        'version = 1\nrevision = 3\nrequires-python = "==3.12.13"\n',
        '[[package]]\nname = "fixture-environment"\nversion = "0.0.0"\n'
        'source = { virtual = "." }\ndependencies = [\n'
        f"{root_edges}\n]\n",
    ]
    for name, version in packages.items():
        package_edges = "\n".join(
            f'    {{ name = "{child}" }},' for child in dependencies.get(name, ())
        )
        block = f'[[package]]\nname = "{name}"\nversion = "{version}"\n'
        block += 'source = { registry = "https://example.invalid/simple" }\n'
        if package_edges:
            block += f"dependencies = [\n{package_edges}\n]\n"
        blocks.append(block)
    path.write_text("\n".join(blocks))


def make_wheel(path: Path, requirements: tuple[str, ...]) -> Path:
    metadata = [
        "Metadata-Version: 2.4",
        "Name: torch-deepaudioembedding",
        "Version: 0.1.0",
        *(f"Requires-Dist: {requirement}" for requirement in requirements),
        "",
    ]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("torch_deepaudioembedding-0.1.0.dist-info/METADATA", "\n".join(metadata))
    return path


def preflight(wheel: Path, project: Path, lock: Path) -> EnvironmentDependencyClosureResult:
    return validate_wheel_dependency_closure(
        environment_id="fixture-environment",
        wheel_path=wheel,
        project_path=project,
        lock_path=lock,
        python_version="3.12.13",
        platform="macos-arm64",
        result_path="reports/preflight.json",
    )


def names(evidence: tuple[RuntimeRequirementEvidence, ...]) -> list[str]:
    return [item.normalized_name for item in evidence]


def failed_result() -> EnvironmentDependencyClosureResult:
    missing = RuntimeRequirementEvidence(
        requirement="missing>=1",
        normalized_name="missing",
        specifier=">=1",
        reachable_versions=(),
    )
    return EnvironmentDependencyClosureResult(
        schema_version="1.0.0",
        environment_id="synthetic-env",
        package_wheel_identity="torch-deepaudioembedding==0.1.0",
        package_wheel_sha256="b" * 64,
        python_version="3.12.13",
        platform="macos-arm64",
        installation_policy="locked-project-closure-then-local-wheel-no-deps",
        active_runtime_requirements=("missing>=1",),
        satisfied_requirements=(),
        missing_requirements=(missing,),
        incompatible_requirements=(),
        lockfile_sha256="c" * 64,
        status="failed",
        failure_classification=EnvironmentFailureClassification.DEPENDENCY_CLOSURE,
        result_path="reports/preflight.json",
    )
