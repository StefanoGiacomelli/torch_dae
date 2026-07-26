from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from torch_dae.onboarding.contracts import EnvironmentCandidateGenerationResult, SourceStrategy
from torch_dae.onboarding.inspection import (
    InspectionBudget,
    OnboardingInspectionError,
    classify_source_strategy,
    generate_environment_candidates,
    inspect_checkpoints,
    inspect_dependencies,
    inspect_model_candidates,
    inspect_output_candidates,
    inspect_repository,
    inspect_scenario_repository,
)


def fixture(repo_root: Path, name: str) -> Path:
    return repo_root / "tests/skills/fixtures/synthetic_onboarding" / name


def assessed_strategies(result: dict[str, object]) -> set[str]:
    return {
        str(candidate["strategy"])
        for candidate in result["source_strategy_candidates"]  # type: ignore[index]
    }


def test_required_synthetic_scenarios_classify_source_strategy(repo_root: Path) -> None:
    expected = {
        "official_package": SourceStrategy.OFFICIAL_PACKAGE.value,
        "pinned_git": SourceStrategy.PINNED_OFFICIAL_GIT_REPOSITORY.value,
        "minimal_vendoring": SourceStrategy.MINIMAL_VENDORED_ADAPTATION.value,
        "ambiguous_embeddings": SourceStrategy.OFFICIAL_PACKAGE.value,
        "non_pytorch_upstream": SourceStrategy.EXTERNAL_PYTORCH_IMPLEMENTATION.value,
        "unsupported": SourceStrategy.UNSUPPORTED_OR_NON_EQUIVALENT_IMPLEMENTATION.value,
    }
    for name, strategy in expected.items():
        result = classify_source_strategy(
            fixture(repo_root, name),
            external_pytorch_root=fixture(repo_root, "external_pytorch_implementation")
            if name == "non_pytorch_upstream"
            else None,
        )
        assert strategy in assessed_strategies(result)
        if name in {"official_package", "pinned_git"}:
            assert result["officiality_status"] == "locally_observed_behavior"
        else:
            assert result["officiality_status"] == "unresolved"
        assert result["equivalence_status"] == "unresolved"


def test_source_strategy_classifier_is_independent_of_scenario_oracles(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    source = fixture(repo_root, "official_package")
    clone = tmp_path / "official_package"
    subprocess.run(["cp", "-R", str(source), str(clone)], check=True)
    baseline = classify_source_strategy(clone)
    (clone / "SCENARIO.json").write_text(
        '{"scenario_id":"official-package","synthetic":true,'
        '"expected_source_strategy":"unsupported_or_non_equivalent_implementation"}\n'
    )
    assert classify_source_strategy(clone) == baseline
    (clone / "SCENARIO.json").unlink()
    assert classify_source_strategy(clone) == baseline


def test_repository_inventory_is_deterministic_and_marks_synthetic(repo_root: Path) -> None:
    root = fixture(repo_root, "official_package")
    first = inspect_repository(root)
    second = inspect_repository(root)
    assert first == second
    assert first["synthetic_marker"] is True
    assert "pyproject.toml" in [item["path"] for item in first["files"]]


def test_output_inspection_lists_ambiguous_embedding_candidates(repo_root: Path) -> None:
    result = inspect_output_candidates(fixture(repo_root, "ambiguous_embeddings"))
    symbols = {item["symbol"] for item in result["candidates"]}
    assert {"frame_features", "pooled_features", "classifier_input", "logits"} <= symbols


def test_hidden_checkpoint_helper_is_detected(repo_root: Path) -> None:
    result = inspect_checkpoints(fixture(repo_root, "hidden_checkpoint_helper"))
    helpers = [item for item in result["candidates"] if item["kind"] == "checkpoint_helper"]
    assert len(helpers) == 2
    assert helpers[0]["symbol"] == "get_pretrained_checkpoint_url"
    assert "hidden-audio-model.pth" in helpers[0]["filenames"]
    assert (
        "https://example.invalid/assets/hidden-audio-model.pth"
        in helpers[0]["complete_candidate_urls"]
    )
    assert helpers[0]["associated_hashes"] == [
        "0000000000000000000000000000000000000000000000000000000000000000"
    ]
    assert helpers[0]["unresolved_components"] == []
    assert helpers[1]["complete_url"] == ("https://example.invalid/assets/hidden-audio-model-b.pth")
    assert helpers[1]["filename"] == "hidden-audio-model-b.pth"
    assert helpers[1]["associated_hashes"] == [
        "1111111111111111111111111111111111111111111111111111111111111111"
    ]


def test_unpinned_dependencies_generate_ranked_risky_candidate(repo_root: Path) -> None:
    root = fixture(repo_root, "unpinned_dependencies")
    dependencies = inspect_dependencies(root)
    assert {"torch", "torchaudio", "numpy"} <= set(dependencies["unpinned_dependencies"])
    candidates = generate_environment_candidates(root)["candidates"]
    result = EnvironmentCandidateGenerationResult.model_validate(
        generate_environment_candidates(root, target_platform="macos-arm64")
    )
    assert result.target_platform == "macos-arm64"
    assert result.candidates[0].expected_compatibility_evidence
    assert candidates[0]["candidate_id"] == "evidence-ranked-1"
    assert candidates[0]["pytorch_version"] is None
    assert candidates[0]["torchaudio_version"] is None
    assert candidates[0]["numpy_version"] is None
    assert candidates[0]["pytorch_constraint"] is None
    assert {"torch", "torchaudio", "numpy"} <= set(candidates[0]["uncertainty"])
    assert "dependency_conflict" in candidates[0]["predicted_failure_risks"]
    assert "numpy_compatibility" in candidates[0]["predicted_failure_risks"]


def test_model_candidate_inspection_reports_static_candidates(repo_root: Path) -> None:
    result = inspect_model_candidates(fixture(repo_root, "official_package"))
    assert any(item["symbol"] == "ClearAudioModel" for item in result["candidates"])


def test_source_strategy_falls_back_to_static_evidence(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    package_root.mkdir()
    (package_root / "pyproject.toml").write_text(
        '[project]\nname = "static-package"\nversion = "1.0.0"\n'
    )
    assert SourceStrategy.OFFICIAL_PACKAGE.value in assessed_strategies(
        classify_source_strategy(package_root)
    )

    vendored_root = tmp_path / "vendored"
    vendored_root.mkdir()
    (vendored_root / "BROKEN_PACKAGING").write_text("synthetic broken packaging")
    assert SourceStrategy.MINIMAL_VENDORED_ADAPTATION.value not in assessed_strategies(
        classify_source_strategy(vendored_root)
    )
    (vendored_root / "COMMIT").write_text("0123456789abcdef0123456789abcdef01234567")
    (vendored_root / "setup.py").write_text(
        "from setuptools import setup\nsetup(name='vendored')\n"
    )
    (vendored_root / "VENDORING.md").write_text(
        "Package and pinned Git are unsuitable. Copied subset is `core.py`.\n"
    )
    (vendored_root / "core.py").write_text("import torch\n")
    assert SourceStrategy.MINIMAL_VENDORED_ADAPTATION.value in assessed_strategies(
        classify_source_strategy(vendored_root)
    )

    git_root = tmp_path / "git"
    git_root.mkdir()
    (git_root / "COMMIT").write_text("0123456789abcdef0123456789abcdef01234567")
    assert SourceStrategy.PINNED_OFFICIAL_GIT_REPOSITORY.value in assessed_strategies(
        classify_source_strategy(git_root)
    )

    non_pytorch_root = tmp_path / "non-pytorch"
    non_pytorch_root.mkdir()
    (non_pytorch_root / "model.py").write_text("import tensorflow as tf\n")
    assert SourceStrategy.EXTERNAL_PYTORCH_IMPLEMENTATION.value not in assessed_strategies(
        classify_source_strategy(non_pytorch_root)
    )
    torch_root = tmp_path / "external-torch"
    torch_root.mkdir()
    (torch_root / "model.py").write_text("import torch\n")
    assert SourceStrategy.EXTERNAL_PYTORCH_IMPLEMENTATION.value in assessed_strategies(
        classify_source_strategy(non_pytorch_root, external_pytorch_root=torch_root)
    )

    unsupported_root = tmp_path / "unsupported"
    unsupported_root.mkdir()
    assert SourceStrategy.UNSUPPORTED_OR_NON_EQUIVALENT_IMPLEMENTATION.value in assessed_strategies(
        classify_source_strategy(unsupported_root)
    )


def test_external_wrapper_assessment_uses_metadata(repo_root: Path, tmp_path: Path) -> None:
    synthetic = repo_root / "tests/skills/fixtures/synthetic_onboarding"
    upstream = synthetic / "non_pytorch_upstream"
    external = synthetic / "external_pytorch_implementation"
    assessment = classify_source_strategy(upstream, external_pytorch_root=external)[
        "external_wrapper_assessment"
    ]
    assert assessment["upstream_repository_identity"] == "synthetic-non-pytorch-upstream"
    assert assessment["external_repository_identity"] == "synthetic-external-pytorch-wrapper"
    assert assessment["architecture_mapping"] == "declared synthetic architecture mapping"
    assert "EQUIVALENCE.json" in assessment["evidence_files"]

    cloned_external = tmp_path / "external"
    subprocess.run(["cp", "-R", str(external), str(cloned_external)], check=True)
    (cloned_external / "EQUIVALENCE.json").unlink()
    unresolved = classify_source_strategy(upstream, external_pytorch_root=cloned_external)[
        "external_wrapper_assessment"
    ]
    assert unresolved["architecture_mapping"] is None
    assert unresolved["unresolved_equivalence_questions"] == (
        "semantic equivalence requires user review",
    )


def test_repository_inventory_preserves_paths_and_limits(repo_root: Path, tmp_path: Path) -> None:
    result = inspect_repository(fixture(repo_root, "unpinned_dependencies"))
    paths = {item["path"] for item in result["files"]}
    assert ".github/workflows/ci.yml" in paths
    assert "environment.yml" in paths

    root = tmp_path / "limits"
    root.mkdir()
    (root / "tiny.py").write_text("import torch\n")
    (root / "large.py").write_text("x" * 32)
    files = inspect_repository(root)["files"]
    skipped = inspect_repository(root)["skipped_files"]
    assert any(item["path"] == "large.py" for item in files)
    limited = inspect_repository(root)
    assert limited["synthetic_marker"] is False
    direct = inspect_repository(root)
    assert direct == limited
    assert not skipped


def test_iter_static_files_marks_large_and_external_symlink(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "big.txt").write_text("x" * 8)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    (root / "outside.txt").symlink_to(outside)
    from torch_dae.onboarding.inspection import iter_static_files

    with pytest.raises(OnboardingInspectionError, match="file exceeds inspection size limit"):
        iter_static_files(root, max_file_size_bytes=4)
    (root / "big.txt").unlink()
    files = iter_static_files(root, max_file_size_bytes=4)
    by_path = {item.path: item for item in files}
    assert by_path["outside.txt"].kind == "external_symlink"


def test_fixed_name_readers_reject_unsafe_files(tmp_path: Path) -> None:
    outside = tmp_path / "outside.toml"
    outside.write_text("[project]\nname='unsafe'\nversion='1.0.0'\n")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").symlink_to(outside)
    with pytest.raises(OnboardingInspectionError, match="symlinked"):
        inspect_dependencies(root)

    binary_root = tmp_path / "binary"
    binary_root.mkdir()
    (binary_root / "pyproject.toml").write_bytes(b"[project]\x00name='bad'\n")
    with pytest.raises(OnboardingInspectionError, match="binary"):
        inspect_dependencies(binary_root)


@pytest.mark.parametrize(
    "relative",
    [
        "requirements.txt",
        "sub/requirements-dev.txt",
        "setup.cfg",
        "uv.lock",
        "environment.yml",
    ],
)
def test_supported_artifact_symlinks_raise(relative: str, tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    root = tmp_path / f"repo-{relative.replace('/', '-')}"
    target = root / relative
    target.parent.mkdir(parents=True)
    target.symlink_to(outside)
    with pytest.raises(OnboardingInspectionError, match="symlinked supported artifact"):
        inspect_repository(root)


def test_missing_supported_artifact_symlink_raises(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "requirements.txt").symlink_to(tmp_path / "missing-outside.txt")
    with pytest.raises(OnboardingInspectionError, match="symlinked supported artifact"):
        inspect_repository(root)


def test_dependency_provenance_formats_are_normalized(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "sub").mkdir(parents=True)
    (root / "requirements.txt").write_text(
        "wheelpkg @ https://example.invalid/pkg.whl\n"
        "gitpkg @ git+https://example.invalid/repo.git@0123456789abcdef\n"
        "-e git+https://example.invalid/editable.git@main#egg=editablepkg\n"
        "./localpkg\n"
    )
    (root / "sub/requirements-dev.txt").write_text("pytest==8.4.0\n")
    (root / "Pipfile").write_text(
        '[packages]\nrequests = "==2.31.0"\n[dev-packages]\npytest = "==8.4.0"\n'
    )
    (root / "Pipfile.lock").write_text(
        '{"default":{"requests":{"version":"==2.31.0"}},'
        '"develop":{"pytest":{"version":"==8.4.0"}}}\n'
    )
    (root / "poetry.lock").write_text('[[package]]\nname = "numpy"\nversion = "1.26.4"\n')
    (root / "uv.lock").write_text('[[package]]\nname = "torch"\nversion = "2.2.2"\n')
    (root / "environment.yml").write_text("dependencies:\n  - python=3.10\n")
    (root / "Dockerfile").write_text("RUN pip install torchaudio==2.2.2\n")
    workflow = root / ".github/workflows/ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "jobs:\n  test:\n    strategy:\n      matrix:\n        python-version: '3.11'\n"
    )

    records = inspect_dependencies(root)["dependency_records"]
    by_raw = {item["raw_declaration"]: item for item in records}
    assert by_raw["wheelpkg @ https://example.invalid/pkg.whl"]["dependency_kind"] == "direct_url"
    assert by_raw["wheelpkg @ https://example.invalid/pkg.whl"]["direct_url"] is True
    assert (
        by_raw["gitpkg @ git+https://example.invalid/repo.git@0123456789abcdef"]["dependency_kind"]
        == "vcs"
    )
    assert by_raw["gitpkg @ git+https://example.invalid/repo.git@0123456789abcdef"]["vcs"] == "git"
    assert (
        by_raw["git+https://example.invalid/editable.git@main#egg=editablepkg"]["normalized_name"]
        == "editablepkg"
    )
    assert (
        by_raw["git+https://example.invalid/editable.git@main#egg=editablepkg"]["editable"] is True
    )
    assert by_raw["./localpkg"]["local_path"] is True
    assert by_raw["pytest==8.4.0"]["source_file"] == "sub/requirements-dev.txt"
    assert by_raw["requests==2.31.0"]["source_file"] == "Pipfile.lock"
    assert by_raw["numpy==1.26.4"]["source_file"] == "poetry.lock"
    assert by_raw["torch==2.2.2"]["source_file"] == "uv.lock"
    assert by_raw["python=3.10"]["dependency_kind"] == "conda"
    assert by_raw["torchaudio==2.2.2"]["source_section"] == "dockerfile.pip"
    assert by_raw["python==3.11"]["source_file"] == ".github/workflows/ci.yml"
    assert by_raw["python==3.11"]["source_section"] == "matrix.python-version"


@pytest.mark.parametrize(
    ("raw", "normalized_name", "constraint", "exact_version", "valid"),
    [
        ("name", "name", None, None, True),
        ("python=3.10", "python", "==3.10", "3.10", True),
        ("name==1.0", "name", "==1.0", "1.0", True),
        ("numpy<1.24", "numpy", "<1.24", None, True),
        ("numpy<=1.24", "numpy", "<=1.24", None, True),
        ("numpy>1.24", "numpy", ">1.24", None, True),
        ("numpy>=1.24", "numpy", ">=1.24", None, True),
        ("numpy!=1.24", "numpy", "!=1.24", None, True),
        ("numpy~=1.24", "numpy", "~=1.24", None, True),
        ("pytorch=1.13.1=py310_cuda11.7_cudnn8.5.0_0", "torch", "==1.13.1", "1.13.1", True),
        ("numpy=>1.24", "numpy", None, None, False),
    ],
)
def test_conda_dependency_parsing(
    raw: str,
    normalized_name: str,
    constraint: str | None,
    exact_version: str | None,
    valid: bool,
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "environment.yml").write_text(f"dependencies:\n  - {raw}\n")
    records = inspect_dependencies(root)["dependency_records"]
    record = records[0]
    assert record["raw_declaration"] == raw
    assert record["normalized_name"] == normalized_name
    assert record["constraint"] == constraint
    assert record["exact_version"] == exact_version
    assert record["valid"] is valid


def test_conda_numpy_range_is_principal_constraint(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "environment.yml").write_text("dependencies:\n  - numpy<1.24\n")
    result = generate_environment_candidates(root)
    assert result["candidates"][0]["numpy_constraint"] == "<1.24"
    assert "numpy" not in result["candidates"][0]["other_principal_dependencies"]


def test_ci_matrix_dependency_parsing(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    workflow = root / ".github/workflows/ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "jobs:\n"
        "  scalar:\n"
        "    strategy:\n"
        "      matrix:\n"
        '        python-version: "3.11"\n'
        "  inline-single:\n"
        "    strategy:\n"
        "      matrix:\n"
        "        python-version: ['3.9', '3.10']\n"
        "  inline-double:\n"
        "    strategy:\n"
        "      matrix:\n"
        '        torch-version: ["1.13.1"]\n'
        "  block:\n"
        "    strategy:\n"
        "      matrix:\n"
        "        numpy-version:\n"
        '          - "1.23"\n'
        '          - "1.24"\n'
        '        pytorch-version: ["2.0.0"]\n'
        '        torchaudio-version: ["2.0.0"]\n'
    )
    records = inspect_dependencies(root)["dependency_records"]
    by_raw_section = {(item["raw_declaration"], item["source_section"]): item for item in records}
    assert by_raw_section[("python==3.11", "matrix.python-version")]["source_file"] == (
        ".github/workflows/ci.yml"
    )
    assert ("python==3.9", "matrix.python-version") in by_raw_section
    assert ("python==3.10", "matrix.python-version") in by_raw_section
    assert ("torch==1.13.1", "matrix.torch-version") in by_raw_section
    assert ("numpy==1.23", "matrix.numpy-version") in by_raw_section
    assert ("numpy==1.24", "matrix.numpy-version") in by_raw_section
    assert ("torch==2.0.0", "matrix.pytorch-version") in by_raw_section
    assert ("torchaudio==2.0.0", "matrix.torchaudio-version") in by_raw_section
    evidence_ids = {item["evidence_id"] for item in records}
    assert len(evidence_ids) == len(records)


def test_ci_matrix_expression_reference_is_not_dependency_evidence(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    workflow = root / ".github/workflows/ci.yml"
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
        "      - run: echo ${{ matrix.python-version }}\n"
        "        env:\n"
        "          PYTHON_VERSION: ${{ matrix.python-version }}\n"
    )
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'ci-matrix-fixture'\nversion = '0.1.0'\n"
    )

    records = inspect_dependencies(root)["dependency_records"]
    python_records = [record for record in records if record["normalized_name"] == "python"]
    assert len(python_records) == 1
    assert python_records[0]["raw_declaration"] == "python==3.11"
    assert python_records[0]["exact_version"] == "3.11"
    assert python_records[0]["source_section"] == "matrix.python-version"
    assert not any(
        "${{ matrix.python-version }}" in record["raw_declaration"] for record in records
    )

    result = generate_environment_candidates(root)
    assert result["candidates"][0]["python_version"] == "3.11"
    assert not any(item.startswith("python ") for item in result["unresolved_constraints"])
    assert "dependency_conflict" not in result["candidates"][0]["predicted_failure_risks"]
    assert not any(item.startswith("python ") for item in result["candidates"][0]["uncertainty"])


def test_invalid_dependency_record_does_not_erase_valid_exact_version(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    workflow = root / ".github/workflows/ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        'jobs:\n  test:\n    strategy:\n      matrix:\n        python-version: ["3.11"]\n'
    )
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'invalid-diagnostic-fixture'\nversion = '0.1.0'\n"
    )
    (root / "environment.yml").write_text("dependencies:\n  - python=>3.9\n")

    result = generate_environment_candidates(root)
    python_records = [
        record for record in result["dependency_records"] if record["normalized_name"] == "python"
    ]
    assert any(record["valid"] is False for record in python_records)
    assert any(
        record["valid"] is True and record["exact_version"] == "3.11" for record in python_records
    )
    assert result["candidates"][0]["python_version"] == "3.11"
    assert not any(item.startswith("python ") for item in result["unresolved_constraints"])


def test_conflicting_valid_ci_exact_versions_remain_unresolved(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    workflow = root / ".github/workflows/ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        'jobs:\n  test:\n    strategy:\n      matrix:\n        python-version: ["3.10", "3.11"]\n'
    )
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'ci-conflict-fixture'\nversion = '0.1.0'\n"
    )

    result = generate_environment_candidates(root)
    assert result["candidates"][0]["python_version"] is None
    assert "python exact version conflict: 3.10, 3.11" in result["unresolved_constraints"]
    assert "dependency_conflict" in result["candidates"][0]["predicted_failure_risks"]


def test_unpinned_fixture_conda_ci_conflicts_are_unresolved(repo_root: Path) -> None:
    result = generate_environment_candidates(fixture(repo_root, "unpinned_dependencies"))
    candidate = result["candidates"][0]
    records = result["dependency_records"]
    by_raw = {item["raw_declaration"]: item for item in records}
    assert by_raw["python=3.8"]["exact_version"] == "3.8"
    assert by_raw["python==3.9"]["source_file"] == ".github/workflows/ci.yml"
    assert by_raw["python==3.10"]["source_file"] == ".github/workflows/ci.yml"
    assert by_raw["pytorch=1.12"]["normalized_name"] == "torch"
    assert by_raw["torch==1.13.1"]["source_file"] == ".github/workflows/ci.yml"
    assert candidate["python_version"] is None
    assert candidate["pytorch_version"] is None
    assert "python exact version conflict: 3.10, 3.8, 3.9" in result["unresolved_constraints"]
    assert "torch exact version conflict: 1.12, 1.13.1" in result["unresolved_constraints"]
    assert "dependency_conflict" in candidate["predicted_failure_risks"]


def test_onboarding_evidence_path_contract() -> None:
    from torch_dae.onboarding.contracts import DependencyEvidenceRecord, EvidenceItem

    EvidenceItem.model_validate(
        {
            "evidence_id": "ev-ci",
            "kind": "configuration_file",
            "claim_status": "locally_observed_behavior",
            "description": "CI evidence.",
            "source_file": ".github/workflows/ci.yml",
        }
    )
    EvidenceItem.model_validate(
        {
            "evidence_id": "ev-runtime-report",
            "kind": "source_file",
            "claim_status": "locally_observed_behavior",
            "description": "Locally observed runtime report.",
            "source_file": ".torch-dae/reports/file.json",
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
    for bad_path in (
        ".git/config",
        "../outside.py",
        "/tmp/outside.py",
    ):
        with pytest.raises(ValidationError):
            EvidenceItem.model_validate(
                {
                    "evidence_id": "ev-bad",
                    "kind": "source_file",
                    "claim_status": "locally_observed_behavior",
                    "description": "Bad path.",
                    "source_file": bad_path,
                }
            )


def test_shared_inspection_budget_and_cache(repo_root: Path, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "README.md").write_text("synthetic fixture\n")
    (root / "pyproject.toml").write_text(
        "[project]\nname='budget-fixture'\nversion='0.1.0'\ndependencies=['numpy<1.24']\n"
    )
    budget = InspectionBudget(maximum_total_files=4, maximum_total_inspected_bytes=200)
    inspect_repository(root, budget=budget)
    bytes_after_inventory = budget.bytes_read
    inspect_dependencies(root, budget=budget)
    assert budget.bytes_read == bytes_after_inventory
    cached = budget.cached_text(f"{root.resolve().as_posix()}::pyproject.toml")
    assert cached is not None and "budget-fixture" in cached

    with pytest.raises(OnboardingInspectionError, match="maximum total files"):
        many = tmp_path / "many"
        many.mkdir()
        for index in range(3):
            (many / f"file{index}.txt").write_text("x")
        inspect_repository(many, budget=InspectionBudget(maximum_total_files=2))

    with pytest.raises(OnboardingInspectionError, match="maximum total inspected bytes"):
        bytes_root = tmp_path / "bytes"
        bytes_root.mkdir()
        for index in range(3):
            (bytes_root / f"file{index}.txt").write_text("x" * 10)
        inspect_repository(bytes_root, budget=InspectionBudget(maximum_total_inspected_bytes=20))

    with pytest.raises(OnboardingInspectionError, match="file exceeds inspection size limit"):
        oversized = tmp_path / "oversized"
        oversized.mkdir()
        (oversized / "big.txt").write_text("x" * 8)
        inspect_repository(oversized, budget=InspectionBudget(maximum_single_file_bytes=4))

    separate = InspectionBudget(maximum_total_files=1)
    inspect_repository(root, budget=InspectionBudget(maximum_total_files=4))
    assert separate.files_visited == 0

    scenario_budget = InspectionBudget()
    inspect_scenario_repository(
        fixture(repo_root, "official_package"),
        scenario_id="official-package",
        budget=scenario_budget,
    )
    assert scenario_budget.files_visited > 0
    assert scenario_budget.bytes_read > 0


def test_real_git_revision_is_observed_in_environment_candidate(tmp_path: Path) -> None:
    repo = tmp_path / "git-repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(
        "[project]\nname='real-git-audio'\nversion='0.1.0'\ndependencies=['torch==2.1.0']\n"
    )
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "codex@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Codex"], cwd=repo, check=True)
    subprocess.run(["git", "add", "pyproject.toml"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=repo, check=True, capture_output=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    result = EnvironmentCandidateGenerationResult.model_validate(
        generate_environment_candidates(repo)
    )
    pinned = [
        candidate
        for candidate in result.candidates
        if candidate.installation_strategy == SourceStrategy.PINNED_OFFICIAL_GIT_REPOSITORY
    ]
    assert pinned
    assert pinned[0].source_revision == revision


def test_skill_scripts_help_and_json_smoke(repo_root: Path) -> None:
    script = repo_root / "skills/audio-model-onboarding/scripts/inspect_repository.py"
    help_result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0
    json_result = subprocess.run(
        [sys.executable, str(script), str(fixture(repo_root, "official_package")), "--json"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert json_result.returncode == 0
    assert '"synthetic_marker": true' in json_result.stdout
