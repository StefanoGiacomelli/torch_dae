from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from torch_dae.onboarding.contracts import (
    AnalysisReport,
    FailureClassification,
    RecommendedNextMode,
    SkillEvaluationScenario,
    SourceStrategy,
)
from torch_dae.onboarding.evaluation import evaluate_analysis_report
from torch_dae.onboarding.inspection import inspect_scenario_repository


def observation_for(repo_root: Path, scenario_id: str, *, fixture_root: Path | None = None):
    synthetic_root = fixture_root or repo_root / "tests/skills/fixtures/synthetic_onboarding"
    return inspect_scenario_repository(
        synthetic_root / scenario_id.replace("-", "_"),
        scenario_id=scenario_id,
        external_pytorch_root=synthetic_root / "external_pytorch_implementation"
        if scenario_id == "non-pytorch-upstream"
        else None,
    )


def test_evaluation_harness_accepts_all_golden_scenarios(repo_root: Path) -> None:
    expectations = repo_root / "tests/skills/scenario_expectations"
    golden = repo_root / "tests/skills/golden"
    for path in sorted(expectations.glob("*.json")):
        scenario = SkillEvaluationScenario.model_validate_json(path.read_text())
        report = AnalysisReport.model_validate_json(
            (golden / f"{scenario.scenario_id}.analysis.json").read_text()
        )
        assert (
            evaluate_analysis_report(
                scenario, report, observation_for(repo_root, scenario.scenario_id)
            )
            == ()
        )


def test_evaluation_harness_reports_missing_decision_and_failure_classification(
    repo_root: Path,
) -> None:
    data = json.loads(
        (repo_root / "tests/fixtures/valid/analysis-report.synthetic.json").read_text()
    )
    data["open_questions"] = []
    data["source_strategy_candidates"][0]["user_decision_required"] = False
    data["confidence_summary"]["unresolved_count"] -= 1
    report = AnalysisReport.model_validate(data)
    scenario = SkillEvaluationScenario(
        scenario_id="unsupported",
        synthetic=True,
        expected_source_strategy=SourceStrategy.EXTERNAL_PYTORCH_IMPLEMENTATION,
        requires_user_decision=True,
        expected_failure_classification=FailureClassification.INSUFFICIENT_EVIDENCE,
        expected_checkpoint_ids=("missing-checkpoint",),
        expected_embedding_decision=True,
        expected_next_mode=RecommendedNextMode.ANALYZE,
    )
    failures = evaluate_analysis_report(scenario, report)
    assert "missing expected source strategy: external_pytorch_implementation" in failures
    assert "expected user decision gate was not produced" in failures
    assert "expected failure classification not represented: insufficient_evidence" in failures
    assert "missing expected checkpoint candidates: ['missing-checkpoint']" in failures
    assert "expected embedding decision gate was not produced" in failures


def test_evaluation_harness_reports_next_mode_mutation(repo_root: Path) -> None:
    data = json.loads(
        (repo_root / "tests/skills/golden/official-package.analysis.json").read_text()
    )
    data["recommended_next_mode"] = "analyze"
    report = AnalysisReport.model_validate(data)
    scenario = SkillEvaluationScenario.model_validate_json(
        (repo_root / "tests/skills/scenario_expectations/official-package.json").read_text()
    )
    assert (
        "unexpected recommended next mode: analyze != resolve-environment"
        in evaluate_analysis_report(scenario, report)
    )


def test_grounded_evaluation_reports_fixture_mutations(repo_root: Path, tmp_path: Path) -> None:
    synthetic = repo_root / "tests/skills/fixtures/synthetic_onboarding"
    clone = tmp_path / "synthetic_onboarding"
    shutil.copytree(synthetic, clone)
    golden = repo_root / "tests/skills/golden"
    expectations = repo_root / "tests/skills/scenario_expectations"

    mutations = {
        "official-package": (
            clone / "official_package/pyproject.toml",
            "[project]\nname='changed-package'\nversion='1.2.3'\ndependencies=[]\n",
            "package name disagrees with inspection",
        ),
        "pinned-git": (
            clone / "pinned_git/audio_git/model.py",
            "import torch\nclass Changed(torch.nn.Module):\n    pass\n",
            "model variant was not observed",
        ),
        "hidden-checkpoint-helper": (
            clone / "hidden_checkpoint_helper/downloads.py",
            "def get_pretrained_checkpoint_url():\n    return 'https://example.invalid/assets/changed.pth'\n",
            "checkpoint URL was not observed",
        ),
        "unpinned-dependencies": (
            clone / "unpinned_dependencies/environment.yml",
            "name: synthetic\n",
            "dependency declaration was not observed: python=3.8",
        ),
        "ambiguous-embeddings": (
            clone / "ambiguous_embeddings/model.py",
            "import torch\n"
            "class AmbiguousEmbeddingModel(torch.nn.Module):\n"
            "    def forward(self, waveform):\n"
            "        return {'logits': waveform}\n",
            "embedding tensor origin was not observed",
        ),
        "non-pytorch-upstream": (
            clone / "external_pytorch_implementation/model.py",
            "class ExternalTorchModel:\n    pass\n",
            "expected source strategy was not observed",
        ),
        "minimal-vendoring": (
            clone / "minimal_vendoring/VENDORING.md",
            "Package installation is unsuitable.\n",
            "expected source strategy was not observed",
        ),
    }

    for scenario_id, (path, replacement, expected_failure) in mutations.items():
        path.write_text(replacement)
        scenario = SkillEvaluationScenario.model_validate_json(
            (expectations / f"{scenario_id}.json").read_text()
        )
        report = AnalysisReport.model_validate_json(
            (golden / f"{scenario_id}.analysis.json").read_text()
        )
        failures = evaluate_analysis_report(
            scenario,
            report,
            observation_for(repo_root, scenario_id, fixture_root=clone),
        )
        assert any(expected_failure in failure for failure in failures), failures


def test_grounded_evaluation_rejects_missing_evidence_symbol(repo_root: Path) -> None:
    data = json.loads(
        (repo_root / "tests/skills/golden/official-package.analysis.json").read_text()
    )
    data["evidence_items"][0]["source_line_or_symbol"] = "definitely_missing_symbol"
    report = AnalysisReport.model_validate(data)
    scenario = SkillEvaluationScenario.model_validate_json(
        (repo_root / "tests/skills/scenario_expectations/official-package.json").read_text()
    )
    failures = evaluate_analysis_report(
        scenario, report, observation_for(repo_root, "official-package")
    )
    assert any("evidence symbol was not observed" in failure for failure in failures)


def test_grounded_evaluation_rejects_wrong_symbol_source_file(repo_root: Path) -> None:
    data = json.loads(
        (repo_root / "tests/skills/golden/official-package.analysis.json").read_text()
    )
    for item in data["evidence_items"]:
        if item["source_file"] == "synthetic_model/model.py":
            item["source_line_or_symbol"] = "ClearAudioModel"
            item["source_file"] = "README.md"
            break
    report = AnalysisReport.model_validate(data)
    scenario = SkillEvaluationScenario.model_validate_json(
        (repo_root / "tests/skills/scenario_expectations/official-package.json").read_text()
    )
    failures = evaluate_analysis_report(
        scenario, report, observation_for(repo_root, "official-package")
    )
    assert any("evidence symbol was not observed" in failure for failure in failures)


def test_grounded_evaluation_rejects_unrelated_source_strategy_evidence(repo_root: Path) -> None:
    data = json.loads(
        (repo_root / "tests/skills/golden/official-package.analysis.json").read_text()
    )
    data["source_strategy_candidates"][0]["evidence_ids"] = ["ev-readme-md"]
    report = AnalysisReport.model_validate(data)
    scenario = SkillEvaluationScenario.model_validate_json(
        (repo_root / "tests/skills/scenario_expectations/official-package.json").read_text()
    )
    failures = evaluate_analysis_report(
        scenario, report, observation_for(repo_root, "official-package")
    )
    assert any(
        "source strategy evidence did not match inspection" in failure for failure in failures
    )


def hidden_checkpoint_report(repo_root: Path) -> dict[str, object]:
    return json.loads(
        (repo_root / "tests/skills/golden/hidden-checkpoint-helper.analysis.json").read_text()
    )


def hidden_checkpoint_failures(repo_root: Path, data: dict[str, object]) -> tuple[str, ...]:
    report = AnalysisReport.model_validate(data)
    scenario = SkillEvaluationScenario.model_validate_json(
        (repo_root / "tests/skills/scenario_expectations/hidden-checkpoint-helper.json").read_text()
    )
    return evaluate_analysis_report(
        scenario, report, observation_for(repo_root, "hidden-checkpoint-helper")
    )


def test_grounded_evaluation_accepts_candidate_specific_checkpoint_hash(
    repo_root: Path,
) -> None:
    assert hidden_checkpoint_failures(repo_root, hidden_checkpoint_report(repo_root)) == ()


def test_grounded_evaluation_rejects_hash_from_other_checkpoint_candidate(
    repo_root: Path,
) -> None:
    data = hidden_checkpoint_report(repo_root)
    data["checkpoint_candidates"][0]["hash_evidence"] = (
        "1111111111111111111111111111111111111111111111111111111111111111"
    )
    failures = hidden_checkpoint_failures(repo_root, data)
    assert any("hash was not associated" in failure for failure in failures), failures


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("helper_symbol", "wrong_helper", "checkpoint helper symbol was not observed"),
        ("expression_status", "partial", "checkpoint helper expression status mismatch"),
        (
            "unresolved_components",
            ["MODEL_HOST"],
            "checkpoint helper unresolved components mismatch",
        ),
        ("url", "https://example.invalid/assets/other.pth", "checkpoint URL was not observed"),
        ("filename", "other.pth", "checkpoint filename was not observed"),
        (
            "hash_evidence",
            "1111111111111111111111111111111111111111111111111111111111111111",
            "checkpoint hash was not associated with observed candidate",
        ),
    ],
)
def test_grounded_evaluation_rejects_checkpoint_helper_mutations(
    repo_root: Path,
    field: str,
    value: object,
    expected: str,
) -> None:
    data = hidden_checkpoint_report(repo_root)
    data["checkpoint_candidates"][0][field] = value
    failures = hidden_checkpoint_failures(repo_root, data)
    assert any(expected in failure for failure in failures), failures


def test_grounded_evaluation_rejects_missing_checkpoint_helper_symbol(repo_root: Path) -> None:
    data = hidden_checkpoint_report(repo_root)
    data["checkpoint_candidates"][0].pop("helper_symbol")
    data["checkpoint_candidates"][0].pop("expression_status")
    data["checkpoint_candidates"][0].pop("unresolved_components")
    failures = hidden_checkpoint_failures(repo_root, data)
    assert any("checkpoint helper symbol missing" in failure for failure in failures), failures


def checkpoint_helper_fixture_failures(
    repo_root: Path,
    tmp_path: Path,
    source: str,
    *,
    hash_evidence: str | None,
    unresolved_components: list[str],
) -> tuple[str, ...]:
    fixture_root = tmp_path / "synthetic_onboarding"
    scenario_root = fixture_root / "hidden_checkpoint_helper"
    shutil.copytree(
        repo_root / "tests/skills/fixtures/synthetic_onboarding/hidden_checkpoint_helper",
        scenario_root,
    )
    (scenario_root / "downloads.py").write_text(source)
    data = hidden_checkpoint_report(repo_root)
    data["checkpoint_candidates"][0]["hash_evidence"] = hash_evidence
    data["checkpoint_candidates"][0]["unresolved_components"] = unresolved_components
    report = AnalysisReport.model_validate(data)
    scenario = SkillEvaluationScenario.model_validate_json(
        (repo_root / "tests/skills/scenario_expectations/hidden-checkpoint-helper.json").read_text()
    )
    return evaluate_analysis_report(
        scenario,
        report,
        observation_for(repo_root, "hidden-checkpoint-helper", fixture_root=fixture_root),
    )


def test_grounded_evaluation_rejects_hash_associated_only_with_other_helper(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    failures = checkpoint_helper_fixture_failures(
        repo_root,
        tmp_path,
        "def get_pretrained_checkpoint_url():\n"
        "    base = 'https://example.invalid/assets'\n"
        "    filename = 'hidden-audio-model.pth'\n"
        "    return f'{base}/{filename}'\n\n"
        "def get_other_checkpoint_url():\n"
        "    sha256 = '0000000000000000000000000000000000000000000000000000000000000000'\n"
        "    return 'https://example.invalid/assets/other.pth'\n",
        hash_evidence="0000000000000000000000000000000000000000000000000000000000000000",
        unresolved_components=["hash association"],
    )
    assert any("hash was not associated" in failure for failure in failures), failures


def test_grounded_evaluation_rejects_repository_global_hash(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    failures = checkpoint_helper_fixture_failures(
        repo_root,
        tmp_path,
        "def get_pretrained_checkpoint_url():\n"
        "    base = 'https://example.invalid/assets'\n"
        "    filename = 'hidden-audio-model.pth'\n"
        "    return f'{base}/{filename}'\n\n"
        "UNRELATED_SHA256 = "
        "'0000000000000000000000000000000000000000000000000000000000000000'\n",
        hash_evidence="0000000000000000000000000000000000000000000000000000000000000000",
        unresolved_components=["hash association"],
    )
    assert any("hash was not associated" in failure for failure in failures), failures


def test_grounded_evaluation_accepts_unresolved_helper_hash_when_omitted(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    source = (
        "def get_pretrained_checkpoint_url():\n"
        "    base = 'https://example.invalid/assets'\n"
        "    filename = 'hidden-audio-model.pth'\n"
        "    return f'{base}/{filename}'\n"
    )
    failures = checkpoint_helper_fixture_failures(
        repo_root,
        tmp_path,
        source,
        hash_evidence=None,
        unresolved_components=["hash association"],
    )
    assert failures == ()


def test_grounded_evaluation_rejects_claimed_unresolved_helper_hash(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    source = (
        "def get_pretrained_checkpoint_url():\n"
        "    base = 'https://example.invalid/assets'\n"
        "    filename = 'hidden-audio-model.pth'\n"
        "    return f'{base}/{filename}'\n"
    )
    failures = checkpoint_helper_fixture_failures(
        repo_root,
        tmp_path,
        source,
        hash_evidence="0000000000000000000000000000000000000000000000000000000000000000",
        unresolved_components=["hash association"],
    )
    assert any("hash was not associated" in failure for failure in failures), failures


@pytest.mark.integration
def test_real_git_grounded_scenario(repo_root: Path, tmp_path: Path) -> None:
    root = tmp_path / "real_git_audio"
    package = root / "real_git_audio"
    package.mkdir(parents=True)
    (root / "README.md").write_text(
        "Synthetic real-Git fixture for Phase 02 tests. It is not a real model.\n"
    )
    (root / "pyproject.toml").write_text(
        "[project]\n"
        "name = 'real-git-audio'\n"
        "version = '0.1.0'\n"
        "requires-python = '>=3.10'\n"
        "dependencies = ['torch==2.1.0']\n"
    )
    (package / "__init__.py").write_text("from .model import AudioGitModel\n")
    (package / "model.py").write_text(
        "class AudioGitModel:\n"
        "    def forward(self, waveform, sample_rate=None):\n"
        "        return {'embedding': waveform, 'sample_rate': sample_rate}\n"
    )
    (root / "SCENARIO.json").write_text(
        json.dumps(
            {
                "scenario_id": "real-git-grounded",
                "synthetic": True,
                "expected_source_strategy": "pinned_official_git_repository",
                "requires_user_decision": True,
                "expected_next_mode": "resolve-environment",
            }
        )
    )
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Phase 02",
            "-c",
            "user.email=phase02@example.invalid",
            "commit",
            "-m",
            "create synthetic real git fixture",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    observation = inspect_scenario_repository(root, scenario_id="real-git-grounded")
    source_assessment = observation.source_strategy_assessment
    report_data = json.loads(
        (repo_root / "tests/fixtures/valid/analysis-report.synthetic.json").read_text()
    )
    report_data["report_id"] = "real-git-grounded-analysis"
    report_data["repository"]["repository_name"] = "real_git_audio"
    report_data["repository"]["package_name"] = "real-git-audio"
    report_data["repository"]["revision_inspected"] = sha
    report_data["revision"] = sha
    report_data["checkpoint_candidates"] = []
    report_data["dependency_evidence"]["claims"] = [
        {
            "statement": "Observed dependency: torch==2.1.0",
            "status": "locally_observed_behavior",
            "evidence_ids": ["ev-package"],
            "rationale": None,
        },
        {
            "statement": "Observed Python constraint: >=3.10",
            "status": "locally_observed_behavior",
            "evidence_ids": ["ev-package"],
            "rationale": None,
        },
    ]
    report_data["source_strategy_candidates"] = [
        {
            "strategy": "pinned_official_git_repository",
            "status": "unresolved_ambiguity",
            "rationale": (
                "A real immutable Git revision was inspected, but officiality still "
                "requires primary upstream evidence."
            ),
            "evidence_ids": ["ev-git-revision"],
            "user_decision_required": True,
            "unresolved_reason": "Primary upstream evidence still required.",
        }
    ]
    report_data["open_questions"][0]["evidence_ids"] = ["ev-git-revision"]
    report_data["evidence_items"] = [
        {
            "evidence_id": "ev-git-revision",
            "kind": "runtime_observation",
            "claim_status": "locally_observed_behavior",
            "description": "Observed immutable Git revision for the synthetic repository.",
            "source_file": None,
            "source_line_or_symbol": None,
            "url": None,
            "revision": sha,
            "rationale": None,
        },
        {
            "evidence_id": "ev-package",
            "kind": "package_metadata",
            "claim_status": "locally_observed_behavior",
            "description": "Observed local package metadata from pyproject.toml.",
            "source_file": "pyproject.toml",
            "source_line_or_symbol": None,
            "url": None,
            "revision": sha,
            "rationale": None,
            "package_name": "real-git-audio",
            "package_version": "0.1.0",
        },
    ]
    report_data["confidence_summary"] = {
        "verified_fact_count": 0,
        "locally_observed_count": 4,
        "inference_count": 0,
        "unresolved_count": 4,
        "unsupported_claim_count": 0,
    }
    report = AnalysisReport.model_validate(report_data)
    scenario = SkillEvaluationScenario.model_validate_json((root / "SCENARIO.json").read_text())

    assert evaluate_analysis_report(scenario, report, observation) == ()
    assert source_assessment["observed_repository_revision"]["revision"] == sha
    assert source_assessment["observed_packaging"]["package_name"] == "real-git-audio"
    assert source_assessment["officiality_status"] == "unresolved"
    assert report.repository.release_or_tag_evidence == ()
    assert all(
        candidate.source_revision == sha
        for candidate in observation.environment_candidates.candidates
        if candidate.installation_strategy.value == "pinned_official_git_repository"
    )
