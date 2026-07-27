from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def test_agent_skill_paths_resolve_to_canonical(repo_root: Path) -> None:
    canonical = (repo_root / "skills/audio-model-onboarding").resolve()
    codex = (repo_root / ".agents/skills/audio-model-onboarding").resolve()
    claude = (repo_root / ".claude/skills/audio-model-onboarding").resolve()
    assert codex == canonical
    assert claude == canonical
    assert (codex / "SKILL.md").resolve() == (claude / "SKILL.md").resolve()


def test_skill_front_matter_mentions_integration_boundary(repo_root: Path) -> None:
    text = (repo_root / "skills/audio-model-onboarding/SKILL.md").read_text()
    assert "project_spec.md" in text
    assert "onboarding skill" in text
    assert "the user explicitly requested `MODE: integrate`" in text
    assert "explicitly authorized production integration" in text
    assert "never create a Git commit" in text


def test_onboarding_skill_defines_required_modes_and_links(repo_root: Path) -> None:
    text = (repo_root / "skills/audio-model-onboarding/SKILL.md").read_text()
    for mode in ("analyze", "resolve-environment", "integrate", "verify", "card", "profile"):
        assert f"## `{mode}` Mode" in text
        for phrase in (
            "Purpose:",
            "Required inputs:",
            "Optional inputs:",
            "Prerequisites:",
            "Ordered procedure:",
            "Evidence requirements:",
            "Generated outputs:",
            "User-decision gates:",
            "Failure conditions:",
            "Prohibited behavior:",
            "Completion criteria:",
            "Next allowed lifecycle transition:",
        ):
            assert phrase in text
    for name in [
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
    ]:
        assert f"references/{name}" in text
        assert (repo_root / "skills/audio-model-onboarding/references" / name).exists()
    for name in ("agent-request.md", "agent-response.md"):
        assert f"templates/{name}" in text
        assert (repo_root / "skills/audio-model-onboarding/templates" / name).exists()


def test_agent_aliases_share_skill_bytes(repo_root: Path) -> None:
    canonical = repo_root / "skills/audio-model-onboarding/SKILL.md"
    codex = repo_root / ".agents/skills/audio-model-onboarding/SKILL.md"
    claude = repo_root / ".claude/skills/audio-model-onboarding/SKILL.md"
    assert codex.read_bytes() == canonical.read_bytes()
    assert claude.read_bytes() == canonical.read_bytes()
    assert not [
        path
        for path in repo_root.glob("**/audio-model-onboarding/SKILL.md")
        if path.resolve() != canonical.resolve()
    ]


def test_skill_artifact_validator_fails_on_mutated_fixture_evidence(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    synthetic = tmp_path / "synthetic_onboarding"
    shutil.copytree(repo_root / "tests/skills/fixtures/synthetic_onboarding", synthetic)
    (synthetic / "hidden_checkpoint_helper/downloads.py").write_text(
        "def get_pretrained_checkpoint_url():\n"
        "    return 'https://example.invalid/assets/changed.pth'\n"
    )
    script = repo_root / "skills/audio-model-onboarding/scripts/validate_skill_artifacts.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(repo_root),
            "--synthetic-root",
            str(synthetic),
            "--json",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "checkpoint URL was not observed" in result.stdout
