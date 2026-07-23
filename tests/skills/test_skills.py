from __future__ import annotations

from pathlib import Path


def test_agent_skill_paths_resolve_to_canonical(repo_root: Path) -> None:
    canonical = (repo_root / "skills/audio-model-onboarding").resolve()
    codex = (repo_root / ".agents/skills/audio-model-onboarding").resolve()
    claude = (repo_root / ".claude/skills/audio-model-onboarding").resolve()
    assert codex == canonical
    assert claude == canonical
    assert (codex / "SKILL.md").resolve() == (claude / "SKILL.md").resolve()


def test_skill_front_matter_mentions_phase00_boundary(repo_root: Path) -> None:
    text = (repo_root / "skills/audio-model-onboarding/SKILL.md").read_text()
    assert "project_spec.md" in text
    assert "Phase 00 scaffold" in text
    assert "Do not invent model metadata" in text
