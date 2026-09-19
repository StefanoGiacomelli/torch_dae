from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_wave_b_public_guides_exist() -> None:
    required = (
        "docs/skill/prompt-library.md",
        "docs/tutorials/profiling.md",
        "docs/user-guide/technical-cards.md",
        "skills/audio-model-profiling/templates/README.md",
        "skills/audio-model-profiling/templates/agent-request.md",
    )
    for relative in required:
        assert (ROOT / relative).is_file(), relative


def test_profiling_documentation_describes_implemented_contract() -> None:
    corpus = "\n".join(
        _read(path)
        for path in (
            "docs/profiling/overview.md",
            "docs/profiling/protocol.md",
            "docs/profiling/technical-cards.md",
            "docs/profiling/contributing.md",
        )
    )
    stale_claims = (
        "Profiling v1 is not implemented yet",
        "Planned selectors",
        "These commands remain specification-only until implementation lands",
        (
            "CI wiring for pull requests that promote cards into "
            "`technical_cards/` is not part of this pass"
        ),
    )
    for claim in stale_claims:
        assert claim not in corpus

    for required in (
        "audio-inference-v1",
        "--allow-privileged-energy",
        "campaign-result.json",
        "coverage_complete",
        "unaccounted_components",
        "allow_pickle=False",
    ):
        assert required in corpus


def test_prompt_library_preserves_phase_boundaries() -> None:
    prompt_library = _read("docs/skill/prompt-library.md")
    for mode in (
        "MODE: analyze",
        "MODE: resolve-environment",
        "MODE: integrate",
        "MODE: verify",
        "MODE: card",
        "MODE: resolve",
        "MODE: plan",
        "MODE: profile",
        "MODE: validate",
        "MODE: finalize",
    ):
        assert mode in prompt_library

    assert "one phase at a time" in prompt_library
    assert "Do not promote" in prompt_library or "do not promote" in prompt_library


def test_profiling_agent_template_has_required_safety_controls() -> None:
    template = _read("skills/audio-model-profiling/templates/agent-request.md")
    for token in (
        "MODEL_CARD_ID:",
        "DEVICES:",
        "ENERGY:",
        "OUTPUT_DIR:",
        "ALLOW_PRIVILEGED_ENERGY:",
        "CANDIDATE_DIRECTORY:",
    ):
        assert token in template

    assert "never mutate it" in template
    assert "Never write candidate evidence directly" in template
    assert "Never request privileged hardware counters" in template
    assert "do not create a Git commit" in template


def test_wave_b_index_surfaces_new_user_paths() -> None:
    index = _read("docs/index.md")
    for target in (
        "tutorials/profiling",
        "user-guide/technical-cards",
        "skill/prompt-library",
    ):
        assert target in index
