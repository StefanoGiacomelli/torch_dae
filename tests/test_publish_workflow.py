from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLISH_WORKFLOW = REPO_ROOT / ".github/workflows/publish.yml"


def test_publish_validation_installs_profiling_extra() -> None:
    text = PUBLISH_WORKFLOW.read_text(encoding="utf-8")

    assert "uv sync --all-groups --extra profiling --frozen" in text
    assert "uv run mypy src scripts" in text
