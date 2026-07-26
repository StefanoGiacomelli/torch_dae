from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from torch_dae.onboarding.contracts import AnalysisReport
from torch_dae.onboarding.rendering import render_analysis_markdown


def test_analysis_markdown_renderer_is_deterministic(repo_root: Path) -> None:
    report = AnalysisReport.model_validate_json(
        (repo_root / "tests/fixtures/valid/analysis-report.synthetic.json").read_text()
    )
    first = render_analysis_markdown(report)
    second = render_analysis_markdown(report)
    assert first == second
    assert "## Source Strategy Candidates" in first
    assert "`official_package`" in first


def test_render_analysis_report_script_check(repo_root: Path) -> None:
    script = repo_root / "skills/audio-model-onboarding/scripts/render_analysis_report.py"
    report = repo_root / "skills/audio-model-onboarding/templates/technical-analysis-report.json"
    markdown = repo_root / "skills/audio-model-onboarding/templates/technical-analysis-report.md"
    result = subprocess.run(
        [sys.executable, str(script), str(report), "--check", str(markdown), "--json"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"valid": true' in result.stdout
