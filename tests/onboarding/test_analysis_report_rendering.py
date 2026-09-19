from __future__ import annotations

import json
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


def test_analysis_markdown_renders_material_candidate_fields(repo_root: Path) -> None:
    data = json.loads(
        (repo_root / "tests/fixtures/valid/analysis-report.synthetic.json").read_text()
    )
    data["evidence_items"].append(
        {
            "evidence_id": "ev-host",
            "kind": "official_documentation",
            "claim_status": "verified_upstream_fact",
            "description": "Authoritative checkpoint host metadata.",
            "url": "https://example.invalid/record",
        }
    )
    data["variants"] = [
        {
            "variant_id": "variant-a",
            "name": "Variant A",
            "status": "locally_observed_behavior",
            "evidence_ids": ["ev-static"],
            "unresolved_reason": None,
        }
    ]
    data["outputs"]["claims"] = [
        {
            "statement": "Output mapping is candidate-specific.",
            "status": "locally_observed_behavior",
            "evidence_ids": ["ev-static"],
            "variant_ids": ["variant-a"],
            "checkpoint_ids": ["clear-audio"],
            "rationale": None,
        }
    ]
    checkpoint = data["checkpoint_candidates"][0]
    checkpoint.update(
        {
            "model_variant": "variant-a",
            "loader": "load_checkpoint",
            "published_checksums": [
                {
                    "algorithm": "md5",
                    "digest": "a" * 32,
                    "evidence_id": "ev-host",
                    "verification_state": "published_not_locally_verified",
                    "provenance_note": "Host metadata only.",
                }
            ],
            "access_or_license_notes": "Authentication may be required.",
            "helper_symbol": "checkpoint_url",
            "expression_status": "resolved",
            "unresolved_components": [],
        }
    )
    data["embedding_candidates"] = [
        {
            "embedding_id": "embedding-a",
            "tensor_origin": "Encoder.forward",
            "semantic_kind": "pooled_representation",
            "shape_semantics": "B,D",
            "batch_dimension": "B",
            "time_dimension": None,
            "status": "locally_observed_behavior",
            "evidence_ids": ["ev-static"],
            "variant_ids": ["variant-a"],
            "checkpoint_ids": ["clear-audio"],
            "requires_user_decision": True,
            "unresolved_reason": None,
        }
    ]
    data["confidence_summary"]["verified_fact_count"] = 1
    data["confidence_summary"]["locally_observed_count"] = 5

    markdown = render_analysis_markdown(AnalysisReport.model_validate(data))

    assert "scope=variants=variant-a;checkpoints=clear-audio" in markdown
    assert "published checksum: md5:" in markdown
    assert "loader: load_checkpoint" in markdown
    assert "expression status: resolved" in markdown
    assert "unresolved components: none" in markdown
    assert "tensor origin: Encoder.forward" in markdown
    assert "shape semantics: B,D" in markdown
    assert "variant scope: variant-a" in markdown
    assert "checkpoint scope: clear-audio" in markdown
    assert "decision requirement: required" in markdown
