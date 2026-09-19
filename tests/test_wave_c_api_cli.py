from __future__ import annotations

from pathlib import Path


def test_wave_c_reference_pages_are_exposed(repo_root: Path) -> None:
    index = (repo_root / "docs/index.md").read_text(encoding="utf-8")
    cli = (repo_root / "docs/reference/cli.md").read_text(encoding="utf-8")
    runtime_api = (repo_root / "docs/reference/panns-runtime-api.md").read_text(encoding="utf-8")

    assert "reference/panns-runtime-api" in index
    assert "torch-dae --version" in cli
    assert "CI-protected" in cli
    assert "repository artifact" in " ".join(cli.split())
    assert "from_pretrained()" in runtime_api
    assert "[B, 527]" in runtime_api
    assert "[B, 2048]" in runtime_api
    assert "does **not** automatically" in runtime_api


def test_panns_runtime_source_remains_accepted_evidence(repo_root: Path) -> None:
    handoff = (
        repo_root / "onboarding_reports/panns-audioset-three-tuple/integrate/handoff.json"
    ).read_text(encoding="utf-8")
    assert '"path": "src/torch_dae/models/panns/model.py"' in handoff

    source = (repo_root / "src/torch_dae/models/panns/model.py").read_text(encoding="utf-8")
    assert "class PannsCnn14_16kMap0438" in source
    assert "class PannsResNet38Map0434" in source
    assert "class PannsWavegramLogmelCnn14Map0439" in source
