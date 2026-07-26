"""Phase 02 model-onboarding contracts and deterministic static utilities."""

from torch_dae.onboarding.contracts import (
    AnalysisReport,
    EnvironmentCandidate,
    EnvironmentResolutionReport,
    EvidenceItem,
    FailureClassification,
    SourceStrategy,
)
from torch_dae.onboarding.rendering import render_analysis_markdown

__all__ = [
    "AnalysisReport",
    "EnvironmentCandidate",
    "EnvironmentResolutionReport",
    "EvidenceItem",
    "FailureClassification",
    "SourceStrategy",
    "render_analysis_markdown",
]
