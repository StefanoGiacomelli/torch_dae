"""Deterministic renderers for Phase 02 onboarding artifacts."""

from __future__ import annotations

from torch_dae.onboarding.contracts import AnalysisReport, EvidenceBackedClaim


def render_analysis_markdown(report: AnalysisReport) -> str:
    """Render a technical analysis report without adding facts absent from JSON."""

    lines: list[str] = [
        f"# Technical Analysis Report: {report.report_id}",
        "",
        "## Repository Identity",
        "",
        f"- Repository: {report.repository.repository_name or 'unresolved'}",
        f"- Revision: {report.revision or 'unresolved'}",
        f"- Official status: {_claim(report.official_status)}",
        "",
    ]
    for title, section in [
        ("Scientific Identity", report.scientific_identity),
        ("Architecture", report.architecture),
        ("Runtime Interface", report.runtime_interface),
        ("Preprocessing", report.preprocessing),
        ("Outputs", report.outputs),
        ("Dependency Evidence", report.dependency_evidence),
        ("Environment Evidence", report.environment_evidence),
    ]:
        lines.extend([f"## {title}", "", section.summary, ""])
        for claim in section.claims:
            lines.append(f"- {_claim(claim)}")
        if section.claims:
            lines.append("")
    lines.extend(["## Variants", ""])
    for variant in report.variants:
        lines.append(
            f"- `{variant.variant_id}`: {variant.name} [{variant.status.value}] "
            f"evidence={','.join(variant.evidence_ids) or 'none'}"
        )
    if not report.variants:
        lines.append("- none")
    lines.extend(["", "## Checkpoint Candidates", ""])
    for checkpoint in report.checkpoint_candidates:
        lines.append(
            f"- `{checkpoint.checkpoint_id}`: {checkpoint.source_type} "
            f"[{checkpoint.status.value}] evidence={','.join(checkpoint.evidence_ids) or 'none'}"
        )
    if not report.checkpoint_candidates:
        lines.append("- none")
    lines.extend(["", "## Embedding Candidates", ""])
    for embedding in report.embedding_candidates:
        decision = " decision-required" if embedding.requires_user_decision else ""
        lines.append(
            f"- `{embedding.embedding_id}`: {embedding.semantic_kind} "
            f"[{embedding.status.value}]{decision} "
            f"evidence={','.join(embedding.evidence_ids) or 'none'}"
        )
    if not report.embedding_candidates:
        lines.append("- none")
    lines.extend(["", "## Source Strategy Candidates", ""])
    for source_strategy in report.source_strategy_candidates:
        decision = " decision-required" if source_strategy.user_decision_required else ""
        lines.append(
            f"- `{source_strategy.strategy.value}` [{source_strategy.status.value}]{decision}: "
            f"{source_strategy.rationale} "
            f"evidence={','.join(source_strategy.evidence_ids) or 'none'}"
        )
    if not report.source_strategy_candidates:
        lines.append("- none")
    lines.extend(["", "## Open Questions", ""])
    for question in report.open_questions:
        lines.append(
            f"- `{question.question_id}` {question.classification.value}: "
            f"{question.description} evidence={','.join(question.evidence_ids) or 'none'}"
        )
    if not report.open_questions:
        lines.append("- none")
    lines.extend(["", "## Decisions", ""])
    for decision_record in report.decisions:
        selected = decision_record.selected_option or "unresolved"
        lines.append(
            f"- `{decision_record.decision_id}` [{decision_record.status.value}]: "
            f"{decision_record.decision} selected={selected} "
            f"evidence={','.join(decision_record.evidence_ids) or 'none'}"
        )
    if not report.decisions:
        lines.append("- none")
    lines.extend(["", "## Evidence", ""])
    for evidence in report.evidence_items:
        location = evidence.source_file or (str(evidence.url) if evidence.url else "none")
        lines.append(
            f"- `{evidence.evidence_id}` {evidence.kind.value} "
            f"[{evidence.claim_status.value}] {location}: {evidence.description}"
        )
    lines.extend(
        [
            "",
            "## Confidence Summary",
            "",
            f"- verified_fact_count: {report.confidence_summary.verified_fact_count}",
            f"- locally_observed_count: {report.confidence_summary.locally_observed_count}",
            f"- inference_count: {report.confidence_summary.inference_count}",
            f"- unresolved_count: {report.confidence_summary.unresolved_count}",
            f"- unsupported_claim_count: {report.confidence_summary.unsupported_claim_count}",
            "",
            f"Recommended next mode: `{report.recommended_next_mode.value}`",
            "",
        ]
    )
    return "\n".join(lines)


def _claim(claim: EvidenceBackedClaim) -> str:
    return (
        f"{claim.statement} [{claim.status.value}] "
        f"evidence={','.join(claim.evidence_ids) or 'none'}"
    )
