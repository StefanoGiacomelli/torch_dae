"""Deterministic renderers for onboarding artifacts."""

from __future__ import annotations

from torch_dae.onboarding.contracts import AnalysisReport, EvidenceBackedClaim


def render_analysis_markdown(report: AnalysisReport) -> str:
    """Render a deterministic Markdown view of an analysis report.

    Parameters
    ----------
    report
        Validated evidence-grounded analysis report.

    Returns
    -------
    str
        Markdown containing the report sections, candidates, decisions, evidence, confidence
        summary, and recommended next mode.

    Notes
    -----
    Rendering is side-effect free and never adds facts that are absent from ``report``.
    """

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
        lines.append(f"- `{checkpoint.checkpoint_id}`")
        _field(lines, "filename", checkpoint.filename)
        _field(lines, "model variant or variant scope", checkpoint.model_variant)
        _field(lines, "source type", checkpoint.source_type)
        _field(lines, "URL or source reference", checkpoint.url)
        _field(lines, "loader", checkpoint.loader)
        for checksum in checkpoint.published_checksums:
            lines.append(
                "  - published checksum: "
                f"{checksum.algorithm.value}:{checksum.digest} "
                f"[{checksum.verification_state}] evidence={checksum.evidence_id}"
            )
            _field(lines, "checksum provenance note", checksum.provenance_note, indent=4)
        _field(lines, "legacy hash evidence", checkpoint.hash_evidence)
        _field(lines, "expression status", checkpoint.expression_status)
        _field(lines, "helper symbol", checkpoint.helper_symbol)
        if checkpoint.unresolved_components is not None:
            _field(
                lines,
                "unresolved components",
                ",".join(checkpoint.unresolved_components) or "none",
            )
        _field(lines, "access or licensing notes", checkpoint.access_or_license_notes)
        _field(lines, "claim status", checkpoint.status.value)
        _field(lines, "unresolved reason", checkpoint.unresolved_reason)
        _field(lines, "evidence IDs", ",".join(checkpoint.evidence_ids) or "none")
    if not report.checkpoint_candidates:
        lines.append("- none")
    lines.extend(["", "## Embedding Candidates", ""])
    for embedding in report.embedding_candidates:
        lines.append(f"- `{embedding.embedding_id}`")
        _field(lines, "tensor origin", embedding.tensor_origin)
        _field(lines, "semantic kind", embedding.semantic_kind)
        _field(lines, "shape semantics", embedding.shape_semantics)
        _field(lines, "batch dimension", embedding.batch_dimension)
        _field(lines, "time dimension", embedding.time_dimension)
        _field(lines, "variant scope", _scope_values(embedding.variant_ids))
        _field(lines, "checkpoint scope", _scope_values(embedding.checkpoint_ids))
        _field(
            lines,
            "decision requirement",
            "required" if embedding.requires_user_decision else "not required",
        )
        _field(lines, "status", embedding.status.value)
        _field(lines, "unresolved reason", embedding.unresolved_reason)
        _field(lines, "evidence IDs", ",".join(embedding.evidence_ids) or "none")
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
    rendered = (
        f"{claim.statement} [{claim.status.value}] "
        f"evidence={','.join(claim.evidence_ids) or 'none'}"
    )
    scope_parts = []
    if claim.variant_ids:
        scope_parts.append(f"variants={','.join(claim.variant_ids)}")
    if claim.checkpoint_ids:
        scope_parts.append(f"checkpoints={','.join(claim.checkpoint_ids)}")
    return f"{rendered} scope={';'.join(scope_parts)}" if scope_parts else rendered


def _field(
    lines: list[str],
    label: str,
    value: object | None,
    *,
    indent: int = 2,
) -> None:
    if value is not None:
        lines.append(f"{' ' * indent}- {label}: {value}")


def _scope_values(values: tuple[str, ...]) -> str:
    return ",".join(values) if values else "report-wide"
