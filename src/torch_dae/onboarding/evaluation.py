"""Deterministic validation harness for synthetic onboarding workflow skill outputs."""

from __future__ import annotations

from torch_dae.onboarding.contracts import (
    AnalysisReport,
    FailureClassification,
    OpenQuestionClassification,
    ScenarioInspectionResult,
    SkillEvaluationScenario,
)


def evaluate_analysis_report(
    scenario: SkillEvaluationScenario,
    report: AnalysisReport,
    observation: ScenarioInspectionResult | None = None,
) -> tuple[str, ...]:
    """Return deterministic evaluation failures for a synthetic scenario."""

    failures: list[str] = []
    if not report.evidence_items:
        failures.append("analysis report contains no evidence items")
    if not report.source_strategy_candidates:
        failures.append("analysis report contains no source strategy candidates")

    if scenario.expected_source_strategy is not None:
        strategies = {candidate.strategy for candidate in report.source_strategy_candidates}
        if scenario.expected_source_strategy not in strategies:
            failures.append(
                f"missing expected source strategy: {scenario.expected_source_strategy.value}"
            )
    has_decision_gate = any(
        question.classification == OpenQuestionClassification.NEEDS_USER_DECISION
        for question in report.open_questions
    )
    has_candidate_gate = any(
        candidate.requires_user_decision for candidate in report.embedding_candidates
    ) or any(candidate.user_decision_required for candidate in report.source_strategy_candidates)
    if scenario.requires_user_decision and not (has_decision_gate or has_candidate_gate):
        failures.append("expected user decision gate was not produced")
    if scenario.expected_failure_classification is not None:
        represented = _failure_classification_represented(
            scenario.expected_failure_classification,
            report,
        )
        if not represented:
            failures.append(
                "expected failure classification not represented: "
                f"{scenario.expected_failure_classification.value}"
            )
    missing_checkpoints = sorted(
        set(scenario.expected_checkpoint_ids)
        - {candidate.checkpoint_id for candidate in report.checkpoint_candidates}
    )
    if missing_checkpoints:
        failures.append(f"missing expected checkpoint candidates: {missing_checkpoints}")
    if scenario.expected_embedding_decision:
        has_embedding_decision = any(
            candidate.requires_user_decision for candidate in report.embedding_candidates
        )
        if not has_embedding_decision:
            failures.append("expected embedding decision gate was not produced")
    if (
        scenario.expected_next_mode is not None
        and report.recommended_next_mode != scenario.expected_next_mode
    ):
        failures.append(
            "unexpected recommended next mode: "
            f"{report.recommended_next_mode.value} != {scenario.expected_next_mode.value}"
        )
    if observation is not None:
        failures.extend(_evaluate_against_observation(scenario, report, observation))
    return tuple(failures)


def _evaluate_against_observation(
    scenario: SkillEvaluationScenario,
    report: AnalysisReport,
    observation: ScenarioInspectionResult,
) -> list[str]:
    failures: list[str] = []
    if observation.scenario_id != scenario.scenario_id:
        failures.append(
            f"observation scenario mismatch: {observation.scenario_id} != {scenario.scenario_id}"
        )
    observed_files = {
        item["path"] for item in observation.repository_inventory.get("files", ())
    } | {item["path"] for item in observation.repository_inventory.get("skipped_files", ())}
    for item in report.evidence_items:
        if item.source_file and item.source_file not in observed_files:
            failures.append(
                f"evidence source file was not inspected: {item.evidence_id} -> {item.source_file}"
            )
        if item.source_line_or_symbol and not _symbol_observed(
            observation, item.source_file, item.source_line_or_symbol
        ):
            failures.append(
                "evidence symbol was not observed: "
                f"{item.evidence_id} -> {item.source_file}:{item.source_line_or_symbol}"
            )

    package_name = _observed_package_name(observation)
    if (
        report.repository.package_name
        and package_name
        and report.repository.package_name != package_name
    ):
        failures.append(
            "package name disagrees with inspection: "
            f"{report.repository.package_name} != {package_name}"
        )

    observed_revision = observation.source_strategy_assessment.get("observed_repository_revision")
    if report.revision and (
        not isinstance(observed_revision, dict)
        or report.revision != observed_revision.get("revision")
    ):
        failures.append(f"revision not observed in fixture: {report.revision}")

    observed_strategies = {
        item["strategy"]
        for item in observation.source_strategy_assessment.get("source_strategy_candidates", ())
    }
    reported_strategies = {
        candidate.strategy.value for candidate in report.source_strategy_candidates
    }
    missing_reported_strategies = sorted(reported_strategies - observed_strategies)
    if missing_reported_strategies:
        failures.append(
            f"reported source strategy lacks inspected evidence: {missing_reported_strategies}"
        )
    if scenario.expected_source_strategy and (
        scenario.expected_source_strategy.value not in observed_strategies
    ):
        failures.append(
            f"expected source strategy was not observed: {scenario.expected_source_strategy.value}"
        )
    failures.extend(_source_strategy_evidence_failures(report, observation))

    failures.extend(_checkpoint_failures(report, observation))
    failures.extend(_variant_failures(report, observation))
    failures.extend(_embedding_failures(report, observation))
    failures.extend(_dependency_failures(report, observation))
    failures.extend(_environment_failures(report, observation))

    if scenario.expected_failure_classification == FailureClassification.DEPENDENCY_CONFLICT:
        if not observation.dependency_evidence.get("unpinned_dependencies"):
            failures.append(
                "dependency_conflict expected but unpinned dependencies were not observed"
            )
    if scenario.expected_failure_classification == FailureClassification.INSUFFICIENT_EVIDENCE:
        if observed_files - {"README.md", "LICENSE", "SCENARIO.json"}:
            failures.append("insufficient_evidence expected despite material inspected files")
    return failures


def _failure_classification_represented(
    classification: FailureClassification,
    report: AnalysisReport,
) -> bool:
    return any(
        question.failure_classification == classification for question in report.open_questions
    )


def _observed_package_name(observation: ScenarioInspectionResult) -> str | None:
    pyproject = observation.packaging_evidence.get("pyproject", {})
    pyproject_name = pyproject.get("name") if isinstance(pyproject, dict) else None
    if isinstance(pyproject_name, str):
        return pyproject_name
    setup_py = observation.packaging_evidence.get("setup_py", {})
    if isinstance(setup_py, dict):
        values = setup_py.get("values", {})
        setup_py_name = values.get("name") if isinstance(values, dict) else None
        if isinstance(setup_py_name, str):
            return setup_py_name
    setup_cfg = observation.packaging_evidence.get("setup_cfg", {})
    setup_cfg_name = setup_cfg.get("name") if isinstance(setup_cfg, dict) else None
    if isinstance(setup_cfg_name, str):
        return setup_cfg_name
    return None


def _symbol_observed(
    observation: ScenarioInspectionResult,
    source_file: str | None,
    symbol: str,
) -> bool:
    by_file = _observed_symbols_by_file(observation)
    if source_file is None:
        return any(symbol in symbols for symbols in by_file.values())
    return symbol in by_file.get(source_file, set())


def _observed_symbols_by_file(observation: ScenarioInspectionResult) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for group in (
        observation.model_candidates.get("candidates", ()),
        observation.output_candidates.get("candidates", ()),
        observation.checkpoint_candidates.get("candidates", ()),
    ):
        for item in group:
            path = item.get("path")
            symbol = item.get("symbol")
            if isinstance(path, str) and isinstance(symbol, str):
                result.setdefault(path, set()).add(symbol)
    return result


def _source_strategy_evidence_failures(
    report: AnalysisReport,
    observation: ScenarioInspectionResult,
) -> list[str]:
    failures: list[str] = []
    evidence_by_id = {item.evidence_id: item for item in report.evidence_items}
    observed_by_strategy = {
        item["strategy"]: set(item.get("evidence", ()))
        for item in observation.source_strategy_assessment.get("source_strategy_candidates", ())
    }
    for candidate in report.source_strategy_candidates:
        observed_evidence = observed_by_strategy.get(candidate.strategy.value, set())
        if not observed_evidence:
            continue
        cited_files: set[str] = set()
        for evidence_id in candidate.evidence_ids:
            if evidence_id not in evidence_by_id:
                continue
            source_file = evidence_by_id[evidence_id].source_file
            if source_file is not None:
                cited_files.add(source_file)
        if cited_files and not cited_files <= observed_evidence:
            failures.append(
                "source strategy evidence did not match inspection: "
                f"{candidate.strategy.value} -> {sorted(cited_files - observed_evidence)}"
            )
    return failures


def _checkpoint_failures(
    report: AnalysisReport,
    observation: ScenarioInspectionResult,
) -> list[str]:
    failures: list[str] = []
    observed_candidates = tuple(observation.checkpoint_candidates.get("candidates", ()))
    evidence_by_id = {item.evidence_id: item.source_file for item in report.evidence_items}
    for candidate in report.checkpoint_candidates:
        evidence_files = {
            evidence_by_id[evidence_id]
            for evidence_id in candidate.evidence_ids
            if evidence_id in evidence_by_id and evidence_by_id[evidence_id] is not None
        }
        exact_matches = [
            item
            for item in observed_candidates
            if item.get("kind") in {"checkpoint_helper", "checkpoint_url"}
            and (item.get("source_file") or item.get("path")) in evidence_files
            and (item.get("helper_symbol") or item.get("symbol")) == candidate.helper_symbol
            and item.get("complete_url") == candidate.url
            and item.get("filename") == candidate.filename
        ]
        if not exact_matches:
            helper_candidates = [
                item
                for item in observed_candidates
                if item.get("kind") == "checkpoint_helper"
                and (item.get("source_file") or item.get("path")) in evidence_files
            ]
            if not candidate.helper_symbol and any(
                item.get("complete_url") == candidate.url
                and item.get("filename") == candidate.filename
                for item in helper_candidates
            ):
                failures.append(f"checkpoint helper symbol missing: {candidate.checkpoint_id}")
                continue
            symbol_matches = [
                item
                for item in observed_candidates
                if (item.get("helper_symbol") or item.get("symbol")) == candidate.helper_symbol
            ]
            if candidate.helper_symbol and not symbol_matches:
                failures.append(
                    f"checkpoint helper symbol was not observed: {candidate.checkpoint_id}"
                )
                continue
            source_matches = [
                item
                for item in symbol_matches
                if (item.get("source_file") or item.get("path")) in evidence_files
            ]
            if candidate.helper_symbol and not source_matches:
                failures.append(
                    f"checkpoint helper source file was not observed: {candidate.checkpoint_id}"
                )
                continue
            identity_scope = source_matches or [
                item
                for item in observed_candidates
                if (item.get("source_file") or item.get("path")) in evidence_files
            ]
            if candidate.url is not None and not any(
                item.get("complete_url") == candidate.url for item in identity_scope
            ):
                failures.append(f"checkpoint URL was not observed: {candidate.checkpoint_id}")
            if candidate.filename is not None and not any(
                item.get("filename") == candidate.filename for item in identity_scope
            ):
                failures.append(f"checkpoint filename was not observed: {candidate.checkpoint_id}")
            if candidate.hash_evidence:
                failures.append(
                    "checkpoint hash was not associated with observed candidate: "
                    f"{candidate.checkpoint_id}"
                )
            continue
        if len(exact_matches) != 1:
            failures.append(f"checkpoint helper candidate was ambiguous: {candidate.checkpoint_id}")
            continue
        observed = exact_matches[0]
        if candidate.helper_symbol:
            if candidate.expression_status != observed.get("expression_status"):
                failures.append(
                    f"checkpoint helper expression status mismatch: {candidate.checkpoint_id}"
                )
            if tuple(candidate.unresolved_components or ()) != tuple(
                _string_tuple(observed.get("unresolved_components", ()))
            ):
                failures.append(
                    f"checkpoint helper unresolved components mismatch: {candidate.checkpoint_id}"
                )
        if candidate.hash_evidence and candidate.hash_evidence not in _string_set(
            observed.get("associated_hashes", ())
        ):
            failures.append(
                "checkpoint hash was not associated with observed candidate: "
                f"{candidate.checkpoint_id}"
            )
    return failures


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple, set)):
        return ()
    return tuple(str(item) for item in value)


def _string_set(value: object) -> set[str]:
    return set(_string_tuple(value))


def _variant_failures(
    report: AnalysisReport,
    observation: ScenarioInspectionResult,
) -> list[str]:
    observed_symbols = {
        str(item["symbol"])
        for item in observation.model_candidates.get("candidates", ())
        if item.get("symbol")
    }
    return [
        f"model variant was not observed: {candidate.variant_id}"
        for candidate in report.variants
        if candidate.name not in observed_symbols
    ]


def _embedding_failures(
    report: AnalysisReport,
    observation: ScenarioInspectionResult,
) -> list[str]:
    failures: list[str] = []
    observed_symbols = {
        str(item["symbol"])
        for item in observation.output_candidates.get("candidates", ())
        if item.get("symbol")
    }
    for candidate in report.embedding_candidates:
        if candidate.tensor_origin not in observed_symbols:
            failures.append(f"embedding tensor origin was not observed: {candidate.embedding_id}")
    return failures


def _environment_failures(
    report: AnalysisReport,
    observation: ScenarioInspectionResult,
) -> list[str]:
    failures: list[str] = []
    observed_candidate_strategies = {
        candidate.installation_strategy.value
        for candidate in observation.environment_candidates.candidates
    }
    reported_strategies = {
        candidate.strategy.value for candidate in report.source_strategy_candidates
    }
    missing = sorted(reported_strategies - observed_candidate_strategies)
    if missing:
        failures.append(f"environment candidates missing reported strategies: {missing}")
    if report.recommended_next_mode.value == "resolve-environment":
        if not observation.environment_candidates.candidates:
            failures.append(
                "resolve-environment recommended with no observed environment candidates"
            )
    return failures


def _dependency_failures(
    report: AnalysisReport,
    observation: ScenarioInspectionResult,
) -> list[str]:
    failures: list[str] = []
    observed_dependencies = set(observation.dependency_evidence.get("declared_dependencies", ()))
    observed_python = observation.dependency_evidence.get("python_constraint")
    for claim in report.dependency_evidence.claims:
        if claim.statement.startswith("Observed dependency: "):
            raw = claim.statement.removeprefix("Observed dependency: ")
            if raw not in observed_dependencies:
                failures.append(f"dependency declaration was not observed: {raw}")
        if claim.statement.startswith("Observed Python constraint: "):
            expected = claim.statement.removeprefix("Observed Python constraint: ")
            if expected != observed_python:
                failures.append(
                    f"Python constraint was not observed: {expected} != {observed_python}"
                )
    return failures
