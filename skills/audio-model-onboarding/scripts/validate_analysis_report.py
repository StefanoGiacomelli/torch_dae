"""Validate a Phase 02 technical-analysis report JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import emit_json
from pydantic import ValidationError

from torch_dae.onboarding.contracts import AnalysisReport


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="analysis report JSON path")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    try:
        report = AnalysisReport.model_validate_json(args.report.read_text())
    except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    payload = {
        "valid": True,
        "report_id": report.report_id,
        "recommended_next_mode": report.recommended_next_mode.value,
        "open_questions": len(report.open_questions),
    }
    if args.json:
        emit_json(payload)
    else:
        print(f"valid analysis report: {report.report_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
