"""Render or validate deterministic Markdown for a onboarding workflow analysis report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import emit_json
from pydantic import ValidationError

from torch_dae.onboarding.contracts import AnalysisReport
from torch_dae.onboarding.rendering import render_analysis_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="analysis report JSON path")
    parser.add_argument("--output", type=Path, help="optional Markdown output path")
    parser.add_argument("--check", type=Path, help="fail if Markdown file is not current")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    try:
        report = AnalysisReport.model_validate_json(args.report.read_text())
    except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        if args.json:
            emit_json({"valid": False, "error": str(exc)})
        else:
            print(f"error: {exc}")
        return 2
    markdown = render_analysis_markdown(report)
    if args.check is not None:
        valid = args.check.exists() and args.check.read_text() == markdown
        if args.json:
            emit_json({"valid": valid, "report_id": report.report_id})
        elif not valid:
            print(f"Markdown out of date: {args.check}")
        return 0 if valid else 2
    if args.output is not None:
        args.output.write_text(markdown)
    elif args.json:
        emit_json({"report_id": report.report_id, "markdown": markdown})
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
