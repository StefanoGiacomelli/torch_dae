"""Enforce line and branch thresholds from pytest-cov JSON output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def percentage(covered: object, total: object, *, label: str) -> float:
    if not isinstance(covered, (int, float)) or not isinstance(total, (int, float)):
        raise ValueError(f"{label} totals must be numeric")
    if covered < 0 or total < 0 or covered > total:
        raise ValueError(f"{label} totals are inconsistent")
    return 100.0 if total == 0 else (float(covered) / float(total)) * 100.0


def coverage_percentages(payload: object) -> tuple[float, float]:
    if not isinstance(payload, dict):
        raise ValueError("coverage JSON must contain an object")
    totals: Any = payload.get("totals")
    if not isinstance(totals, dict):
        raise ValueError("coverage JSON is missing totals")
    line = percentage(
        totals.get("covered_lines"),
        totals.get("num_statements"),
        label="line coverage",
    )
    branch = percentage(
        totals.get("covered_branches"),
        totals.get("num_branches"),
        label="branch coverage",
    )
    return line, branch


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("coverage_json", type=Path)
    result.add_argument("--min-line", type=float, required=True)
    result.add_argument("--min-branch", type=float, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        payload = json.loads(args.coverage_json.read_text())
        line, branch = coverage_percentages(payload)
        if not 0 <= args.min_line <= 100 or not 0 <= args.min_branch <= 100:
            raise ValueError("coverage thresholds must be between 0 and 100")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Line coverage: {line:.2f}% (required {args.min_line:.2f}%)")
    print(f"Branch coverage: {branch:.2f}% (required {args.min_branch:.2f}%)")
    return 0 if line >= args.min_line and branch >= args.min_branch else 1


if __name__ == "__main__":
    raise SystemExit(main())
