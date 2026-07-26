"""Shared CLI helpers for Phase 02 skill-local scripts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from torch_dae.onboarding.inspection import OnboardingInspectionError


def emit_json(payload: Any) -> None:
    """Emit deterministic JSON."""

    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def repository_argument(parser: argparse.ArgumentParser) -> None:
    """Add the standard repository argument."""

    parser.add_argument("repository", type=Path, help="local repository checkout to inspect")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")


def run_json_command(
    func: Callable[[argparse.Namespace], Any], parser: argparse.ArgumentParser
) -> int:
    """Run a script command with typed, traceback-free expected failures."""

    args = parser.parse_args()
    try:
        payload = func(args)
    except (OnboardingInspectionError, ValidationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if getattr(args, "json", False):
        emit_json(payload)
    else:
        print(_summary(payload))
    return 0


def _summary(payload: Any) -> str:
    if isinstance(payload, dict):
        if "candidates" in payload and isinstance(payload["candidates"], list):
            return f"{len(payload['candidates'])} candidates"
        if "files" in payload and isinstance(payload["files"], list):
            return f"{len(payload['files'])} files"
        return ", ".join(sorted(payload.keys()))
    return str(payload)
