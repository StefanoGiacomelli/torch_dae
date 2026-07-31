"""Manage committed onboarding phase handoffs and deterministic review bundles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from torch_dae.onboarding.contracts import OnboardingPhase
from torch_dae.onboarding.handoff import (
    HandoffManagementError,
    bundle_workflow,
    cleanup_workflow,
    discover_handoff,
    discover_repository_root,
    promote_phase,
    validate_workflow,
)


def _phase(value: str) -> OnboardingPhase:
    try:
        return OnboardingPhase(value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in OnboardingPhase)
        raise argparse.ArgumentTypeError(f"phase must be one of: {choices}") from exc


def build_parser() -> argparse.ArgumentParser:
    """Build the stable root-control-plane command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover", help="discover one accepted prerequisite phase")
    discover.add_argument("--workflow-id")
    discover.add_argument("--required-phase", type=_phase, required=True)
    discover.add_argument("--attachment", action="append", type=Path, default=[])
    discover.add_argument("--include-superseded", action="store_true")
    discover.add_argument("--json", action="store_true")

    validate = subparsers.add_parser("validate", help="validate one workflow or phase")
    validate.add_argument("--workflow-id", required=True)
    validate.add_argument("--phase", type=_phase)
    validate.add_argument("--json", action="store_true")

    promote = subparsers.add_parser("promote", help="atomically promote a managed phase")
    promote.add_argument("--workflow-id", required=True)
    promote.add_argument("--phase", type=_phase, required=True)
    promote.add_argument("--source-dir", type=Path, required=True)
    promote.add_argument("--supersede", action="store_true")
    promote.add_argument("--json", action="store_true")

    bundle = subparsers.add_parser("bundle", help="build a deterministic review archive")
    bundle.add_argument("--workflow-id", required=True)
    bundle.add_argument("--through-phase", type=_phase, required=True)
    bundle.add_argument("--output-dir", type=Path, required=True)
    bundle.add_argument("--include-working-tree", action="store_true")
    bundle.add_argument("--json", action="store_true")

    cleanup = subparsers.add_parser("cleanup", help="clean recorded managed runtime paths")
    cleanup.add_argument("--workflow-id", required=True)
    mode = cleanup.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    cleanup.add_argument("--include-repository-caches", action="store_true")
    cleanup.add_argument("--include-package-caches", action="store_true")
    cleanup.add_argument("--include-environments", action="store_true")
    cleanup.add_argument("--include-checkpoints", action="store_true")
    cleanup.add_argument("--json", action="store_true")
    return parser


def command_payload(args: argparse.Namespace, repository_root: Path) -> dict[str, object]:
    """Execute one parsed command and return its machine-readable result."""

    if args.command == "discover":
        return discover_handoff(
            repository_root,
            workflow_id=args.workflow_id,
            required_phase=args.required_phase,
            attachments=args.attachment,
            include_superseded=args.include_superseded,
        )
    if args.command == "validate":
        return validate_workflow(
            repository_root,
            args.workflow_id,
            phase=args.phase,
        )
    if args.command == "promote":
        return promote_phase(
            repository_root,
            source_dir=args.source_dir,
            workflow_id=args.workflow_id,
            phase=args.phase,
            supersede=args.supersede,
        )
    if args.command == "bundle":
        return bundle_workflow(
            repository_root,
            workflow_id=args.workflow_id,
            through_phase=args.through_phase,
            output_dir=args.output_dir,
            include_working_tree=args.include_working_tree,
        )
    if args.command == "cleanup":
        return cleanup_workflow(
            repository_root,
            workflow_id=args.workflow_id,
            dry_run=args.dry_run,
            include_repository_caches=args.include_repository_caches,
            include_package_caches=args.include_package_caches,
            include_environments=args.include_environments,
            include_checkpoints=args.include_checkpoints,
        )
    raise HandoffManagementError(f"unsupported command: {args.command}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        root = discover_repository_root()
        payload = command_payload(args, root)
    except (HandoffManagementError, OSError, ValidationError, ValueError) as exc:
        failure = {"valid": False, "error": str(exc)}
        if getattr(args, "json", False):
            print(json.dumps(failure, indent=2, sort_keys=True))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if args.command == "cleanup" and payload.get("errors"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
