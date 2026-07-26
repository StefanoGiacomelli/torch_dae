"""Generate ordered, evidence-motivated environment candidates without trial execution."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import repository_argument, run_json_command

from torch_dae.onboarding.contracts import EnvironmentCandidateGenerationResult
from torch_dae.onboarding.inspection import generate_environment_candidates


def command(args: argparse.Namespace) -> dict[str, object]:
    """Generate unverified compatibility candidates."""

    target_platform = args.target_platform
    result = generate_environment_candidates(
        args.repository,
        target_platform=target_platform,
        external_pytorch_root=args.external_pytorch_root,
    )
    return EnvironmentCandidateGenerationResult.model_validate(result).model_dump(mode="json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repository_argument(parser)
    parser.add_argument("--target-platform", help="optional target platform evidence")
    parser.add_argument(
        "--external-pytorch-root",
        type=Path,
        help="explicit inspected external PyTorch implementation repository root",
    )
    return run_json_command(command, parser)


if __name__ == "__main__":
    raise SystemExit(main())
