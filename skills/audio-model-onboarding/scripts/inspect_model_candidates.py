"""Inspect static model, loader, factory, and preprocessing candidates."""

from __future__ import annotations

import argparse

from common import repository_argument, run_json_command

from torch_dae.onboarding.inspection import inspect_model_candidates


def command(args: argparse.Namespace) -> dict[str, object]:
    """Run model-candidate inspection."""

    return inspect_model_candidates(args.repository)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repository_argument(parser)
    return run_json_command(command, parser)


if __name__ == "__main__":
    raise SystemExit(main())
