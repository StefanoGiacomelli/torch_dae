"""Inspect dependency declarations and static imports."""

from __future__ import annotations

import argparse

from common import repository_argument, run_json_command

from torch_dae.onboarding.inspection import inspect_dependencies


def command(args: argparse.Namespace) -> dict[str, object]:
    """Run dependency inspection."""

    return inspect_dependencies(args.repository)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repository_argument(parser)
    return run_json_command(command, parser)


if __name__ == "__main__":
    raise SystemExit(main())
