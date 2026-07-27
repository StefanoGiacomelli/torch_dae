"""Validate the project version used by a release workflow."""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

VERSION_PATTERN = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)")


def read_project_version(path: Path) -> str:
    """Read and validate the PEP 621 project version.

    Parameters
    ----------
    path
        Path to the project's ``pyproject.toml``.

    Returns
    -------
    str
        The validated three-component project version.

    Raises
    ------
    ValueError
        If the file is invalid TOML or does not contain a supported project version.
    """

    try:
        data = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"cannot read valid pyproject.toml: {exc}") from None
    version = data.get("project", {}).get("version")
    if not isinstance(version, str) or VERSION_PATTERN.fullmatch(version) is None:
        raise ValueError("pyproject.toml has no valid project.version")
    return version


def validate_release_tag(tag: str, version: str) -> None:
    """Require an exact ``v<project-version>`` release tag.

    Parameters
    ----------
    tag
        GitHub Release tag to validate.
    version
        Validated project version.

    Raises
    ------
    ValueError
        If the tag is malformed or does not identify ``version`` exactly.
    """

    if not tag or not tag.startswith("v") or VERSION_PATTERN.fullmatch(tag[1:]) is None:
        raise ValueError("release tag must use exact format v<major>.<minor>.<patch>")
    expected = f"v{version}"
    if tag != expected:
        raise ValueError(f"release tag {tag!r} does not match project version {expected!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", nargs="?", help="GitHub Release tag in v<version> form")
    parser.add_argument(
        "--current",
        action="store_true",
        help="validate project.version without requiring a GitHub Release tag",
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path("pyproject.toml"),
        help="path to pyproject.toml (default: pyproject.toml)",
    )
    args = parser.parse_args(argv)
    if args.current == (args.tag is not None):
        parser.error("provide exactly one of TAG or --current")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run release-version validation with concise expected-error output."""

    try:
        args = parse_args(argv)
        version = read_project_version(args.pyproject)
        if args.current:
            print(f"project version validated: {version}")
        else:
            validate_release_tag(args.tag, version)
            print(f"release version validated: v{version}")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
