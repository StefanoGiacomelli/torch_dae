"""Generate the strict Draft 2020-12 JSON Schema for Technical Cards.

Kept separate from `scripts/generate_schemas.py` deliberately: that shared script's bytes are
pinned by already-accepted onboarding evidence (`onboarding_reports/panns-audioset-three-tuple/`),
and Profiling v1 is an independent subsystem that must not perturb accepted onboarding artifact
hashes (project_spec.md Section 27; "Technical Cards ... MUST NOT mutate ... that Model Card").
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from torch_dae.profiling.contracts import TechnicalCard

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "technical-card.schema.json"
SCHEMA_ID = "https://torch-dae.local/schemas/technical-card.schema.json"


def add_string_min_length(value: Any) -> None:
    """Recursively reject empty strings, matching `scripts/generate_schemas.py` conventions."""

    if isinstance(value, dict):
        if value.get("type") == "string" and "minLength" not in value:
            value["minLength"] = 1
        for child in value.values():
            add_string_min_length(child)
    elif isinstance(value, list):
        for child in value:
            add_string_min_length(child)


def render() -> str:
    schema = TechnicalCard.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = SCHEMA_ID
    add_string_min_length(schema)
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if the schema is not current")
    args = parser.parse_args()

    content = render()
    if args.check:
        if not SCHEMA_PATH.is_file() or SCHEMA_PATH.read_text() != content:
            print(f"schema out of date: {SCHEMA_PATH}")
            return 1
        return 0

    SCHEMA_PATH.parent.mkdir(exist_ok=True)
    SCHEMA_PATH.write_text(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
