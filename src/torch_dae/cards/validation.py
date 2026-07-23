"""Dual Pydantic and JSON Schema validation for model cards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import ValidationError as PydanticValidationError

from torch_dae.cards.models import ModelCard


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""

    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return value


def validate_model_card_pydantic(data: dict[str, Any]) -> ModelCard:
    """Validate a card through Pydantic."""

    return ModelCard.model_validate(data)


def validate_model_card_schema(data: dict[str, Any], schema_path: Path) -> None:
    """Validate a card through JSON Schema with format checking."""

    schema = load_json(schema_path)
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(data)


def assert_valid_pydantic(data: dict[str, Any]) -> ModelCard:
    """Assert that Pydantic accepts a model card."""

    return validate_model_card_pydantic(data)


def assert_invalid_pydantic(data: dict[str, Any]) -> None:
    """Assert that Pydantic rejects a model card."""

    try:
        validate_model_card_pydantic(data)
    except PydanticValidationError:
        return
    raise AssertionError("expected Pydantic validation to fail")


def assert_valid_json_schema(data: dict[str, Any], schema_path: Path) -> None:
    """Assert that JSON Schema accepts a model card."""

    validate_model_card_schema(data, schema_path)


def assert_invalid_json_schema(data: dict[str, Any], schema_path: Path) -> None:
    """Assert that JSON Schema rejects a model card."""

    try:
        validate_model_card_schema(data, schema_path)
    except JsonSchemaValidationError:
        return
    raise AssertionError("expected JSON Schema validation to fail")


def validate_model_card_path(path: Path, schema_path: Path | None = None) -> ModelCard:
    """Validate a model-card file through both validators when a schema is available."""

    data = load_json(path)
    card = validate_model_card_pydantic(data)
    if schema_path is not None:
        validate_model_card_schema(data, schema_path)
    return card
