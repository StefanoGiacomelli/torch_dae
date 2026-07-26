"""Generate strict Draft 2020-12 JSON schemas from Pydantic contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from torch_dae.cards.models import ModelCard
from torch_dae.core.checkpoint import CheckpointSpec
from torch_dae.core.embeddings import EmbeddingSpec
from torch_dae.environment.specification import EnvironmentSourcesManifest, EnvironmentSpecification
from torch_dae.environment.verification import VerificationReport
from torch_dae.onboarding.contracts import AnalysisReport, EnvironmentResolutionReport

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "schemas"

SCHEMAS: dict[str, tuple[str, type[BaseModel]]] = {
    "model-card.schema.json": ("https://torch-dae.local/schemas/model-card.schema.json", ModelCard),
    "checkpoint.schema.json": (
        "https://torch-dae.local/schemas/checkpoint.schema.json",
        CheckpointSpec,
    ),
    "environment.schema.json": (
        "https://torch-dae.local/schemas/environment.schema.json",
        EnvironmentSpecification,
    ),
    "environment-sources.schema.json": (
        "https://torch-dae.local/schemas/environment-sources.schema.json",
        EnvironmentSourcesManifest,
    ),
    "embedding.schema.json": (
        "https://torch-dae.local/schemas/embedding.schema.json",
        EmbeddingSpec,
    ),
    "verification-report.schema.json": (
        "https://torch-dae.local/schemas/verification-report.schema.json",
        VerificationReport,
    ),
    "analysis-report.schema.json": (
        "https://torch-dae.local/schemas/analysis-report.schema.json",
        AnalysisReport,
    ),
    "environment-resolution-report.schema.json": (
        "https://torch-dae.local/schemas/environment-resolution-report.schema.json",
        EnvironmentResolutionReport,
    ),
}


def normalize_schema(schema: dict[str, Any], schema_id: str) -> dict[str, Any]:
    """Normalize generated schemas for deterministic commits."""

    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = schema_id
    add_string_min_length(schema)
    return schema


def add_string_min_length(value: Any) -> None:
    """Recursively reject empty strings in generated schemas."""

    if isinstance(value, dict):
        if value.get("type") == "string" and "minLength" not in value:
            value["minLength"] = 1
        for child in value.values():
            add_string_min_length(child)
    elif isinstance(value, list):
        for child in value:
            add_string_min_length(child)


def ref(def_name: str) -> dict[str, str]:
    return {"$ref": f"#/$defs/{def_name}"}


def augment_checkpoint_schema(schema: dict[str, Any]) -> None:
    def null_fields(*names: str) -> dict[str, dict[str, str]]:
        return {name: {"type": "null"} for name in names}

    schema["allOf"] = [
        {
            "if": {"properties": {"source_type": {"const": "https"}}},
            "then": {
                "required": ["url"],
                "properties": {
                    "url": {"type": "string"},
                    **null_fields(
                        "repository_id",
                        "package",
                        "package_version",
                        "revision",
                        "release_tag",
                        "local_path",
                    ),
                },
            },
        },
        {
            "if": {"properties": {"source_type": {"const": "github_release"}}},
            "then": {
                "required": ["repository_id", "release_tag", "filename"],
                "properties": {
                    "repository_id": {"type": "string"},
                    "release_tag": {"type": "string"},
                    "filename": {"type": "string"},
                    **null_fields("url", "package", "package_version", "local_path"),
                },
            },
        },
        {
            "if": {"properties": {"source_type": {"const": "huggingface"}}},
            "then": {
                "required": ["repository_id", "filename"],
                "properties": {
                    "repository_id": {"type": "string"},
                    "filename": {"type": "string"},
                    **null_fields("url", "package", "package_version", "release_tag", "local_path"),
                },
            },
        },
        {
            "if": {"properties": {"source_type": {"const": "package_bundle"}}},
            "then": {
                "required": ["package", "package_version", "filename"],
                "properties": {
                    "package": {"type": "string"},
                    "package_version": {"type": "string"},
                    "filename": {"type": "string"},
                    **null_fields("url", "repository_id", "revision", "release_tag", "local_path"),
                },
            },
        },
        {
            "if": {"properties": {"source_type": {"const": "local_path"}}},
            "then": {
                "required": ["local_path"],
                "properties": {
                    "local_path": {"type": "string"},
                    **null_fields(
                        "url",
                        "repository_id",
                        "package",
                        "package_version",
                        "revision",
                        "release_tag",
                        "filename",
                    ),
                },
            },
        },
    ]


def augment_model_card_schema(schema: dict[str, Any]) -> None:
    defs = schema["$defs"]
    defs["BooleanCapability"]["allOf"] = [
        {
            "if": {"properties": {"supported": {"const": False}}},
            "then": {"required": ["reason"], "properties": {"reason": {"type": "string"}}},
        }
    ]
    defs["EvidenceRecord"]["allOf"] = [
        {
            "if": {"properties": {"status": {"const": "inferred"}}},
            "then": {"required": ["rationale"], "properties": {"rationale": {"type": "string"}}},
        }
    ]
    defs["ProfilingSection"]["allOf"] = [
        {
            "if": {"properties": {"status": {"const": "profiled"}}},
            "then": {"required": ["report"], "properties": {"report": {"type": "string"}}},
        }
    ]
    defs["EmbeddingsSection"]["properties"]["items"]["contains"] = {
        "type": "object",
        "required": ["default"],
        "properties": {"default": {"const": True}},
    }
    defs["EmbeddingsSection"]["properties"]["items"]["minContains"] = 1
    defs["EmbeddingsSection"]["properties"]["items"]["maxContains"] = 1
    schema["allOf"] = [
        {
            "if": {
                "properties": {
                    "card_status": {
                        "enum": [
                            "environment_resolved",
                            "checkpoint_verified",
                            "runtime_verified",
                            "profiled",
                        ]
                    }
                }
            },
            "then": {
                "properties": {
                    "usage": {
                        "properties": {
                            "recommended_environment": {
                                "properties": {"verified": {"const": True}},
                                "required": ["verified"],
                            }
                        }
                    }
                }
            },
        },
        {
            "if": {
                "properties": {
                    "card_status": {"enum": ["checkpoint_verified", "runtime_verified", "profiled"]}
                }
            },
            "then": {
                "properties": {
                    "checkpoint": {
                        "properties": {"observed_sha256": {"type": "string"}},
                        "required": ["observed_sha256"],
                    }
                }
            },
        },
        {
            "if": {"properties": {"card_status": {"enum": ["runtime_verified", "profiled"]}}},
            "then": {
                "required": ["verification_report"],
                "properties": {"verification_report": {"type": "string"}},
            },
        },
        {
            "if": {"properties": {"card_status": {"const": "profiled"}}},
            "then": {
                "properties": {
                    "architectural_profiling": {"properties": {"status": {"const": "profiled"}}},
                    "inference_profiling": {"properties": {"status": {"const": "profiled"}}},
                    "energy_profiling": {"properties": {"status": {"const": "profiled"}}},
                }
            },
        },
        {
            "if": {
                "properties": {
                    "capabilities": {
                        "properties": {
                            "probabilities": {"properties": {"supported": {"const": True}}}
                        }
                    }
                }
            },
            "then": {
                "properties": {
                    "outputs": {
                        "required": ["probability_output"],
                        "properties": {"probability_output": ref("OutputComponent")},
                    }
                }
            },
        },
        {
            "if": {
                "properties": {
                    "capabilities": {
                        "properties": {
                            "probabilities": {"properties": {"supported": {"const": False}}}
                        }
                    }
                }
            },
            "then": {
                "properties": {"outputs": {"properties": {"probability_output": {"type": "null"}}}}
            },
        },
        {
            "if": {
                "properties": {
                    "capabilities": {
                        "properties": {
                            "embeddings": {"properties": {"supported": {"const": False}}}
                        }
                    }
                }
            },
            "then": {"properties": {"embeddings": {"properties": {"items": {"maxItems": 0}}}}},
        },
    ]


def render() -> dict[Path, str]:
    """Render all schema files."""

    rendered: dict[Path, str] = {}
    for filename, (schema_id, model) in SCHEMAS.items():
        schema = normalize_schema(model.model_json_schema(), schema_id)
        if filename == "checkpoint.schema.json":
            augment_checkpoint_schema(schema)
        elif filename == "model-card.schema.json":
            augment_model_card_schema(schema)
        rendered[SCHEMA_DIR / filename] = json.dumps(schema, indent=2, sort_keys=True) + "\n"
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if schemas are not current")
    args = parser.parse_args()

    SCHEMA_DIR.mkdir(exist_ok=True)
    changed: list[Path] = []
    for path, content in render().items():
        if path.exists() and path.read_text() == content:
            continue
        changed.append(path)
        if not args.check:
            path.write_text(content)

    if args.check and changed:
        for path in changed:
            print(f"schema out of date: {path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
