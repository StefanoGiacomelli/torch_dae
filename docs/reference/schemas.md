# Schema reference

Generated JSON Schemas under `schemas/` mirror the strict Pydantic contracts for model cards,
checkpoints, embeddings, environments, source manifests, verification reports, analysis reports,
and environment-resolution reports.

```bash
uv run python scripts/generate_schemas.py --check
uv run python scripts/validate_repository.py
```

Do not hand-edit generated schemas. Change the typed contract, regenerate, review the diff, and run
both Pydantic and JSON Schema fixture tests.

Authored field semantics and cross-field invariants are in {doc}`../api/model-cards`,
{doc}`../api/environment-specifications`, {doc}`../api/runtime-verification`, and
{doc}`../api/onboarding-contracts`. Generated schemas are serialization artifacts; inherited
Pydantic implementation methods are not part of the curated user API.
