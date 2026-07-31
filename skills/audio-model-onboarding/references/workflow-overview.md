# Workflow Overview

The skill converts unfamiliar upstream evidence into a structured onboarding package. The workflow is:
upstream identifier or local checkout -> static repository inspection -> evidence collection ->
technical analysis report -> user decision gates -> compatibility candidates -> integration plan ->
verification plan -> model-card draft preparation.

Static analysis does not execute upstream code or acquire checkpoints. Production integration and
controlled verification are available only through explicitly requested modes after their
prerequisites are satisfied. Static utilities gather evidence; the agent interprets architecture,
embeddings, source strategy, and scientific ambiguity.

Run model-agnostic skill tooling through the locked root control plane:
`uv sync --all-groups --frozen`, followed by `uv run <command>`. The root `.venv` is expected
ignored runtime state and is not a model environment. PyTorch, TorchAudio, Transformers, and other
model-runtime dependencies remain confined to the model-environment subsystem. `pypdf` is a
control-plane document-analysis dependency. Report creation of the root environment when it occurs,
but do not call the legitimate root `.venv` contamination. Bytecode and tool caches remain ignored
and must not enter audit archives.

Supplied local papers may be read with `scripts/extract_pdf_text.py`. The utility extracts only an
available text layer, preserves page boundaries, and provides no OCR. Extracted text is a reading
aid, not automatically verified evidence, and is not stored in the repository unless explicitly
requested.

The allowed modes are `analyze`, `resolve-environment`, `integrate`, `verify`, `card`, and
`profile`. `profile` is reserved until a runtime-verified model and an explicitly implemented
profiling workflow exist.

Every mode accepts `WORKFLOW_ID`. Use `scripts/onboarding_handoff.py discover` before requesting a
prerequisite attachment. The command searches only `onboarding_reports/`, validates the accepted
handoff and all local hashes, and returns canonical artifact paths. Without `WORKFLOW_ID`, automatic
selection is allowed only when exactly one compatible active workflow exists.

Phase runs use `.torch-dae/workspaces/<workflow-id>/<phase>/<run-id>/` with a managed run manifest.
Validate there, promote accepted pre-runtime outputs under `onboarding_reports/`, build the
deterministic external review bundle, then run scoped cleanup. Reusable repository/package caches,
materialized model environments, and checkpoint caches are managed runtime state, not accidental
contamination, and are not removed by default.
