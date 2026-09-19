# Repository Analysis

Analyze repository identity, owner/name, inspected revision, license, package metadata, release/tag
evidence, maintenance evidence, and official versus third-party status.

Use static utilities first:

- `inspect_repository.py`
- `inspect_python_project.py`
- `inspect_dependencies.py`
- `inspect_model_candidates.py`
- `inspect_output_candidates.py`
- `inspect_checkpoints.py`

Static candidates remain candidates until source reading or runtime evidence confirms their meaning.
Do not execute `setup.py`, notebooks, shell scripts, or upstream imports in the root environment.

For source/checkpoint chronology, use this bounded evidence hierarchy:

1. a revision explicitly recorded by authoritative checkpoint-host or release metadata;
2. a signed or annotated tag explicitly associated with the checkpoint;
3. an upstream release or commit explicitly cited by authoritative documentation;
4. a checkpoint-era revision supported by repository history and dates;
5. the nearest preceding commit, recorded only as a candidate and never as proof;
6. current default-branch HEAD, recorded only as current-source evidence.

Record every inspected revision, keep selection separate for different checkpoint candidates, and
state whether compatibility is exact evidence or chronological inference. Do not infer equivalence
from filename, metric, or date proximity. When no exact relationship can be demonstrated, stop the
bounded search with unresolved status and name the missing release, tag, commit, or host metadata
that would improve confidence; exhaustive historical archaeology is not required.

Use `scripts/extract_pdf_text.py` only for a supplied local PDF. It can expose machine-readable text
and metadata for review, but it does not perform OCR or turn extracted prose into verified evidence.
