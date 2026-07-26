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
