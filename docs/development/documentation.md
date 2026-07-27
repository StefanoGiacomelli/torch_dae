# Documentation

The documentation combines MyST Markdown guides with explicit reStructuredText API pages. Sphinx
uses Furo and enables autodoc, autosummary, Napoleon, intersphinx, and viewcode.

Build with all warnings fatal:

```bash
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
```

Only symbols entered once in `docs/api/public-api.toml` may be added to the curated reference. The
manifest records path, display name, category, audience, explicit methods, and source page. List
every symbol explicitly; do not use broad `:members:`, recursive module discovery, model-directory
scanning, or automatic wrapper catalogs. Add NumPy-style docstrings, authored Pydantic
field/invariant tables, exact cross-references, and focused manifest/import/render tests when the
surface changes.

Equations use native Sphinx math. Required flows use theme-aware browser text containers rather than
external image hosts or executables. The reference begins at {doc}`../api/index`.
