# Scripts

Skill-local deterministic utilities live here. They accept explicit arguments, support `--json`, and
perform static inspection without executing upstream repository code.

`extract_pdf_text.py <local.pdf> [--json]` extracts only a local PDF's machine-readable text layer
with explicit page boundaries and safe metadata. Defaults limit input to 25,000,000 bytes, 200 pages,
and 1,000,000 extracted characters; validated overrides are capped at 100,000,000 bytes, 1,000
pages, and 5,000,000 characters. Encrypted, malformed, non-PDF, no-text-layer, and limit outcomes are
reported explicitly. It never retrieves a URL, invokes an external process, executes PDF actions, or
provides OCR. Extraction is read-only and does not automatically make the resulting text verified
scientific evidence.
