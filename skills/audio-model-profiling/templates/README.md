# Profiling skill templates

This directory contains user-facing request templates for the canonical
`audio-model-profiling` skill.

Use `agent-request.md` when you want a compact machine-oriented request. For guided examples and
phase-specific prompts, see `docs/skill/prompt-library.md`.

The profiling skill is intentionally phase-scoped:

```text
resolve -> plan -> profile -> validate -> finalize
```

Send one mode per request. In particular, `plan` is a review boundary: it should not execute a real
profiling campaign until the user explicitly requests `profile`.

Candidate evidence must stay outside `technical_cards/` until a later independent human review and
promotion decision.
