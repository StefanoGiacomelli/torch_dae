# Workflow Overview

The skill converts unfamiliar upstream evidence into a structured onboarding package. The workflow is:
upstream identifier or local checkout -> static repository inspection -> evidence collection ->
technical analysis report -> user decision gates -> compatibility candidates -> integration plan ->
verification plan -> model-card draft preparation.

Static analysis does not execute upstream code or acquire checkpoints. Production integration and
controlled verification are available only through explicitly requested modes after their
prerequisites are satisfied. Static utilities gather evidence; the agent interprets architecture,
embeddings, source strategy, and scientific ambiguity.

The allowed modes are `analyze`, `resolve-environment`, `integrate`, `verify`, `card`, and
`profile`. `profile` is reserved until a runtime-verified model and an explicitly implemented
profiling workflow exist.
