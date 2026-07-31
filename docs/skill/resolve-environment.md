# Resolve-environment mode

`resolve-environment` converts analyzed dependency evidence into bounded compatibility candidates
and, when authorized, controlled trials in ignored runtime state. It records the exact Python
version, direct dependencies, lockfile, source installation strategy, platform evidence,
verification command, outcomes, and failure classifications.

The mode accepts `WORKFLOW_ID` and discovers the accepted analyze handoff locally. Direct imports
must be direct dependencies, and exact pins are checked by the verification script. Shared source
and environment evidence requires per-tuple compatibility evidence.

A constructor trial proves only dependency resolution, import, construction, and parameter-device
placement. Draft resolution may complete successfully without materialization, fingerprint, or
lifecycle promotion. Only canonical materialization and successful verification can advance to
`environment_resolved`. Accepted output is promoted under
`onboarding_reports/<workflow-id>/resolve-environment/`, bundled, and cleaned.
