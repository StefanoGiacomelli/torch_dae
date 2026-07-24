# Changelog

## Unreleased

- Closed Phase 01 environment/checkpoint cache integrity gaps: backend-built local wheels,
  sanitized model-environment subprocesses, Git source wheel metadata, installed wheel drift checks,
  package-bundle ownership checks, checkpoint spec fingerprints, local offline acquisition, and
  runtime command diagnostics.
- Corrected Phase 01 card/environment identity handling, exact cross-document environment path
  validation, local package content identity coverage, online Git checkout recovery, atomic Git clone
  staging, wheel-cache metadata verification, and failed-materialization metadata.
- Added shared sanitized runtime report sinks for environment commands, source commands, checkpoint
  acquisition events, failed materialization metadata, real Git recovery coverage, response-closure
  checks, interrupted-download cleanup, and cross-distribution package-bundle rejection.
- Normalized expected checkpoint acquisition failures into typed errors with hash-validation,
  offline-cache-miss, metadata-write, cache-finalization, cleanup, response-close, redaction, and
  no-traceback CLI regression coverage.
- Implemented Phase 01 environment materialization, source verification, checkpoint acquisition,
  offline/cache policies, runtime metadata, CLI behavior, repository validation, and synthetic tests.
- Added the Phase 00 repository scaffold.
- Added strict typed contracts, schema generation, CLI skeleton, environment and checkpoint interfaces, registry support, shared skill links, and synthetic validation fixtures.
