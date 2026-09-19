# Failure Classification

Resolve-environment failures must be classified with a specific cause, including:
`python_constraint`, `dependency_conflict`, `resolution_failure`, `removed_api`, `deprecated_api`,
`binary_or_abi_incompatibility`, `missing_binary_wheel`, `torch_torchaudio_mismatch`,
`numpy_compatibility`, `checkpoint_incompatibility`, `source_build_failure`, `import_failure`,
`runtime_failure`, `platform_incompatibility`, `access_or_authentication_blocker`, and
`sandbox_or_execution_policy`, `network_or_dns`, `package_index`, `rate_limit`, and
`insufficient_evidence`.

The next candidate must be motivated by evidence or failure diagnostics.

Card-independent environment operations additionally distinguish invalid specification, artifact
hash mismatch, interpreter unavailable, platform incompatibility, dependency installation, source
preparation, direct-dependency mismatch, verification-script failure, sandbox/execution policy, and
external-command failure. Environment failures never imply checkpoint incompatibility unless a
checkpoint was actually acquired and inspected in an authorized verify phase.

Local-wheel preflight failures use `dependency_closure` and retain separate nonempty collections for
missing requirements and incompatible reachable versions. A lock entry that is not reachable under
locked-project synchronization is missing for the local-wheel `--no-deps` installation policy.

Authoritative checkpoint operations distinguish `metadata_identity_mismatch`,
`metadata_response_invalid`, `metadata_response_oversized`, `untrusted_authority_url`,
`exact_size_mismatch`, `truncated_transfer`, `oversized_response`,
`published_checksum_mismatch`, `expected_hash_mismatch`, and `offline_cache_miss`. Exact-size or
published-checksum disagreement is always fatal, removes temporary payload bytes, and creates no
valid cache entry. Maximum bytes is an independent safety ceiling, not an integrity substitute.

Sandbox, DNS, package-index, authentication, and rate-limit failures are external execution
failures, not model/dependency incompatibilities. Preserve the original log and classification. When
execution policy permits, one identical evidence-motivated rerun may be recorded alongside the
initial outcome.

Licenses are informational and non-blocking. Missing, ambiguous, or restrictive license text must be
recorded as evidence or an open question, but it must not automatically classify a model as
unsupported.
