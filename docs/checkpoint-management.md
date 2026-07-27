# Checkpoint Management

The checkpoint subsystem resolves a card checkpoint into ignored cache state:

```text
.torch-dae/checkpoints/<checkpoint-id>/<sha256>/
```

Each cache entry contains the checkpoint file and `checkpoint-materialization.json`. SHA-256 is always
computed while streaming or copying bytes. Expected and observed hashes are enforced when present and
recorded through explicit `hash-validation` reports. Successful metadata includes report references
for acquisition, hash-validation, and cache-finalization operations.

Supported sources are:

- `https`
- `github_release`
- `huggingface`
- `package_bundle`
- `local_path`

Remote downloads use an injectable transport for tests and the Python standard library in production.
GitHub and Hugging Face tokens may be read from the environment, but tokens are not stored in metadata
or reports.

Use:

```bash
torch-dae checkpoint ensure <card-id>
torch-dae checkpoint info <card-id> --json
torch-dae checkpoint remove <card-id>
```

Cache entries are bound to a deterministic checkpoint-specification fingerprint covering acquisition
fields such as source type, URL, repository/revision, package resource, local path, expected/observed
hashes, format, and loader. Changing any acquisition field prevents silent reuse of an older asset
with the same checkpoint ID.

Package-bundle checkpoints are located inside the ensured model environment without importing model
packages into the root control plane. The lookup resolves the exact distribution, verifies the exact
version, requires an exact `distribution.files` member match, and rejects traversal, absolute paths,
backslashes, empty segments, missing resources, and files owned by other distributions.

`--offline` reuses valid cached remote checkpoints and fails before remote access on a network cache
miss. The miss is recorded as an `offline-cache-lookup` report with `offline_cache_miss`
classification. Offline mode still permits first acquisition from local resources: `local_path` and
`package_bundle` when the required environment/package is already available. Remote response bodies
are closed and download temporaries are unique runtime files.

Checkpoint acquisition reports are written under
`.torch-dae/reports/checkpoints/<checkpoint-id>/`. They cover remote open, remote streaming, remote
finalization, local-path copy, package-bundle lookup/copy, hash validation, cache finalization,
metadata writes, offline cache misses, response-close failures, and failure cleanup when those
operations execute. Failed acquisitions retain runtime failure reports even when no valid checkpoint
metadata is created. Cleanup reports are separate from the acquisition-failure report they follow.
Reports include checkpoint ID, source type, sanitized source description, result status, byte count,
SHA-256 when known, failure detail, and failure classification such as `OSError`, `URLError`,
`HTTPError`, `expected_hash_mismatch`, `observed_hash_mismatch`, `offline_cache_miss`, or
`JSONDecodeError`.

Expected operational failures are normalized into `CheckpointAcquisitionError`,
`CheckpointHashMismatchError`, `CheckpointNotFoundError`, or `OfflineResourceUnavailableError` with
the original exception preserved as `__cause__` when one exists. Local and package-bundle acquisitions
use unique temporary files before cache placement. If cache finalization fails, the temporary file is
removed. If the checkpoint file is placed but `checkpoint-materialization.json` cannot be written, the
incomplete cache entry is removed so later cache validation cannot treat it as valid.

Response-close failures are deterministic: an earlier acquisition exception remains the raised error
and the close failure is recorded separately; if acquisition otherwise succeeded, the close failure is
raised as `CheckpointAcquisitionError`. The checkpoint CLI prints concise errors without tracebacks:
not-found and offline-unavailable failures exit with code `3`, while acquisition and hash failures
exit with code `4`. Authorization values, bearer tokens, token-like arguments, credential-bearing
URLs, and secret environment-variable values are redacted from reports and user-facing acquisition
messages.
