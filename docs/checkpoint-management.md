# Checkpoint Management

The checkpoint subsystem resolves a card checkpoint into ignored cache state:

```text
.torch-dae/checkpoints/<checkpoint-id>/<sha256>/
```

Each cache entry contains the checkpoint file and `checkpoint-materialization.json`. SHA-256 is always
computed while streaming or copying bytes. Expected and observed hashes are enforced when present and
recorded through explicit `hash-validation` reports. Successful metadata includes report references
for acquisition, hash-validation, and cache-finalization operations.

Schema `2.0.0` checkpoint specifications may bind structured authority. Zenodo is the first
production provider: the manager derives the official record API from the canonical record ID,
resolves metadata before requesting payload bytes, selects exactly one equal filename, and retains a
bounded structured metadata-response record. Arbitrary HTTPS URLs remain ordinary sources and are
never treated as authoritative records implicitly.

The checkpoint filename grammar accepts safe metric-bearing names such as
`Cnn14_16k_mAP=0.438.pth` and safe nested provider paths. It rejects absolute and drive paths,
traversal, empty or doubled segments, backslashes, NULs, URL queries/fragments, and cache/package
escapes.

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
torch-dae checkpoint resolve --spec <checkpoint-spec.json> --json
torch-dae checkpoint ensure-spec --spec <checkpoint-spec.json> --json
torch-dae checkpoint info-spec --spec <checkpoint-spec.json> --json
```

The three `--spec` operations do not require a model card. `resolve` never downloads the payload;
`ensure-spec` delegates to `CheckpointManager.ensure_checkpoint`; and `info-spec` performs no network
access. Card-based `ensure` is a convenience delegating to the same manager primitive.

Cache entries are bound to a deterministic checkpoint-specification fingerprint covering acquisition
fields such as source type, URL, repository/revision, package resource, local path, expected/observed
hashes, format, and loader. Changing any acquisition field prevents silent reuse of an older asset
with the same checkpoint ID.

Authority inputs—provider, record ID, requested filename, exact expected size, and canonical
published checksum set—are fingerprinted. Retrieval timestamps, response headers, resolved mutable
payload URLs, cache paths, and runtime logs are excluded.

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

Authority-complete cache reuse rehashes cached bytes, requires the exact expected size, revalidates
every provider-published checksum, proves observed SHA-256 equals the content-addressed cache
identity, and validates the retained metadata-resolution reference and hash. A file-existence check
is never a cache hit.

Terminology is intentionally strict:

- **Published checksum**: evidence declared by the authoritative provider.
- **Observed checksum**: digest calculated from acquired or cached local bytes.
- **Exact expected size**: authority identity and integrity constraint.
- **Maximum bytes**: independent resource-safety ceiling.

An MD5-only authority is supported: MD5 is verified as published evidence, while locally computed
SHA-256 remains separately labeled observed evidence and becomes the cache identity. Failures also
distinguish metadata/file identity mismatch, exact-size mismatch, truncated transfer, published
checksum mismatch, and an oversized response.

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
