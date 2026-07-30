# Checkpoint Discovery

Collect checkpoint URLs, release assets, Hugging Face references, helper download functions, package
resources, local paths, hashes, filenames, archive behavior, authentication needs, licenses, and
variant mappings.

Never download a real checkpoint during static analysis or synthetic evaluation. Hidden helper
functions should be identified as evidence, but the helper must not be executed in the root
environment. Controlled acquisition is allowed only in `verify` mode through the checkpoint
subsystem for the explicitly selected model and checkpoint.

Checkpoint candidates derived from helpers preserve `helper_symbol`, `expression_status`,
`unresolved_components`, source file, complete URL, filename, and hash evidence. Literal URLs may
leave helper fields unset, but helper-derived URLs must remain tied to the observed helper function
for grounded evaluation. Hashes are associated with a candidate only when the static AST relationship
ties the hash to that helper or checkpoint metadata structure. Repository-global or unrelated-helper
hashes cannot satisfy a candidate. When association is unresolved, the observed hash collection is
empty, `hash association` remains unresolved, and reports must omit the hash.

Checkpoint candidates are not verified until acquired, hashed, and loaded in the intended model
environment during a later lifecycle stage.

## Metadata-only authoritative-host procedure

During `analyze`, inspect authoritative metadata without opening asset download URLs:

- For Zenodo, prefer the immutable record metadata/API and record record ID, related identifiers,
  filenames, sizes, licenses, access conditions, and each checksum with its exact algorithm.
- For GitHub Releases, use the official release/API metadata and record repository, release/tag ID,
  asset ID, filename, size, content type when present, and published digest when present.
- For Hugging Face, use official repository and file metadata, recording repository ID, immutable
  revision when available, filename, size, LFS/object identifier or published hash, license, and
  gating/authentication state.
- For package-bundled resources, record official package name/version, resource path, package
  metadata, file size, license, and a package-published checksum if one exists.
- For direct HTTPS assets, use the official immutable metadata page or documentation and record the
  exact URL, filename, size, access requirements, license, and published hashes without requesting
  the asset body.

Prefer official APIs or immutable metadata pages, and do not treat a third-party mirror as
authoritative when an official host exists. Never include authentication tokens in commands, logs,
URLs, or reports. Inaccessible metadata remains unresolved. A host-published checksum is structured
metadata with an explicit algorithm and `published_not_locally_verified` state; MD5, SHA-1, SHA-256,
SHA-512, and BLAKE2b are not interchangeable. In particular, published MD5 does not satisfy later
local SHA-256 verification.

## Source chronology

Keep checkpoint/source revision selection candidate-specific. Prefer an explicit host revision,
associated signed or annotated tag, or authoritative cited commit. Repository chronology can support
an inference, but nearest preceding commit is only a candidate and current default-branch HEAD is
current-source evidence, never automatic checkpoint-equivalence evidence.
