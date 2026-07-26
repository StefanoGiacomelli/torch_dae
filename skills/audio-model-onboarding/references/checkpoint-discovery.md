# Checkpoint Discovery

Collect checkpoint URLs, release assets, Hugging Face references, helper download functions, package
resources, local paths, hashes, filenames, archive behavior, authentication needs, licenses, and
variant mappings.

Never download a real checkpoint in Phase 02 tests. Hidden helper functions should be identified as
evidence, but the helper must not be executed in the root environment.

Checkpoint candidates derived from helpers preserve `helper_symbol`, `expression_status`,
`unresolved_components`, source file, complete URL, filename, and hash evidence. Literal URLs may
leave helper fields unset, but helper-derived URLs must remain tied to the observed helper function
for grounded evaluation. Hashes are associated with a candidate only when the static AST relationship
ties the hash to that helper or checkpoint metadata structure. Repository-global or unrelated-helper
hashes cannot satisfy a candidate. When association is unresolved, the observed hash collection is
empty, `hash association` remains unresolved, and reports must omit the hash.

Checkpoint candidates are not verified until acquired, hashed, and loaded in the intended model
environment during a later lifecycle stage.
