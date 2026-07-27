Use the canonical `audio-model-onboarding` skill available in this repository.

MODE: <analyze | resolve-environment | integrate | verify | card>

MODEL_NAME: <MODEL_NAME>
UPSTREAM_REPOSITORY: <GITHUB_REPOSITORY_URL>
PAPER_OR_TECHNICAL_REFERENCE: <PAPER_URL_OR_NONE>

TARGET_VARIANT: <VARIANT_NAME_OR_AUTO_DISCOVER>
TARGET_CHECKPOINT: <CHECKPOINT_NAME_OR_AUTO_DISCOVER>
PREFERRED_EMBEDDING: <EMBEDDING_NAME_OR_UNRESOLVED>

ADDITIONAL_CONSTRAINTS:
<OPTIONAL_PROJECT_SPECIFIC_CONDITIONING_OR_NONE>

Requirements:

- Preserve the model-agnostic root environment
- Use the repository's canonical contracts, templates, and lifecycle
- Treat upstream code as static input until controlled execution is explicitly authorized
- Do not silently select an ambiguous variant, checkpoint, source strategy, or embedding
- Record evidence and provenance for every material conclusion
- Use isolated model-specific environments
- Validate every generated artifact
- Execute only the requested workflow mode
- Do not create a Git commit
- Request user input only for genuine unresolved decisions
