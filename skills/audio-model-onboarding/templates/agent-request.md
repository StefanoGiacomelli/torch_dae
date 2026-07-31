Use the canonical `audio-model-onboarding` skill available in this repository.

MODE: <analyze | resolve-environment | integrate | verify | card>
WORKFLOW_ID: <STABLE_WORKFLOW_ID_OR_AUTO_DISCOVER>

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
- Run `uv run python scripts/check_worktree_patch.py --json` before phase completion so untracked
  non-ignored outputs receive staged-equivalent whitespace validation without changing the real index
- Discover and validate accepted prerequisite handoffs before requesting attachments
- Promote accepted pre-runtime outputs before declaring a phase complete
- Declare exact artifact supersessions when a later phase changes a shared repository output
- Generate the deterministic review bundle and run scoped cleanup
- Execute only the requested workflow mode
- Do not create a Git commit
- Request user input only for genuine unresolved decisions
