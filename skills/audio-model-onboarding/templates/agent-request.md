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
- Resolve, materialize, and verify environments by environment ID without requiring a model card
- In verify mode, create strict runtime targets before checkpoint acquisition
- Declare ordered required and optional runtime check IDs and require complete passed required-check
  coverage before success
- Keep environment verification evidence separate from checkpoint-specific runtime reports
- Validate every generated artifact
- Allocate any managed workspace exclusively through
  `uv run python scripts/onboarding_handoff.py run-manifest create` before performing managed
  workspace work, and perform that work inside the returned `run_root`
- Discover and validate accepted prerequisite handoffs before requesting attachments
- Preserve accepted prerequisite control-plane hashes as historical provenance and report drift
- Record and validate the current skill and specification hashes for every pending phase candidate
- Promote accepted phase outputs before declaring a phase complete
- Declare exact artifact supersessions when a later phase changes a shared repository output
- Run the canonical
  `uv run python scripts/onboarding_handoff.py finalize --workflow-id <id> --phase <phase> --json`
  after promotion — it validates evidence invariance, runs the required repository gates
  (`validate_repository.py`, `validate_skill_artifacts.py`, `check_worktree_patch.py --json`,
  `git diff --check`), runs cleanup, and generates the deterministic review bundle; do not hand
  assemble a bundle, hashes, or a separate staged-equivalent check
- Execute only the requested workflow mode
- Do not create a Git commit
- Request user input only for genuine unresolved decisions
