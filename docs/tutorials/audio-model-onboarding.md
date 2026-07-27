# Audio-model onboarding

Invoke the canonical `audio-model-onboarding` skill explicitly with one workflow mode and one
checkpoint target. The workflow separates evidence collection from operations that execute code or
acquire assets.

1. Use `analyze` for static, evidence-grounded inspection.
2. Use `resolve-environment` for controlled compatibility resolution in isolated runtime state.
3. Use `integrate` only after the required evidence and user decisions exist.
4. Use `verify` for controlled checkpoint loading and runtime behavior.
5. Use `card` to author the checkpoint-specific model card from validated artifacts.

Start from the repository's
[agent request template](https://github.com/StefanoGiacomelli/torch_dae/blob/main/skills/audio-model-onboarding/templates/agent-request.md).
Each invocation remains scoped to its selected mode. Unresolved scientific or technical questions
stay explicit rather than being guessed.

See {doc}`../skill/overview` for the mode-by-mode contract.

The machine contracts behind evidence, claims, candidates, decisions, and reports are documented in
{doc}`../api/onboarding-contracts`. Static calls such as
{func}`torch_dae.onboarding.inspect_repository` and
{func}`torch_dae.onboarding.inspect_scenario_repository` are documented in
{doc}`../api/onboarding-inspection`; they never execute inspected code. Environment and checkpoint
side effects begin only at the explicit manager calls in {doc}`../api/environment-lifecycle` and
{doc}`../api/checkpoints`.
