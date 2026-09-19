Use the canonical `audio-model-profiling` skill available in this repository.

MODE: <resolve | plan | profile | validate | finalize>

MODEL_CARD_ID: <RUNTIME_VERIFIED_MODEL_CARD_ID_OR_NA>
DEVICES: <auto | cpu | mps | cuda | cuda:index | comma-separated-list | NA>
ENERGY: <auto | off | NA>
OUTPUT_DIR: <repository-relative-candidate-directory | NA>
ALLOW_PRIVILEGED_ENERGY: <true | false>
CANDIDATE_DIRECTORY: <existing-candidate-directory | NA>

ADDITIONAL_CONSTRAINTS:
<OPTIONAL_CONSTRAINTS_OR_NONE>

Requirements:

- Consume only an accepted `runtime_verified` Model Card; never mutate it
- Execute only the requested profiling mode
- In `resolve`, report eligibility and the accepted model/runtime contract only
- In `plan`, report the exact Profiling v1 campaign and stop before execution
- In `profile`, keep generated evidence in a repository-local candidate/workspace path outside
  `technical_cards/`
- Never write candidate evidence directly into the canonical `technical_cards/` tree
- Use Profiling v1 (`audio-inference-v1`) without changing its fixed warmup/repetition/batch/input
  methodology
- Never silently substitute an explicit failed device with CPU
- Never request privileged hardware counters unless `ALLOW_PRIVILEGED_ENERGY: true`
- Never edit sudoers, persist credentials, or perform silent geolocation
- Preserve unavailable/partial energy, FLOPs, and accelerator-memory evidence honestly
- Validate every generated Technical Card together with its raw NPZ asset
- Report campaign-result, Technical Card JSON, and NPZ paths explicitly
- Do not promote candidate evidence and do not create a Git commit
