# CLI reference

The `torch-dae` command is the repository-backed control-plane interface. Use `--help` at any
level for the authoritative option set, and use `--version` to report the installed package
version.

```bash
torch-dae --version
torch-dae --help
torch-dae <group> --help
torch-dae <group> <command> --help
```

The CLI operates on the current `torch-dae` repository workspace whenever a command needs Model
Cards, environment definitions, checkpoint specifications, runtime targets, or Technical Cards.
See {doc}`../getting-started/installation` for the distinction between a package installation and a
full repository workspace.

## Model Cards

```text
torch-dae card list
torch-dae card show <card-id>
torch-dae card validate <card-id-or-path>
```

- `list` reports accepted Model Card identifiers.
- `show` prints the normalized card JSON.
- `validate` accepts either a registered id or an explicit JSON path.

## Environments

```text
torch-dae env create <card-id>
torch-dae env ensure <card-id>
torch-dae env resolve <environment-id>
torch-dae env preflight <environment-id>
torch-dae env materialize <environment-id>
torch-dae env verify <environment-id>
torch-dae env remove <card-id>
torch-dae env info <card-id>
torch-dae env run <card-id> -- <command>
```

The two families are intentionally different:

- `resolve`, `preflight`, `materialize`, and `verify` act directly on accepted environment
  definitions;
- `create`, `ensure`, `remove`, `info`, and `run` start from a Model Card and follow its accepted
  environment reference.

`env preflight` is network-free and does not materialize an environment. It checks active local
wheel requirements against packages reachable from the accepted lock. `env run` executes the
command after `--` inside the managed model environment.

## Checkpoints

```text
torch-dae checkpoint ensure <card-id> [--offline] [--json]
torch-dae checkpoint info <card-id> [--json]
torch-dae checkpoint remove <card-id> [--json]

torch-dae checkpoint resolve --spec <checkpoint-spec.json> [--offline] [--json]
torch-dae checkpoint ensure-spec --spec <checkpoint-spec.json> \
  [--offline] [--maximum-bytes N] [--json]
torch-dae checkpoint info-spec --spec <checkpoint-spec.json> [--json]
```

Card-based commands use the checkpoint declared by an accepted Model Card. The `--spec` commands
are card-independent:

- `resolve` resolves authoritative metadata without acquiring payload bytes;
- `ensure-spec` acquires and verifies the explicit specification through the canonical manager;
- `info-spec` inspects authority/cache state without network access.

Model wrappers do not silently download checkpoints. Acquire the payload first, then pass the
materialized local path to the wrapper.

## Runtime verification

```text
torch-dae model verify --target <runtime-target.json> [--offline] [--json]
```

`model verify` executes a schema-2 runtime target in its accepted model environment before a Model
Card needs to exist. `--offline` forbids network access. See {doc}`../runtime-execution`.

`model inspect` is a reserved placeholder and currently exits with an unavailable-feature error.

## Profiling

```text
torch-dae model profile \
  --model <card-id> \
  [--device auto|cpu|mps|cuda|cuda:<index> ...] \
  [--protocol audio-inference-v1] \
  [--energy auto|off] \
  [--allow-privileged-energy] \
  [--output-dir ./candidate_technical_cards] \
  [--json]
```

The target Model Card must be `runtime_verified`. The command creates candidate Technical Card
JSON/NPZ evidence only; it never writes to the canonical `technical_cards/` tree.

### Controls you choose

- `--model`: accepted Model Card id;
- `--device`: repeatable device selector; omitted means `auto`;
- `--protocol`: currently `audio-inference-v1`;
- `--energy`: `auto` or `off`;
- `--allow-privileged-energy`: explicit consent for a backend that needs local privileged hardware
  counters;
- `--output-dir`: repository-local candidate workspace;
- `--json`: machine-readable campaign result.

Warmups, measured iterations, batch sizes, duration rules, synthetic-input generation, and CPU thread
regimes are fixed by Profiling v1 and are intentionally not CLI controls. See
{doc}`../profiling/protocol`.

## Technical Cards

```text
torch-dae technical-card list [--json]
torch-dae technical-card inspect <technical-card.json>
torch-dae technical-card validate <technical-card.json> [--json]
```

`validate` checks the card schema, identity recomputation, Model Card reference, raw NPZ hash, raw
timing arrays, summary statistics, privacy constraints, and supersession references. The repository
validator applies the same Technical Card validation to every card promoted under
`technical_cards/<model-id>/`; canonical profiling evidence is therefore a CI-protected repository
artifact.

See {doc}`../user-guide/technical-cards` for interpretation rather than schema-level detail.

## Onboarding control-plane script

The model-integration lifecycle also exposes repository-maintenance operations through
`scripts/onboarding_handoff.py`:

```text
uv run python scripts/onboarding_handoff.py discover ...
uv run python scripts/onboarding_handoff.py validate ...
uv run python scripts/onboarding_handoff.py promote ...
uv run python scripts/onboarding_handoff.py bundle ...
uv run python scripts/onboarding_handoff.py cleanup ...
uv run python scripts/onboarding_handoff.py finalize ...
uv run python scripts/onboarding_handoff.py run-manifest create ...
```

These are maintainer/developer operations rather than everyday model-inference commands. `finalize`
is the canonical end-of-phase path: it revalidates accepted evidence, runs required repository
gates, performs cleanup preflight, creates the deterministic external review bundle, and only then
optionally executes managed-workspace cleanup. For the user-oriented integration workflow, start
with {doc}`../skill/overview` and {doc}`../skill/prompt-library`.
