# Contributing profiling evidence

Profiling evidence follows a two-stage model:

```text
profile locally
    -> candidate Technical Card JSON + NPZ
    -> validate and independently review
    -> controlled promotion into technical_cards/<model-id>/
    -> normal pull request
```

The profiler never writes directly to the canonical tree, and profiling never modifies the
referenced Model Card.

## 1. Start from an accepted model

The Model Card must already be `runtime_verified`. If onboarding is incomplete, finish the
onboarding workflow first; profiling is not a substitute for runtime verification.

## 2. Use a candidate workspace

Choose a repository-local directory outside `technical_cards/`, for example:

```text
profiling_candidates/<campaign-name>/
```

Example:

```bash
uv run torch-dae model profile \
  --model <model-card-id> \
  --device auto \
  --protocol audio-inference-v1 \
  --energy auto \
  --output-dir profiling_candidates/<campaign-name> \
  --json
```

The repository root, canonical `technical_cards/` directory, descendants of that directory, and
paths resolving outside the repository are rejected.

## 3. Validate every candidate pair

For each generated JSON:

```bash
uv run torch-dae technical-card validate \
  profiling_candidates/<campaign-name>/<technical-card-id>.json \
  --json
```

Keep the JSON and its referenced `.npz` together. Validation must pass after the pair is relocated
together because the raw asset path is card-relative.

## 4. Review the campaign, not only the exit code

A valid card can still contain limitations that matter scientifically. Review at least:

- requested, detected, attempted, successful, and failed devices;
- CPU thread-regime coverage;
- unsupported benchmark conditions;
- minimum-input search status;
- latency variability and raw samples;
- host/accelerator memory semantics;
- energy measurement kind and `coverage_complete`;
- `unaccounted_components` and explicit limitations;
- runtime classification and execution context.

Use `.torch-dae/profiling/<campaign-id>/campaign-result.json` to understand campaign orchestration
and device smoke failures. Use the Technical Card JSON/NPZ pair as the evidence being considered for
promotion.

## 5. Promotion is a separate review decision

Do not choose `technical_cards/` as the profiler output directory and do not treat a generated
candidate as canonical merely because it validates.

After independent review accepts the session, promotion copies the reviewed JSON/NPZ pair into:

```text
technical_cards/<model-id>/<technical-card-id>.json
technical_cards/<model-id>/<technical-card-id>.npz
```

This is currently a controlled maintainer/reviewer operation rather than an automatic profiler
step. Preserve the reviewed bytes exactly. Do not edit the referenced Model Card as part of the
promotion.

## 6. Pull-request scope

A profiling-evidence pull request should contain only the accepted Technical Card pairs required by
the contribution:

```text
technical_cards/<model-id>/<technical-card-id>.json
technical_cards/<model-id>/<technical-card-id>.npz
```

If code, protocol, schema, documentation, or profiler behavior must also change, treat that as a
separate software change and regenerate/review profiling evidence under the resulting runtime when
appropriate.

## 7. Corrections are append-only

Do not overwrite an accepted Technical Card. Generate a new card and, when appropriate, use its
supersession fields to point to the earlier card with an explicit reason.

## Suggested review prompt

The copy-paste prompt in {doc}`../skill/prompt-library` can ask the profiling skill to validate and
summarize a candidate campaign without promoting it.

For an end-to-end walkthrough, see {doc}`../tutorials/profiling`.
