# Quickstart

This quickstart uses only accepted repository artifacts and does not require importing a model
runtime until the inference step.

## 1. Inspect the supported registry

From the repository root:

```bash
uv run torch-dae card list
```

Expected Model Card IDs:

```text
panns-cnn14-16k-map-0438
panns-resnet38-map-0434
panns-wavegram-logmel-cnn14-map-0439
```

Inspect one card:

```bash
uv run torch-dae card show panns-cnn14-16k-map-0438
```

The normalized JSON contains the checkpoint-specific identity, environment, waveform contract,
outputs, default embedding, verified devices, runtime-verification references, and limitations.

Validate it explicitly if needed:

```bash
uv run torch-dae card validate panns-cnn14-16k-map-0438
```

## 2. Inspect accepted profiling evidence

```bash
uv run torch-dae technical-card list
```

The repository currently contains nine canonical PANNs Technical Cards: CPU `single_thread`, CPU
`native_default`, and Apple MPS evidence for each accepted PANNs Model Card.

Profiling evidence is independent from model acceptance. A Model Card remains the source of truth
for what model/checkpoint is supported; Technical Cards describe empirical execution contexts.

## 3. Choose what you want to do next

### Run a supported PANNs checkpoint

Follow {doc}`../tutorials/panns-inference`. It shows how to:

1. materialize the accepted model environment;
2. acquire the authoritative checkpoint;
3. run Python inside the isolated environment;
4. obtain logits, probabilities, labels, and the `[B,2048]` embedding.

### Understand input/output semantics first

Read {doc}`../user-guide/model-execution` and {doc}`../user-guide/embeddings`.

### Integrate a new model

Start with {doc}`../skill/overview` and {doc}`../tutorials/audio-model-onboarding`.

### Profile an accepted model

Start with {doc}`../profiling/overview`.

## Important execution boundary

Do not expect the root development environment to import every supported audio model. The root is a
model-agnostic control plane. Model execution takes place inside the accepted isolated environment.

For the current PANNs integrations, managed environment materialization is constrained to macOS on
arm64. Registry inspection and static validation remain useful on other platforms.
