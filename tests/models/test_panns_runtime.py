from __future__ import annotations

import gc

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchlibrosa")

from torch_dae.core.errors import UnsupportedCapabilityError  # noqa: E402
from torch_dae.models.panns.model import (  # noqa: E402
    PannsCnn14_16kMap0438,
    PannsResNet38Map0434,
    PannsWavegramLogmelCnn14Map0439,
    _PannsAudioTagger,
)


@pytest.mark.parametrize(
    ("model_class", "sample_rate"),
    [
        (PannsCnn14_16kMap0438, 16_000),
        (PannsResNet38Map0434, 32_000),
        (PannsWavegramLogmelCnn14Map0439, 32_000),
    ],
)
def test_random_initialized_synthetic_forward(
    model_class: type[_PannsAudioTagger], sample_rate: int
) -> None:
    torch.manual_seed(0)
    model = model_class.from_random()
    model.eval()
    assert next(model.parameters()).device.type == "cpu"
    sample_count = sample_rate * 64 // 100
    waveform = torch.linspace(-0.5, 0.5, sample_count).reshape(1, 1, -1)
    valid_lengths = torch.tensor([sample_count])

    with torch.no_grad():
        first = model(waveform, sample_rate, valid_lengths=valid_lengths)
        second = model(waveform, sample_rate, valid_lengths=valid_lengths)

    logits = first.tensors["logits"]
    probabilities = first.tensors["probabilities"]
    embedding = first.tensors["embedding"]
    assert logits.shape == (1, 527)
    assert probabilities.shape == (1, 527)
    assert embedding.shape == (1, 2048)
    assert torch.isfinite(logits).all()
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
    torch.testing.assert_close(torch.sigmoid(logits), probabilities, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(first.primary, second.primary, rtol=0, atol=0)
    torch.testing.assert_close(
        first.tensors["embedding"], second.tensors["embedding"], rtol=0, atol=0
    )
    assert first.native_output["clipwise_output"] is probabilities
    assert model.available_embeddings()[0].dimension == 2048
    assert model.default_embedding_id == "panns-official-post-fc1-embedding"
    assert len(model.class_labels) == 527

    del first, second, model
    gc.collect()


@pytest.mark.parametrize(
    ("model_class", "sample_rate"),
    [
        (PannsCnn14_16kMap0438, 16_000),
        (PannsResNet38Map0434, 32_000),
        (PannsWavegramLogmelCnn14Map0439, 32_000),
    ],
)
def test_strict_input_contract(model_class: type[_PannsAudioTagger], sample_rate: int) -> None:
    model = model_class.from_random()
    sample_count = sample_rate * 64 // 100
    waveform = torch.zeros(1, 1, sample_count)

    with pytest.raises(ValueError, match="sample_rate"):
        model.preprocess(waveform, sample_rate + 1)
    with pytest.raises(ValueError, match="C=1"):
        model.preprocess(torch.zeros(1, 2, sample_count), sample_rate)
    with pytest.raises(TypeError, match="floating-point"):
        model.preprocess(torch.zeros(1, 1, sample_count, dtype=torch.int16), sample_rate)
    with pytest.raises(UnsupportedCapabilityError, match="padded-batch semantics"):
        model.preprocess(
            waveform,
            sample_rate,
            valid_lengths=torch.tensor([sample_count - 1]),
        )

    del model
    gc.collect()


@pytest.mark.parametrize(
    ("model_class", "sample_rate"),
    [
        (PannsCnn14_16kMap0438, 16_000),
        (PannsResNet38Map0434, 32_000),
        (PannsWavegramLogmelCnn14Map0439, 32_000),
    ],
)
def test_random_initialized_duration_sweep(
    model_class: type[_PannsAudioTagger], sample_rate: int
) -> None:
    torch.manual_seed(0)
    model = model_class.from_random().eval()
    observed_samples = []
    for numerator, denominator in ((64, 100), (1, 1), (2, 1)):
        sample_count = sample_rate * numerator // denominator
        waveform = torch.zeros(1, 1, sample_count)
        with torch.no_grad():
            output = model(waveform, sample_rate)
        assert output.tensors["logits"].shape == (1, 527)
        assert output.tensors["probabilities"].shape == (1, 527)
        assert output.tensors["embedding"].shape == (1, 2048)
        observed_samples.append(sample_count)
    assert observed_samples == [sample_rate * 64 // 100, sample_rate, sample_rate * 2]

    del model
    gc.collect()


@pytest.mark.parametrize(
    ("model_class", "sample_rate", "minimum_samples"),
    [
        (PannsCnn14_16kMap0438, 16_000, 4_960),
        (PannsResNet38Map0434, 32_000, 9_920),
        (PannsWavegramLogmelCnn14Map0439, 32_000, 10_236),
    ],
)
def test_duration_minimum_is_explicit(
    model_class: type[_PannsAudioTagger], sample_rate: int, minimum_samples: int
) -> None:
    model = model_class.from_random()
    with pytest.raises(ValueError, match=rf"at least {minimum_samples} waveform samples"):
        model.preprocess(torch.zeros(1, 1, minimum_samples - 1), sample_rate)

    del model
    gc.collect()


def test_wavegram_branch_alignment_is_explicit() -> None:
    model = PannsWavegramLogmelCnn14Map0439.from_random()
    with pytest.raises(ValueError, match=r"T % 640 must be 0\.\.319 or 636\.\.639"):
        model.preprocess(torch.zeros(1, 1, 10_560), 32_000)

    del model
    gc.collect()
