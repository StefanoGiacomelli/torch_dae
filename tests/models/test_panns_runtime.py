from __future__ import annotations

import gc
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("torchlibrosa")

from torch_dae.core.errors import UnsupportedCapabilityError  # noqa: E402
from torch_dae.models.panns.model import (  # noqa: E402
    PannsCnn14_16kMap0438,
    PannsResNet38Map0434,
    PannsWavegramLogmelCnn14Map0439,
    _load_panns_checkpoint_payload,
    _PannsAudioTagger,
)
from torch_dae.models.panns.verification import Provider  # noqa: E402

_AUTHORITATIVE_CNN14_CHECKPOINT = (
    Path(__file__).resolve().parents[2]
    / ".torch-dae/checkpoints/panns-cnn14-16k-map-0438"
    / "e2ee543a27919542c2ea03eabaa70b24dcd4e6c8e05621de6b67a94e4c5058e6"
    / "Cnn14_16k_mAP=0.438.pth"
)


def test_public_loader_uses_scoped_minimal_numpy_safe_globals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "synthetic.pth"
    path.touch()
    state_dict = {"weight": torch.ones(1, 1), "bias": torch.zeros(1)}
    observed: dict[str, object] = {}

    class SafeGlobals:
        def __init__(self, globals_: list[object]):
            observed["globals"] = globals_

        def __enter__(self) -> None:
            observed["entered"] = True

        def __exit__(self, *args: object) -> None:
            observed["exited"] = True

    def load(checkpoint: Path, **kwargs: object) -> object:
        observed["checkpoint"] = checkpoint
        observed["kwargs"] = kwargs
        return {"model": state_dict}

    monkeypatch.setattr(torch.serialization, "safe_globals", SafeGlobals)
    monkeypatch.setattr(torch, "load", load)
    model = _PannsAudioTagger.__new__(_PannsAudioTagger)
    torch.nn.Module.__init__(model)
    model.upstream_model = torch.nn.Linear(1, 1)
    model.load_checkpoint(path, strict=True, map_location="cpu")

    safe_globals = observed["globals"]
    assert isinstance(safe_globals, list)
    assert safe_globals[0][1] == "numpy.core.multiarray._reconstruct"
    assert safe_globals[1:] == [np.ndarray, np.dtype, type(np.dtype(np.int64))]
    assert observed["entered"] is True and observed["exited"] is True
    assert observed["checkpoint"] == path
    assert observed["kwargs"] == {"map_location": "cpu", "weights_only": True}
    assert torch.equal(model.upstream_model.weight, state_dict["weight"])
    assert torch.equal(model.upstream_model.bias, state_dict["bias"])


@pytest.mark.skipif(
    not _AUTHORITATIVE_CNN14_CHECKPOINT.is_file(),
    reason="retained authoritative Cnn14 checkpoint is unavailable",
)
def test_authoritative_legacy_cnn14_checkpoint_strict_loads_publicly() -> None:
    model = PannsCnn14_16kMap0438.from_pretrained(_AUTHORITATIVE_CNN14_CHECKPOINT).eval()
    assert len(model.upstream_model.state_dict()) == 84
    assert model.upstream_model.fc_audioset.weight.shape == (527, 2048)
    del model
    gc.collect()


def test_weights_only_loader_accepts_legacy_numpy_payload_when_available() -> None:
    if not _AUTHORITATIVE_CNN14_CHECKPOINT.is_file():
        pytest.skip("retained authoritative Cnn14 checkpoint is unavailable")
    safe_globals_before = set(torch.serialization.get_safe_globals())
    payload = _load_panns_checkpoint_payload(_AUTHORITATIVE_CNN14_CHECKPOINT, map_location="cpu")
    assert isinstance(payload, dict)
    assert isinstance(payload["sampler"], dict)
    assert all(isinstance(value, np.ndarray) for value in payload["sampler"]["indexes_per_class"])
    assert set(torch.serialization.get_safe_globals()) == safe_globals_before


def test_differentiability_allows_zero_waveform_gradient_with_finite_parameter_gradients() -> None:
    class TinyUpstream(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Linear(1, 1, bias=False)
            self.fc_audioset = torch.nn.Linear(1, 1)

    class TinyWrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.upstream_model = TinyUpstream()

        def forward(self, waveform: torch.Tensor, sample_rate: int) -> object:
            del sample_rate
            feature = self.upstream_model.backbone(torch.ones(1, 1))
            feature = feature + waveform.sum().reshape(1, 1) * 0
            logits = self.upstream_model.fc_audioset(feature)
            return SimpleNamespace(primary=logits, tensors={"embedding": feature})

    provider = Provider.__new__(Provider)
    provider.model = TinyWrapper()
    provider.waveform = torch.ones(1, 1, 4)
    provider.rate = 16_000
    provider.samples = 4

    details = provider.check_cpu_differentiability()
    assert "backbone gradient backbone.weight" in details
    assert "waveform gradient nonzero entries=0/4" in details


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
