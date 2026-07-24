from __future__ import annotations

import pytest

from torch_dae.core.capabilities import Capability, ModelCapabilities
from torch_dae.core.errors import UnsupportedCapabilityError
from torch_dae.core.model import AudioModelProtocol
from torch_dae.core.outputs import AudioModelOutput, EmbeddingOutput, PreprocessingOutput
from torch_dae.core.preprocessing import WaveformInputContract


class Tensor:
    @property
    def shape(self) -> tuple[int, ...]:
        return (2, 1, 16000)


def test_canonical_input_contract() -> None:
    contract = WaveformInputContract()
    assert contract.waveform_shape == "B,C,T"
    assert contract.valid_lengths_shape == "B"
    assert contract.valid_lengths_optional


def test_output_contracts() -> None:
    tensor = Tensor()
    assert AudioModelOutput(primary=tensor, tensors={"x": tensor}).tensors["x"].shape == (
        2,
        1,
        16000,
    )
    assert EmbeddingOutput("e", tensor, "B,D").layout == "B,D"
    output = PreprocessingOutput(model_input={"waveform": tensor}, sample_rate=16000)
    assert output.sample_rate == 16000
    assert output.model_input["waveform"].shape == (2, 1, 16000)


def test_probability_capability_error() -> None:
    capabilities = ModelCapabilities(
        random_initialization=Capability(True),
        checkpoint_loading=Capability(True),
        probabilities=Capability(False, "no probability head"),
        embeddings=Capability(True),
    )
    with pytest.raises(UnsupportedCapabilityError, match="no probability head"):
        capabilities.probabilities.require("predict_probability")


def test_audio_model_protocol_stub_methods_are_importable() -> None:
    assert AudioModelProtocol.from_random() is None
    assert AudioModelProtocol.from_pretrained() is None
    assert AudioModelProtocol.load_checkpoint(object(), "checkpoint") is None
    assert AudioModelProtocol.preprocess(object(), Tensor(), 16000) is None
    assert AudioModelProtocol.forward(object(), Tensor(), 16000) is None
    assert AudioModelProtocol.predict_probability(object(), Tensor(), 16000) is None
    assert AudioModelProtocol.available_embeddings(object()) is None
    assert AudioModelProtocol.compute_embedding(object(), Tensor(), 16000) is None
