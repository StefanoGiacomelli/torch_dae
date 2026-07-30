from __future__ import annotations

from importlib import metadata

from torch_dae.models.panns._vendor.models import Cnn14_16k

assert metadata.version("torch") == "2.13.0"
assert metadata.version("torchlibrosa") == "0.1.0"

model = Cnn14_16k(
    sample_rate=16000,
    window_size=512,
    hop_size=160,
    mel_bins=64,
    fmin=50,
    fmax=8000,
    classes_num=527,
)
assert next(model.parameters()).device.type == "cpu"
print("PANNs Cnn14_16k import and constructor verification passed")
