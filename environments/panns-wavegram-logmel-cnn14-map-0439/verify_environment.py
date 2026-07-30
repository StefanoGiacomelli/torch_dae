from __future__ import annotations

from importlib import metadata

from torch_dae.models.panns._vendor.models import Wavegram_Logmel_Cnn14

assert metadata.version("torch") == "2.13.0"
assert metadata.version("torchlibrosa") == "0.1.0"

model = Wavegram_Logmel_Cnn14(
    sample_rate=32000,
    window_size=1024,
    hop_size=320,
    mel_bins=64,
    fmin=50,
    fmax=14000,
    classes_num=527,
)
assert next(model.parameters()).device.type == "cpu"
print("PANNs Wavegram_Logmel_Cnn14 import and constructor verification passed")
