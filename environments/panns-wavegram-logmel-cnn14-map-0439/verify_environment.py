from __future__ import annotations

from importlib import metadata

from torch_dae.models.panns.model import PannsWavegramLogmelCnn14Map0439

assert metadata.version("numpy") == "2.4.6"
assert metadata.version("torch") == "2.13.0"
assert metadata.version("torchlibrosa") == "0.1.0"

model = PannsWavegramLogmelCnn14Map0439.from_random()
assert next(model.parameters()).device.type == "cpu"
print("PANNs Wavegram_Logmel_Cnn14 wrapper import and constructor verification passed")
