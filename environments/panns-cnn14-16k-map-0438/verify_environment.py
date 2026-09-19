from __future__ import annotations

from importlib import metadata

from torch_dae.models.panns.model import PannsCnn14_16kMap0438

assert metadata.version("numpy") == "2.4.6"
assert metadata.version("torch") == "2.13.0"
assert metadata.version("torchlibrosa") == "0.1.0"

model = PannsCnn14_16kMap0438.from_random()
assert next(model.parameters()).device.type == "cpu"
print("PANNs Cnn14_16k wrapper import and constructor verification passed")
