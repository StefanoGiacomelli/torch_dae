import torch


class ClearAudioModel(torch.nn.Module):
    def preprocess(self, waveform, sample_rate):
        return waveform

    def forward(self, waveform):
        logits = waveform.mean(dim=-1)
        return {"logits": logits}


CHECKPOINT_URL = "https://example.invalid/releases/clear-audio.pt"
