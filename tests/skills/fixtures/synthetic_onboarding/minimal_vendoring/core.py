import torch


class StableSubset(torch.nn.Module):
    def forward(self, waveform):
        return {"logits": waveform.sum(dim=-1)}
