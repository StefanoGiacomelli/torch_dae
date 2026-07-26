import torch


class ExternalTorchModel(torch.nn.Module):
    def forward(self, waveform):
        return {"representation": waveform.mean(dim=-1)}
