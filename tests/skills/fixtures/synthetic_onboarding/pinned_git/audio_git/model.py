import torch


def create_model():
    return GitOnlyAudioModel()


class GitOnlyAudioModel(torch.nn.Module):
    def forward(self, waveform):
        features = waveform.mean(dim=-1)
        return features
