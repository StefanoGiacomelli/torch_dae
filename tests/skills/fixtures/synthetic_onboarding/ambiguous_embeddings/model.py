import torch


class AmbiguousEmbeddingModel(torch.nn.Module):
    def forward(self, waveform):
        frame_features = waveform
        pooled_features = frame_features.mean(dim=-1)
        classifier_input = pooled_features
        logits = classifier_input
        post_classifier_output = logits.sigmoid()
        return {
            "frame_features": frame_features,
            "pooled_features": pooled_features,
            "classifier_input": classifier_input,
            "logits": logits,
            "post_classifier_output": post_classifier_output,
        }
