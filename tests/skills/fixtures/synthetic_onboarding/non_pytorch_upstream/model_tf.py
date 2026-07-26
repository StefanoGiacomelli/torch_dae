import tensorflow as tf


class OfficialTensorFlowModel(tf.Module):
    def __call__(self, waveform):
        return waveform
