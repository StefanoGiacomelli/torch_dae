import numpy as np
import torch
import torchaudio

FRAMEWORKS = (torch.__name__, torchaudio.__name__)


def normalize(value):
    return np.float(value)
