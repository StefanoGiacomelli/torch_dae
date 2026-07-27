"""Root control-plane package for torch-dae.

repository foundation exposes contracts and registry helpers without importing PyTorch.
"""

from torch_dae.core.registry import ModelCardRegistry

__all__ = ["ModelCardRegistry", "__version__"]

__version__ = "0.1.0"
