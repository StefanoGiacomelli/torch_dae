"""Root control-plane package for torch-dae.

Phase 00 exposes contracts and registry helpers without importing PyTorch.
"""

from torch_dae.core.registry import ModelCardRegistry

__all__ = ["ModelCardRegistry", "__version__"]

__version__ = "0.1.0"
