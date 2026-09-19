"""Public root package for the torch-dae control plane.

The root package intentionally stays lightweight: importing :mod:`torch_dae` exposes registry
helpers and package metadata without importing model-runtime dependencies such as PyTorch.
"""

from importlib.metadata import PackageNotFoundError, version

from torch_dae.core.registry import ModelCardRegistry

__all__ = ["ModelCardRegistry", "__version__"]

try:
    __version__ = version("torch-deepaudioembedding")
except PackageNotFoundError:  # pragma: no cover - source tree imported outside an installed env
    __version__ = "0+unknown"
