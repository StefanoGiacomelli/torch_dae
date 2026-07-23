"""Project-specific exceptions."""


class TorchDaeError(Exception):
    """Base error for torch-dae."""


class NotImplementedInPhaseError(TorchDaeError):
    """Raised when a public interface is intentionally deferred."""


class UnsupportedCapabilityError(TorchDaeError):
    """Raised when a model lacks a requested declared capability."""


class MissingModelRuntimeDependencyError(TorchDaeError):
    """Raised when optional model-runtime packages such as PyTorch are missing."""


class DuplicateModelCardError(TorchDaeError):
    """Raised when registry discovery finds duplicate card identifiers."""
