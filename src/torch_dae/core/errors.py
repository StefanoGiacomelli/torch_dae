"""Project-specific exceptions."""


class TorchDaeError(Exception):
    """Base error for torch-dae."""


class FeatureNotAvailableError(TorchDaeError):
    """Indicate that a visible public operation is intentionally unavailable.

    This error distinguishes an explicit placeholder from an unexpected runtime failure. Callers
    may present its message directly as a concise capability explanation.
    """


class UnsupportedCapabilityError(TorchDaeError):
    """Indicate that a requested operation is unsupported by a model integration.

    The exception concerns a capability declared by the checkpoint-specific model card, such as
    probability or embedding output. It does not imply that package dependencies are missing.
    """


class MissingModelRuntimeDependencyError(TorchDaeError):
    """Raised when optional model-runtime packages such as PyTorch are missing."""


class DuplicateModelCardError(TorchDaeError):
    """Raised when registry discovery finds duplicate card identifiers."""


class ExternalCommandError(TorchDaeError):
    """Raised when a managed external command fails."""

    def __init__(self, message: str, *, returncode: int | None = None) -> None:
        super().__init__(message)
        self.returncode = returncode


class UvUnavailableError(ExternalCommandError):
    """Raised when the `uv` executable is unavailable or unusable."""


class GitUnavailableError(ExternalCommandError):
    """Raised when the `git` executable is unavailable or unusable."""


class PythonInterpreterUnavailableError(TorchDaeError):
    """Raised when an exact requested Python interpreter cannot be located."""


class EnvironmentAlreadyExistsError(TorchDaeError):
    """Raised when `env create` targets existing materialization state."""


class EnvironmentNotFoundError(TorchDaeError):
    """Raised when an expected materialized environment is absent."""


class EnvironmentMaterializationError(TorchDaeError):
    """Raised when environment creation fails."""


class EnvironmentVerificationError(TorchDaeError):
    """Raised when environment verification fails."""


class EnvironmentIdentityMismatchError(ValueError, TorchDaeError):
    """Raised when committed environment identities disagree."""


class OfflineResourceUnavailableError(TorchDaeError):
    """Raised when offline mode cannot satisfy a missing resource from cache."""


class SourceMaterializationError(TorchDaeError):
    """Raised when an upstream source cannot be verified or installed."""


class CheckpointAcquisitionError(TorchDaeError):
    """Raised when checkpoint acquisition fails."""


class CheckpointHashMismatchError(CheckpointAcquisitionError):
    """Raised when a checkpoint hash does not match the specification."""


class CheckpointNotFoundError(CheckpointAcquisitionError):
    """Raised when no local checkpoint cache entry is available."""
