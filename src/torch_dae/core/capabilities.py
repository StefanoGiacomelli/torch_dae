"""Capability declarations."""

from dataclasses import dataclass

from torch_dae.core.errors import UnsupportedCapabilityError


@dataclass(frozen=True)
class Capability:
    """Represent support for one model operation.

    Attributes
    ----------
    supported
        Whether callers may request the operation.
    reason
        Optional human-readable explanation, especially when unsupported.
    """

    supported: bool
    reason: str | None = None

    def require(self, name: str) -> None:
        """Require this capability before performing ``name``.

        Parameters
        ----------
        name
            User-facing operation name included in any error.

        Raises
        ------
        UnsupportedCapabilityError
            If :attr:`supported` is false.
        """

        if not self.supported:
            detail = f": {self.reason}" if self.reason else ""
            raise UnsupportedCapabilityError(f"{name} is not supported{detail}")


@dataclass(frozen=True)
class ModelCapabilities:
    """Collect the four generic execution capability declarations.

    Attributes
    ----------
    random_initialization
        Support for constructing random weights.
    checkpoint_loading
        Support for loading pretrained state.
    probabilities
        Support for probability output.
    embeddings
        Support for selectable embeddings.
    """

    random_initialization: Capability
    checkpoint_loading: Capability
    probabilities: Capability
    embeddings: Capability
