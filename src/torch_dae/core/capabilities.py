"""Capability declarations."""

from dataclasses import dataclass

from torch_dae.core.errors import UnsupportedCapabilityError


@dataclass(frozen=True)
class Capability:
    """Boolean capability with an optional reason."""

    supported: bool
    reason: str | None = None

    def require(self, name: str) -> None:
        """Raise when the capability is not supported."""

        if not self.supported:
            detail = f": {self.reason}" if self.reason else ""
            raise UnsupportedCapabilityError(f"{name} is not supported{detail}")


@dataclass(frozen=True)
class ModelCapabilities:
    """Model-independent capability set."""

    random_initialization: Capability
    checkpoint_loading: Capability
    probabilities: Capability
    embeddings: Capability
