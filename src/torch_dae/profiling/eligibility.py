"""Shared Model Card lifecycle eligibility rules for Profiling v1."""

from __future__ import annotations

from torch_dae.cards.models import ModelCard, ModelCardLifecycle


def require_profiling_eligible_model_card(model_card: ModelCard) -> None:
    """Require the sole lifecycle state from which Profiling v1 may run.

    ``profiled`` remains parseable in schema 1 for legacy compatibility, but the project
    specification explicitly excludes it from new Technical Card profiling.
    """

    if model_card.card_status != ModelCardLifecycle.RUNTIME_VERIFIED:
        raise ValueError(
            "Model Card is not eligible for profiling: "
            f"expected card_status='runtime_verified', got {model_card.card_status.value!r}"
        )
