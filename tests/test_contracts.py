from __future__ import annotations

import pytest

from torch_dae.contracts import ensure_canonical_id


@pytest.mark.parametrize("value", ["card-1", "model.variant_2", "a", "abc.def-ghi_jkl9"])
def test_canonical_ids_accept_safe_values(value: str) -> None:
    assert ensure_canonical_id(value) == value


@pytest.mark.parametrize(
    "value",
    ["../escape", "a/b", "a\\b", "has space", "-leading", "trailing-", "UPPERCASE"],
)
def test_canonical_ids_reject_unsafe_values(value: str) -> None:
    with pytest.raises(ValueError):
        ensure_canonical_id(value)
