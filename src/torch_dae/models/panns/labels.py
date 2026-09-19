"""Load the ordered AudioSet label metadata packaged with PANNs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

_EXPECTED_CLASS_COUNT = 527


@dataclass(frozen=True)
class AudioSetLabel:
    """One authoritative AudioSet classifier output entry."""

    index: int
    mid: str
    display_name: str


@lru_cache(maxsize=1)
def load_audioset_labels() -> tuple[AudioSetLabel, ...]:
    """Return the authoritative ordered 527-class AudioSet mapping."""

    resource = files("torch_dae.models.panns.resources").joinpath("class_labels_indices.csv")
    with resource.open("r", encoding="utf-8") as stream:
        rows = tuple(
            AudioSetLabel(
                index=int(row["index"]),
                mid=row["mid"],
                display_name=row["display_name"],
            )
            for row in csv.DictReader(stream)
        )

    expected_indices = tuple(range(_EXPECTED_CLASS_COUNT))
    indices = tuple(row.index for row in rows)
    mids = tuple(row.mid for row in rows)
    if len(rows) != _EXPECTED_CLASS_COUNT:
        raise RuntimeError(f"AudioSet metadata must contain {_EXPECTED_CLASS_COUNT} rows")
    if indices != expected_indices:
        raise RuntimeError("AudioSet metadata indices must be contiguous from 0 through 526")
    if len(set(mids)) != _EXPECTED_CLASS_COUNT:
        raise RuntimeError("AudioSet metadata MIDs must be unique")
    return rows


def class_labels() -> tuple[str, ...]:
    """Return AudioSet display names in classifier-index order."""

    return tuple(label.display_name for label in load_audioset_labels())
