"""Curation rule shared by every loader and by the Vib-v1 held-out builder (frozen at stage S2)."""

from __future__ import annotations

from .schema import Observation


def usable(o: Observation, exclude: list[str]) -> bool:
    """Not flagged with any excluded quality flag, and with a stated standard uncertainty."""
    return not any(f in o.quality_flags for f in exclude) and o.uncertainty.standard_sd() is not None
