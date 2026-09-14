"""The running total, hub by hub."""

from __future__ import annotations

from .hubs import hub_0
from .hubs import hub_1
from .hubs import hub_2
from .hubs import hub_3
from .hubs import hub_4
from .hubs import hub_5
from .hubs import hub_6
from .hubs import hub_7

_HUBS = (hub_0, hub_1, hub_2, hub_3, hub_4, hub_5, hub_6, hub_7,)


def pipeline(start: int) -> int:
    """Push ``start`` through every hub in order."""
    value = start
    for hub in _HUBS:
        value = hub.apply(value)
    return value


def lowest_possible() -> int:
    """The smallest value the pipeline is allowed to return."""
    return 0
