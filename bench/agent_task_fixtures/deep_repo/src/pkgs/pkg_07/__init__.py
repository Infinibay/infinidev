"""One package: thirty leaves applied in order."""

from __future__ import annotations

from . import mod_00
from . import mod_01
from . import mod_02
from . import mod_03
from . import mod_04
from . import mod_05
from . import mod_06
from . import mod_07
from . import mod_08
from . import mod_09
from . import mod_10
from . import mod_11
from . import mod_12
from . import mod_13
from . import mod_14
from . import mod_15
from . import mod_16
from . import mod_17
from . import mod_18
from . import mod_19
from . import mod_20
from . import mod_21
from . import mod_22
from . import mod_23
from . import mod_24
from . import mod_25
from . import mod_26
from . import mod_27
from . import mod_28
from . import mod_29

_LEAVES = (mod_00, mod_01, mod_02, mod_03, mod_04, mod_05, mod_06, mod_07, mod_08, mod_09, mod_10, mod_11, mod_12, mod_13, mod_14, mod_15, mod_16, mod_17, mod_18, mod_19, mod_20, mod_21, mod_22, mod_23, mod_24, mod_25, mod_26, mod_27, mod_28, mod_29,)


def apply(v: int) -> int:
    """Push ``v`` through package 7's leaves."""
    for leaf in _LEAVES:
        v = leaf.shift(v)
    return v
