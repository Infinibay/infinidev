"""One hub: five packages applied in order."""

from __future__ import annotations

from ..pkgs.pkg_00 import apply as pkg_0
from ..pkgs.pkg_01 import apply as pkg_1
from ..pkgs.pkg_02 import apply as pkg_2
from ..pkgs.pkg_03 import apply as pkg_3
from ..pkgs.pkg_04 import apply as pkg_4

_PACKAGES = (pkg_0, pkg_1, pkg_2, pkg_3, pkg_4,)


def apply(v: int) -> int:
    """Push ``v`` through hub 0's packages."""
    for package in _PACKAGES:
        v = package(v)
    return v
