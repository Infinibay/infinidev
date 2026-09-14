"""One hub: five packages applied in order."""

from __future__ import annotations

from ..pkgs.pkg_25 import apply as pkg_0
from ..pkgs.pkg_26 import apply as pkg_1
from ..pkgs.pkg_27 import apply as pkg_2
from ..pkgs.pkg_28 import apply as pkg_3
from ..pkgs.pkg_29 import apply as pkg_4

_PACKAGES = (pkg_0, pkg_1, pkg_2, pkg_3, pkg_4,)


def apply(v: int) -> int:
    """Push ``v`` through hub 5's packages."""
    for package in _PACKAGES:
        v = package(v)
    return v
