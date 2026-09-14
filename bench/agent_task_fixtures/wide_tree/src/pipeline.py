"""The running total, stage by stage."""

from __future__ import annotations

from src.mod_01 import step_1
from src.mod_02 import step_2
from src.mod_03 import step_3
from src.mod_04 import step_4
from src.mod_05 import step_5
from src.mod_06 import step_6
from src.mod_07 import step_7
from src.mod_08 import step_8
from src.mod_09 import step_9
from src.mod_10 import step_10
from src.mod_11 import step_11
from src.mod_12 import step_12
from src.mod_13 import step_13
from src.mod_14 import step_14
from src.mod_15 import step_15
from src.mod_16 import step_16
from src.mod_17 import step_17
from src.mod_18 import step_18
from src.mod_19 import step_19
from src.mod_20 import step_20
from src.mod_21 import step_21
from src.mod_22 import step_22
from src.mod_23 import step_23
from src.mod_24 import step_24
from src.mod_25 import step_25
from src.mod_26 import step_26
from src.mod_27 import step_27
from src.mod_28 import step_28
from src.mod_29 import step_29
from src.mod_30 import step_30
from src.mod_31 import step_31
from src.mod_32 import step_32
from src.mod_33 import step_33
from src.mod_34 import step_34
from src.mod_35 import step_35
from src.mod_36 import step_36
from src.mod_37 import step_37
from src.mod_38 import step_38
from src.mod_39 import step_39
from src.mod_40 import step_40


def pipeline(start: int) -> int:
    """Push ``start`` through every stage in order."""
    value = start
    value = step_1(value)
    value = step_2(value)
    value = step_3(value)
    value = step_4(value)
    value = step_5(value)
    value = step_6(value)
    value = step_7(value)
    value = step_8(value)
    value = step_9(value)
    value = step_10(value)
    value = step_11(value)
    value = step_12(value)
    value = step_13(value)
    value = step_14(value)
    value = step_15(value)
    value = step_16(value)
    value = step_17(value)
    value = step_18(value)
    value = step_19(value)
    value = step_20(value)
    value = step_21(value)
    value = step_22(value)
    value = step_23(value)
    value = step_24(value)
    value = step_25(value)
    value = step_26(value)
    value = step_27(value)
    value = step_28(value)
    value = step_29(value)
    value = step_30(value)
    value = step_31(value)
    value = step_32(value)
    value = step_33(value)
    value = step_34(value)
    value = step_35(value)
    value = step_36(value)
    value = step_37(value)
    value = step_38(value)
    value = step_39(value)
    value = step_40(value)
    return value


def lowest_possible() -> int:
    """The smallest value the pipeline is allowed to return."""
    return 0
