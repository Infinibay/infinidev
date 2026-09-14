"""Hidden regression contract for apply_discount.

Withheld from the agent's workspace by the task record and restored only to be
run. It states the rounding rule exactly: the discount is rounded half up to
the nearest whole cent, and the result is an int.
"""

import sys

from src.pricing import apply_discount

CASES = (
    # cents, percent, expected price
    (1000, 10, 900),
    (999, 33, 669),   # 329.67 -> 330, not 329
    (5, 10, 4),       # 0.5    -> 1,   not 0
    (15, 50, 7),      # 7.5    -> 8
    (50, 1, 49),      # 0.5    -> 1
    (1, 1, 1),        # 0.01   -> 0
    (0, 50, 0),
    (100, 100, 0),
    (250, 0, 250),
    (999999, 33, 669999),
)

failures: list[str] = []
for cents, percent, expected in CASES:
    try:
        observed = apply_discount(cents, percent)
    except Exception as exc:  # noqa: BLE001 - report any failure, keep going
        failures.append(f"apply_discount({cents}, {percent}) raised {exc!r}")
        continue
    if observed != expected:
        failures.append(
            f"apply_discount({cents}, {percent}) == {observed}, expected {expected}"
        )
    if type(observed) is not int:
        failures.append(
            f"apply_discount({cents}, {percent}) returned {type(observed).__name__}, "
            "expected int"
        )

if failures:
    for failure in failures:
        print(failure)
    sys.exit(1)

print(f"{len(CASES)} rounding cases passed")
