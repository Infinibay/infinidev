"""Hidden behavioural contract for report options.

Withheld from the agent's workspace and restored only to be run. Every case goes
through the public ``render`` entry point, so any correct fix passes: the check
is the behaviour the request states, not the location of the defect.
"""

import sys

from src.report import render

CASES = (
    # values, overrides, expected
    ([1.2345], None, "1.23"),
    ([1.2345], {"precision": 3}, "1.234"),
    ([1.2, 3.456], {"separator": " | "}, "1.20 | 3.46"),
    ([1.2345], {"precision": 3, "separator": " | "}, "1.234"),
    ([1.2345], {"precision": 0}, "1"),
    ([1.0, 2.0], None, "1.00, 2.00"),
    ([], {"precision": 3}, ""),
    ([1.2345], {}, "1.23"),
)

failures: list[str] = []
for values, overrides, expected in CASES:
    try:
        observed = render(values, overrides)
    except Exception as exc:  # noqa: BLE001 - report and keep going
        failures.append(f"render({values}, {overrides}) raised {exc!r}")
        continue
    if observed != expected:
        failures.append(
            f"render({values}, {overrides}) == {observed!r}, expected {expected!r}"
        )

if failures:
    for failure in failures:
        print(failure)
    sys.exit(1)

print(f"{len(CASES)} option cases passed")
