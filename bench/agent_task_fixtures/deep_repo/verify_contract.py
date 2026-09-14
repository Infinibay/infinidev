"""Hidden behavioural contract for the aggregation pipeline.

Withheld from the agent's workspace and restored only to be run. It checks the
total the request states, so any fix that restores it passes regardless of which
leaf it lands in.
"""

import sys

from src.pipeline import pipeline

#: The sum of every leaf's declared offset, and the shortfall the broken leaf
#: introduces.
EXPECTED_TOTAL = 14398

CASES = (0, 5, -10, 100, -500)

failures: list[str] = []
for start in CASES:
    expected = start + EXPECTED_TOTAL
    try:
        observed = pipeline(start)
    except Exception as exc:  # noqa: BLE001 - report and keep going
        failures.append(f"pipeline({start}) raised {exc!r}")
        continue
    if observed != expected:
        failures.append(f"pipeline({start}) == {observed}, expected {expected}")

if failures:
    for failure in failures:
        print(failure)
    sys.exit(1)

print(f"{len(CASES)} pipeline cases passed")
