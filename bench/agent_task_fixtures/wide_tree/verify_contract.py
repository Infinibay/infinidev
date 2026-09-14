"""Hidden behavioural contract for the stage pipeline.

Withheld from the agent's workspace and restored only to be run. It checks the
total the request states, so any fix that restores it passes regardless of which
module it lands in.
"""

import sys

from src.pipeline import pipeline

#: Every stage n adds n, so forty stages add 820.
EXPECTED_SUM = 820

CASES = (0, 5, -10, 100, -500)

failures: list[str] = []
for start in CASES:
    expected = start + EXPECTED_SUM
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
