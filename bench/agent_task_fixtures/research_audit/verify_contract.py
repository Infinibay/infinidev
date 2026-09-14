"""Hidden contract for the repository audit.

The report judges the audit; the source tree judges the promise not to touch it.
Nothing here requires a particular wording, only that each module is named and
that every finding carries the line it came from.
"""

import re
import sys
from pathlib import Path

failures: list[str] = []

report = Path("AUDIT.md")
if not report.is_file():
    failures.append("AUDIT.md was not created")
else:
    text = report.read_text(encoding="utf-8")
    lowered = text.lower()
    for module in ("auth.py", "storage.py", "api.py"):
        if module not in lowered:
            failures.append(f"AUDIT.md does not name {module}")
    references = re.findall(r":\d+|\bline\s+\d+", lowered)
    if len(references) < 3:
        failures.append(
            "AUDIT.md cites fewer than three line references "
            f"({len(references)} found); every finding needs its source line"
        )

for module in ("auth", "storage", "api"):
    path = Path("src") / f"{module}.py"
    if path.is_file():
        original = {
            "auth": "SESSION_TTL_SECONDS = 60 * 60 * 24 * 365",
            "storage": "    except Exception:",
            "api": "def create_record(path: str, payload: dict) -> dict:",
        }[module]
        if original not in path.read_text(encoding="utf-8"):
            failures.append(f"src/{module}.py was modified; the task is read-only")

if failures:
    for failure in failures:
        print(failure)
    sys.exit(1)

print("audit contract passed")
