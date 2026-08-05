from pathlib import Path


plan = Path("PLAN.md")
assert plan.is_file(), "PLAN.md was not created"
text = plan.read_text(encoding="utf-8").lower()
for required in (
    "retention",
    "authorization",
    "cancel",
    "idempot",
    "audit",
    "compatib",
    "cleanup",
    "handoff",
    "phase",
    "progress",
    "telemetry",
    "rollout",
    "rollback",
    "test",
):
    assert required in text, f"PLAN.md omits {required}"
assert any(marker in text for marker in ("open decision", "user decision", "confirm"))
assert not Path("src").exists(), "planning task must not add implementation"
