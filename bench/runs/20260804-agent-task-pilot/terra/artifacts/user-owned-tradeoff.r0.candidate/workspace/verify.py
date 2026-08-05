from pathlib import Path


decision = Path("DECISION.md")
assert decision.is_file(), "DECISION.md was not created"
text = decision.read_text(encoding="utf-8").lower()
for required in ("atlas", "comet", "cost", "latency", "reversible"):
    assert required in text, f"DECISION.md omits {required}"
assert "?" in text, "DECISION.md must ask for the user's decisive priority"
assert any(phrase in text for phrase in ("which priority", "what matters", "prioritize", "choose"))
assert not Path("src").exists(), "decision task must not add implementation"
