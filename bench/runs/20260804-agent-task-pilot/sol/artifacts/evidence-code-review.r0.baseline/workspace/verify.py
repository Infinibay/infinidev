from pathlib import Path


review = Path("REVIEW.md")
assert review.is_file(), "REVIEW.md was not created"
text = review.read_text(encoding="utf-8").lower()
assert "token" in text and any(word in text for word in ("plain", "constant", "timing", "hash"))
assert "exception" in text and any(word in text for word in ("allow", "true", "bypass", "fail open"))
assert "log" in text or "audit" in text
assert any(word in text for word in ("secret", "credential", "supplied_token"))
assert any(word in text for word in ("global", "cache", "typing", "type"))
assert any(word in text for word in ("blocker", "critical", "high"))
