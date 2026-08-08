"""Terminal run summaries must reflect the actual terminal state."""

from __future__ import annotations

from infinidev.engine import engine_logging


def test_finish_summary_does_not_label_blocked_run_completed(monkeypatch) -> None:
    lines: list[str] = []
    monkeypatch.setattr(engine_logging, "log", lines.append)

    engine_logging.log_finish("developer", "blocked", 2, 23, 252377)

    rendered = "\n".join(lines)
    assert "Blocked" in rendered
    assert "Completed" not in rendered


def test_finish_summary_labels_done_run_completed(monkeypatch) -> None:
    lines: list[str] = []
    monkeypatch.setattr(engine_logging, "log", lines.append)

    engine_logging.log_finish("developer", "done", 1, 10, 1000)

    assert "Completed" in "\n".join(lines)
