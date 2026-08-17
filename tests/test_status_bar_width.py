"""Width-clamping/truncation for the status bar and footer (D3).

These two single-line bars must never overflow the terminal width.
The clamp path only runs when a column count is available, so the
tests monkeypatch ``terminal_cols`` (real app returns a width; headless
tests would otherwise get None and skip clamping).
"""

from __future__ import annotations

from prompt_toolkit.utils import get_cwidth

from infinidev.ui.controls import status_bar_control, footer_control
from infinidev.ui.controls.status_bar_control import StatusBarControl
from infinidev.ui.controls.footer_control import FooterControl
from infinidev.ui.controls._widthutil import truncate_cells


def _cells(formatted_text) -> int:
    return sum(get_cwidth(t) for _, t in formatted_text)


def _joined(formatted_text) -> str:
    return "".join(t for _, t in formatted_text)


# ── truncate_cells ──────────────────────────────────────────────────

def test_truncate_cells_single_column_is_just_ellipsis():
    assert truncate_cells("abcdef", 1) == "…"


def test_truncate_cells_fits_within_budget_with_ellipsis():
    out = truncate_cells("abcdef", 4)
    assert out == "abc…"
    assert get_cwidth(out) <= 4


def test_truncate_cells_noop_when_it_fits():
    assert truncate_cells("abc", 10) == "abc"


def test_truncate_cells_counts_wide_chars():
    # Two CJK chars = 4 cells; budget 3 leaves room for 1 char + ellipsis.
    out = truncate_cells("漢字", 3)
    assert get_cwidth(out) <= 3
    assert out.endswith("…")


# ── status bar ──────────────────────────────────────────────────────

def test_status_bar_never_exceeds_width_and_keeps_live_status(monkeypatch):
    monkeypatch.setattr(status_bar_control, "terminal_cols", lambda: 40)
    bar = StatusBarControl()
    bar.set_model("ollama/some-very-long-model-name:7b-instruct")
    bar.set_project("/home/andres/Proyects/very/long/path/infinidev")
    bar.set_status("Cancelling...")

    ft = bar._get_text()
    joined = _joined(ft)
    assert _cells(ft) <= 40, joined
    # The live status is preserved intact (it carries the cancel indicator)…
    assert "Cancelling..." in joined
    # …while the project path is the first segment to be dropped…
    assert "/home/andres" not in joined
    # …and the model is truncated with an ellipsis rather than dropped.
    assert "…" in joined


def test_status_bar_unclamped_when_no_app(monkeypatch):
    monkeypatch.setattr(status_bar_control, "terminal_cols", lambda: None)
    bar = StatusBarControl()
    bar.set_model("m")
    bar.set_project("/p")
    bar.set_status("ok")
    ft = bar._get_text()
    joined = _joined(ft)
    assert "m" in joined and "/p" in joined and "ok" in joined


# ── mode badge (autonomous / chain indicator) ─────────────────────

def test_mode_badge_renders_after_brand(monkeypatch):
    """``set_mode`` must render the label after the brand, visible in the bar."""
    monkeypatch.setattr(status_bar_control, "terminal_cols", lambda: None)
    bar = StatusBarControl()
    bar.set_mode("AUTO 2/3 · 12k/50k", "active")
    bar.set_model("foo")
    joined = _joined(bar._get_text())
    assert "infinidev" in joined
    assert "AUTO 2/3 · 12k/50k" in joined
    # The badge is rendered as the first segment after the brand so the user
    # sees it at a glance — check it precedes the model name in source order.
    assert joined.index("AUTO 2/3 · 12k/50k") < joined.index("foo")


def test_mode_badge_survives_transient_set_status(monkeypatch):
    """The mode badge is a persistent channel; ``set_status`` must not wipe it.

    This is the property that fixes the user-reported "no AUTO indicator"
    complaint: cancel-hold progress and flash_status both call
    ``set_status`` with a transient label, and the existing channel is
    overwritten by those. The autonomous chain indicator lives on a
    separate channel so those callers cannot erase it.
    """
    monkeypatch.setattr(status_bar_control, "terminal_cols", lambda: None)
    bar = StatusBarControl()
    bar.set_mode("AUTO 1/3", "active")
    # Simulate the cancel-hold progress path: set_status replaces the
    # transient channel. The mode badge must remain visible.
    bar.set_status("Hold Esc: cancel task [████░░░░░░]")
    joined = _joined(bar._get_text())
    assert "AUTO 1/3" in joined, joined
    assert "Hold Esc: cancel task" in joined


def test_mode_badge_clear(monkeypatch):
    """Passing ``""`` clears the badge — the next render must not show it."""
    monkeypatch.setattr(status_bar_control, "terminal_cols", lambda: None)
    bar = StatusBarControl()
    bar.set_mode("AUTO 1/3", "active")
    assert "AUTO 1/3" in _joined(bar._get_text())
    bar.set_mode("", "idle")
    assert "AUTO 1/3" not in _joined(bar._get_text())


def test_mode_badge_uses_distinct_colours_for_active_vs_idle(monkeypatch):
    """``active`` kind uses ACCENT; ``idle`` kind uses PRIMARY.

    The colour distinction is what makes the badge legible at a glance —
    the user needs to see ACCENT and immediately know "chain is running".
    """
    from infinidev.ui.theme import ACCENT, PRIMARY

    monkeypatch.setattr(status_bar_control, "terminal_cols", lambda: None)
    bar = StatusBarControl()
    bar.set_mode("AUTO 1/3", "active")
    frags = bar._get_text()
    modes = [(style, label) for style, label in frags if label == "AUTO 1/3"]
    assert modes, "mode badge missing from render"
    style, _ = modes[0]
    assert ACCENT in style, f"expected ACCENT in active style, got {style!r}"

    bar.set_mode("phase: planning", "idle")
    frags = bar._get_text()
    modes = [(style, label) for style, label in frags if label == "phase: planning"]
    assert modes, "idle mode badge missing from render"
    style, _ = modes[0]
    assert PRIMARY in style, f"expected PRIMARY in idle style, got {style!r}"
    assert ACCENT not in style, "idle mode must not use ACCENT colour"


# ── footer ──────────────────────────────────────────────────────────

def test_footer_never_exceeds_width_and_drops_trailing_hints(monkeypatch):
    monkeypatch.setattr(footer_control, "terminal_cols", lambda: 24)
    footer = FooterControl(app_state=None)  # None → show all hints
    ft = footer._get_text()
    assert _cells(ft) <= 24

    # Compare against the unclamped render: clamping must drop some hints.
    monkeypatch.setattr(footer_control, "terminal_cols", lambda: None)
    full = footer._get_text()
    assert _cells(ft) < _cells(full)


def test_footer_unclamped_shows_everything(monkeypatch):
    monkeypatch.setattr(footer_control, "terminal_cols", lambda: None)
    footer = FooterControl(app_state=None)
    ft = footer._get_text()
    # No clamp → at least one hint rendered.
    assert len(ft) > 0


def test_footer_keeps_stop_hint_under_truncation(monkeypatch):
    # The stop/cancel affordance must survive a narrow terminal — it is
    # ordered near the front so trailing (lower-value) hints drop first.
    monkeypatch.setattr(footer_control, "terminal_cols", lambda: 40)
    footer = FooterControl(app_state=None)
    joined = _joined(footer._get_text())
    assert "Stop/close" in joined        # the stop hint is retained…
    assert "Chat" not in joined          # …while trailing hints are dropped
