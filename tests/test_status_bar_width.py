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
