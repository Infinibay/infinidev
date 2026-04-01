"""Infinidev TUI Theme — single source of truth for all colors, styles, and dimensions.

Import from this module instead of hardcoding color values. To create a new
theme, copy this file and change the values — no other files contain color
literals.

prompt_toolkit styles use the format: "fg_hex bg:bg_hex bold italic underline"
"""

from __future__ import annotations

# ── Message styles ──────────────────────────────────────────────────────

MSG_USER_FG = "#a8ffc8"
MSG_USER_BG = "#0a1a0a"
MSG_USER_BORDER = "#2df97f"

MSG_AGENT_FG = "#d0e4ff"
MSG_AGENT_BG = "#0a101a"
MSG_AGENT_BORDER = "#4da6ff"

MSG_SYSTEM_FG = "#ffcc4d"
MSG_SYSTEM_BG = "#1a1500"
MSG_SYSTEM_BORDER = "#ffaa00"

MSG_THINK_FG = "#c0b8e0"
MSG_THINK_BG = "#12101a"
MSG_THINK_BORDER = "#8a70c8"

MSG_PENDING_FG = "#7aad7a"
MSG_PENDING_BG = "#0a1a0a"
MSG_PENDING_BORDER = "#2a8a4f"

MSG_QUEUED_FG = "#888888"
MSG_QUEUED_BG = "#161616"
MSG_QUEUED_BORDER = "#ffaa00"

# ── Sender name colors ──────────────────────────────────────────────────

SENDER_COLORS = {
    "user": "#6fbf6f",
    "agent": "#7a9fd4",
    "system": "#888888",
}

NAME_COLORS = {
    "Tool": "#9b8ec4",
    "Step": "#5e9bcf",
    "Reviewer": "#d4a05a",
    "Verifier": "#5ab87a",
    "Shell": "#a0a0a0",
    "System": "#888888",
}

# ── Diff colors ─────────────────────────────────────────────────────────

DIFF_REMOVED = "#ff5577"
DIFF_ADDED = "#00ee77"
DIFF_HUNK = "#55aaff"
DIFF_HEADER = "#888888"
DIFF_TITLE_FG = "#ff8800"
DIFF_TITLE_BG = "#1a1200"

# ── Progress / status bars ──────────────────────────────────────────────

PROGRESS_GOOD = "#44ff44"
PROGRESS_WARNING = "#ffaa00"
PROGRESS_CRITICAL = "#ff4444"

# ── Chrome (layout surfaces, borders, text) ─────────────────────────────

SURFACE = "#1e1e1e"
SURFACE_DARK = "#161616"
SURFACE_DARKER = "#111111"
SURFACE_LIGHT = "#262626"

PRIMARY = "#4da6ff"
PRIMARY_DARK = "#3a7fbf"
PRIMARY_DARKER = "#2a5f8f"

ACCENT = "#ffaa00"

TEXT = "#cccccc"
TEXT_MUTED = "#888888"
TEXT_DIM = "#555555"

WARNING = "#ffaa00"
ERROR = "#ff4444"
SUCCESS = "#2df97f"

# ── Scrollbar ───────────────────────────────────────────────────────────

SCROLLBAR_BG = "#161616"
SCROLLBAR_FG = "#4da6ff"

# ── Thinking indicator ──────────────────────────────────────────────────

THINKING_FG = "#7b9fdf"

# ── Explorer ────────────────────────────────────────────────────────────

EXPLORER_TITLE_FG = "#ffffff"
EXPLORER_TITLE_BG = "#4da6ff"
EXPLORER_TREE_GUIDE = "#2a5f8f"
EXPLORER_HIDDEN = "#888888"

# ── Image viewer ────────────────────────────────────────────────────────

IMAGE_VIEWER_BG = "#111111"

# ── Dimensions ──────────────────────────────────────────────────────────

EXPLORER_WIDTH = 30
SIDEBAR_WIDTH_PERCENT = 30
CHAT_INPUT_HEIGHT = 4
STATUS_BAR_HEIGHT = 1
SIDEBAR_PANEL_MAX_LINES = 8
CONTEXT_PANEL_HEIGHT = 5
BAR_WIDTH = 8          # for context usage bars
AUTOCOMPLETE_MAX_HEIGHT = 8

# ── Modal dimensions ────────────────────────────────────────────────────

MODAL_OVERLAY_BG = "#000000"

MODEL_PICKER_WIDTH = 60
MODEL_PICKER_MAX_HEIGHT = 20

SETTINGS_WIDTH_PCT = 85
SETTINGS_HEIGHT_PCT = 80
SETTINGS_SECTIONS_WIDTH = 22

PERM_DETAIL_WIDTH_PCT = 80
PERM_DETAIL_HEIGHT_PCT = 80

SETTING_EDITOR_WIDTH_PCT = 60
SETTING_EDITOR_MAX_HEIGHT_PCT = 50

FINDINGS_WIDTH_PCT = 90
FINDINGS_HEIGHT_PCT = 85
FINDINGS_LIST_WIDTH_PCT = 40

DOCS_WIDTH_PCT = 90
DOCS_HEIGHT_PCT = 85
DOCS_LIB_WIDTH_PCT = 30
DOCS_SECTION_WIDTH_PCT = 25

PROJECT_SEARCH_WIDTH_PCT = 90
PROJECT_SEARCH_HEIGHT_PCT = 85

UNSAVED_BOX_WIDTH = 55
CANCEL_BOX_WIDTH = 50


# ── Style helpers ───────────────────────────────────────────────────────

def style(fg: str | None = None, bg: str | None = None,
          bold: bool = False, italic: bool = False,
          underline: bool = False, dim: bool = False) -> str:
    """Build a prompt_toolkit style string from components.

    >>> style(fg="#ff0000", bold=True)
    '#ff0000 bold'
    >>> style(fg="#aabbcc", bg="#000000", italic=True)
    '#aabbcc bg:#000000 italic'
    """
    parts: list[str] = []
    if fg:
        parts.append(fg)
    if bg:
        parts.append(f"bg:{bg}")
    if bold:
        parts.append("bold")
    if italic:
        parts.append("italic")
    if underline:
        parts.append("underline")
    if dim:
        # prompt_toolkit doesn't have native dim, approximate via color
        pass
    return " ".join(parts)


# ── Pre-built style strings ────────────────────────────────────────────

STYLE_USER_MSG = style(fg=MSG_USER_FG, bg=MSG_USER_BG)
STYLE_AGENT_MSG = style(fg=MSG_AGENT_FG, bg=MSG_AGENT_BG)
STYLE_SYSTEM_MSG = style(fg=MSG_SYSTEM_FG, bg=MSG_SYSTEM_BG, italic=True)
STYLE_THINK_MSG = style(fg=MSG_THINK_FG, bg=MSG_THINK_BG, italic=True)
STYLE_PENDING_MSG = style(fg=MSG_PENDING_FG, bg=MSG_PENDING_BG)
STYLE_QUEUED_MSG = style(fg=MSG_QUEUED_FG, bg=MSG_QUEUED_BG)

STYLE_USER_HEADER = style(fg=MSG_USER_FG, bold=True)
STYLE_AGENT_HEADER = style(fg=MSG_AGENT_FG, bold=True)
STYLE_SYSTEM_HEADER = style(fg=MSG_SYSTEM_FG, bold=True, italic=True)
STYLE_THINK_HEADER = style(fg=MSG_THINK_FG, bold=True, italic=True)

STYLE_STATUS_BAR = style(fg=TEXT_MUTED, bg=SURFACE_DARK)
STYLE_SIDEBAR_TITLE = style(fg="#ffffff", bg=PRIMARY, bold=True)
STYLE_SIDEBAR_CONTENT = style(fg=TEXT, bg=SURFACE_LIGHT)

STYLE_BORDER = style(fg=PRIMARY)
STYLE_BORDER_ACTIVE = style(fg=ACCENT)

STYLE_DIFF_REMOVED = style(fg=DIFF_REMOVED)
STYLE_DIFF_ADDED = style(fg=DIFF_ADDED)
STYLE_DIFF_HUNK = style(fg=DIFF_HUNK, bold=True)
STYLE_DIFF_HEADER = style(fg=DIFF_HEADER, bold=True)
STYLE_DIFF_TITLE = style(fg=DIFF_TITLE_FG, bg=DIFF_TITLE_BG)

STYLE_THINKING = style(fg=THINKING_FG)

# ── Border characters ──────────────────────────────────────────────────

BORDER_VERTICAL = "│"
BORDER_HORIZONTAL = "─"
BORDER_CORNER_TL = "┌"
BORDER_CORNER_TR = "┐"
BORDER_CORNER_BL = "└"
BORDER_CORNER_BR = "┘"
BORDER_TEE_LEFT = "├"
BORDER_TEE_RIGHT = "┤"

# Block characters for progress bars
BAR_FILLED = "█"
BAR_EMPTY = "░"
