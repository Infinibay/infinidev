"""Infinidev TUI Theme — single source of truth for all colors, styles, and dimensions.

Import from this module instead of hardcoding color values. To create a new
theme, copy this file and change the values — no other files contain color
literals.

prompt_toolkit styles use the format: "fg_hex bg:bg_hex bold italic underline"
"""

from __future__ import annotations

# ── Message styles ──────────────────────────────────────────────────────
#
# Transcript-first design: messages carry NO background fill. Coloured
# blocks fight the terminal's own theme, break text selection in most
# emulators, and make a long conversation read as a stack of boxes rather
# than a conversation. Identity comes from a 1-cell gutter mark plus the
# text colour — the same approach every modern coding CLI converged on.
#
# Setting a *_BG to "" makes the renderer inherit the terminal background.

MSG_USER_FG = "#e8e8e8"
MSG_USER_BG = ""
MSG_USER_BORDER = "#5ad48a"

MSG_AGENT_FG = "#d6dbe4"
MSG_AGENT_BG = ""
MSG_AGENT_BORDER = "#6c8fc7"

MSG_SYSTEM_FG = "#c9a227"
MSG_SYSTEM_BG = ""
MSG_SYSTEM_BORDER = "#c9a227"

MSG_THINK_FG = "#8a8fa3"
MSG_THINK_BG = ""
MSG_THINK_BORDER = "#6b6f80"

MSG_PENDING_FG = "#7f9f8a"
MSG_PENDING_BG = ""
MSG_PENDING_BORDER = "#4c7a5f"

MSG_QUEUED_FG = "#7a7a7a"
MSG_QUEUED_BG = ""
MSG_QUEUED_BORDER = "#c9a227"

# ── Sender name colors ──────────────────────────────────────────────────

SENDER_COLORS = {
    "user": "#5ad48a",
    "agent": "#6c8fc7",
    "system": "#7a7a7a",
}

NAME_COLORS = {
    "Tool": "#9b8ec4",
    "Step": "#5e9bcf",
    "Reviewer": "#d4a05a",
    "Verifier": "#5ab87a",
    "Shell": "#a0a0a0",
    "System": "#888888",
    # Pair-programming critic verdicts — one color per severity so
    # the user can scan and spot rejects/recs at a glance.
    "Assistant · REJECT": "#ff5577",
    "Assistant · RECOMMEND": "#ffaa44",
    "Assistant · INFO": "#5ab8d4",
}

# ── Diff colors ─────────────────────────────────────────────────────────

DIFF_REMOVED = "#d96b6b"
DIFF_ADDED = "#5ad48a"
DIFF_HUNK = "#6c8fc7"
DIFF_HEADER = "#7a8090"
DIFF_TITLE_FG = "#c9a227"
DIFF_TITLE_BG = "#1c1a12"

# ── Progress / status bars ──────────────────────────────────────────────

PROGRESS_GOOD = "#5ad48a"
PROGRESS_WARNING = "#c9a227"
PROGRESS_CRITICAL = "#d96b6b"

# ── Chrome (layout surfaces, borders, text) ─────────────────────────────
#
# Chrome sits close to the terminal background instead of competing with
# it. SURFACE stays empty ("") wherever the transcript is: the chat area
# must feel like the terminal, not like a widget painted on top of it.

SURFACE = "#16181d"       # modal dialogs — the one place a solid fill helps
SURFACE_DARK = "#14161a"  # status line
SURFACE_DARKER = "#0f1114"
SURFACE_LIGHT = "#191c21"  # composer / panels

PRIMARY = "#6c8fc7"
PRIMARY_DARK = "#4d6b99"
PRIMARY_DARKER = "#3a5273"

ACCENT = "#c9a227"

TEXT = "#d6dbe4"
TEXT_MUTED = "#7a8090"
TEXT_DIM = "#4a5060"

WARNING = "#c9a227"
ERROR = "#d96b6b"
SUCCESS = "#5ad48a"

# ── Scrollbar ───────────────────────────────────────────────────────────

SCROLLBAR_BG = "#14161a"
SCROLLBAR_FG = "#4d6b99"

# ── Thinking indicator ──────────────────────────────────────────────────

THINKING_FG = "#7a8090"

# ── Explorer ────────────────────────────────────────────────────────────

EXPLORER_TITLE_FG = "#d6dbe4"
EXPLORER_TITLE_BG = "#191c21"
EXPLORER_TREE_GUIDE = "#3a5273"
EXPLORER_HIDDEN = "#4a5060"

# ── Image viewer ────────────────────────────────────────────────────────

IMAGE_VIEWER_BG = "#0f1114"

# ── Shell mode input ────────────────────────────────────────────────

SHELL_INPUT_BG = "#1c1315"
SHELL_INPUT_FG = "#e8dcdc"
SHELL_BORDER_COLOR = "#d96b6b"
SHELL_LABEL_FG = "#d96b6b"

# ── Composer (the input box at the bottom of the transcript) ────────────

COMPOSER_BORDER = "#3a4050"
COMPOSER_BORDER_FOCUS = "#6c8fc7"
COMPOSER_PROMPT = "#6c8fc7"
COMPOSER_PLACEHOLDER = "#4a5060"

# ── Dimensions ──────────────────────────────────────────────────────────

# Panel widths in columns. Both used to be ~30% of the terminal, which on
# a 200-column screen meant a 60-column file tree — far more than a file
# tree needs, taken straight out of the transcript. Fixed widths sized to
# their content read better and stay predictable when the window resizes.
EXPLORER_WIDTH = 26
SIDEBAR_WIDTH = 30
SIDEBAR_WIDTH_PERCENT = 30  # legacy alias, kept for external callers
CHAT_INPUT_HEIGHT = 3
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
