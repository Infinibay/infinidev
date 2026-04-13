"""Lightweight markdown → prompt_toolkit fragment renderer.

Converts a subset of markdown into styled (style, text) tuples for
use in prompt_toolkit's FormattedText. Supports:
- **bold**, *italic*, `inline code`
- # Headings (H1-H3)
- ```code blocks```
- - bullet lists
- > blockquotes
"""

from __future__ import annotations

import re

from infinidev.ui.theme import (
    PRIMARY, ACCENT, TEXT, TEXT_MUTED, SUCCESS,
)

# ── Styles ──────────────────────────────────────────────────────────────

_BOLD = "bold"
_ITALIC = "italic"
_CODE_INLINE = f"{ACCENT}"
_CODE_BLOCK = f"{TEXT_MUTED}"
_HEADING = f"{PRIMARY} bold"
_BLOCKQUOTE = f"{TEXT_MUTED} italic"
_BULLET = f"{SUCCESS} bold"

# Pre-compiled patterns (avoid re.compile() on every line render)
_HEADING_RE = re.compile(r'^(#{1,3})\s+(.+)')
_BULLET_RE = re.compile(r'^(\s*)([-*])\s+(.+)')
_NUMLIST_RE = re.compile(r'^(\s*)(\d+)[.)]\s+(.+)')
_INLINE_RE = re.compile(
    r'(\*\*(.+?)\*\*)'    # **bold**
    r'|(\*(.+?)\*)'       # *italic*
    r'|(`(.+?)`)'         # `code`
    r'|([^*`]+)'          # plain text
)


def render_markdown_line(line: str, base_style: str, bg_part: str) -> list[tuple[str, str]]:
    """Render a single line of markdown into styled fragments.

    Returns a list of (style, text) tuples. The base_style and bg_part
    are inherited from the message container for consistency.
    """
    stripped = line.rstrip()

    # Code block fence — rendered as a dim separator
    if stripped.startswith("```"):
        lang = stripped[3:].strip()
        label = f"  {lang}" if lang else ""
        style = f"{_CODE_BLOCK} {bg_part}" if bg_part else _CODE_BLOCK
        return [(style, f"───{label}")]

    # Headings
    heading_match = _HEADING_RE.match(stripped)
    if heading_match:
        style = f"{_HEADING} {bg_part}" if bg_part else _HEADING
        return [(style, heading_match.group(2))]

    # Blockquotes
    if stripped.startswith("> "):
        style = f"{_BLOCKQUOTE} {bg_part}" if bg_part else _BLOCKQUOTE
        content = stripped[2:]
        return [(style, f"│ {content}")]

    # Bullet lists
    bullet_match = _BULLET_RE.match(stripped)
    if bullet_match:
        indent = bullet_match.group(1)
        content = bullet_match.group(3)
        bullet_style = f"{_BULLET} {bg_part}" if bg_part else _BULLET
        text_frags = _parse_inline(content, base_style, bg_part)
        return [(bullet_style, f"{indent}• ")] + text_frags

    # Numbered lists
    num_match = _NUMLIST_RE.match(stripped)
    if num_match:
        indent = num_match.group(1)
        num = num_match.group(2)
        content = num_match.group(3)
        num_style = f"{_BULLET} {bg_part}" if bg_part else _BULLET
        text_frags = _parse_inline(content, base_style, bg_part)
        return [(num_style, f"{indent}{num}. ")] + text_frags

    # Regular line — parse inline formatting
    return _parse_inline(stripped, base_style, bg_part)


def _parse_inline(text: str, base_style: str, bg_part: str) -> list[tuple[str, str]]:
    """Parse inline markdown: **bold**, *italic*, `code`."""
    fragments: list[tuple[str, str]] = []

    for m in _INLINE_RE.finditer(text):
        if m.group(2) is not None:
            # **bold**
            style = f"{base_style} bold"
            fragments.append((style, m.group(2)))
        elif m.group(4) is not None:
            # *italic*
            style = f"{base_style} italic"
            fragments.append((style, m.group(4)))
        elif m.group(6) is not None:
            # `code`
            style = f"{_CODE_INLINE} {bg_part}" if bg_part else _CODE_INLINE
            fragments.append((style, m.group(6)))
        elif m.group(7) is not None:
            # plain text
            fragments.append((base_style, m.group(7)))

    if not fragments:
        fragments.append((base_style, text))

    return fragments
