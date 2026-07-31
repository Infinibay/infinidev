"""The pair-programming critic, folded into one line.

The critic is advisory and it talks a lot — a recommendation on most
steps, sometimes several per step. Rendered as ordinary system messages
each verdict cost a name header, a model/source line and the full body,
so a single step could push the assistant's actual reply off screen. In
amber, which is what the severity colours used to be, it also won every
contest for the eye against the content the user came to read.

So it renders the way consecutive tool calls already do: one collapsed
summary line, click to expand into one line per verdict, click a verdict
to read it in full. Same idea, same shape, same affordance — a user who
has learned ``✓ Ran 3 tools ▸`` already knows how this works.

Two deliberate differences from the tool group:

- **A lone verdict still gets the compact treatment.** Tool groups fall
  back to a full render when there is only one call, but one critic
  verdict is exactly as interruptive as three; that is the case in the
  screenshot that prompted this.
- **Severity survives collapse.** The summary line counts rejects
  separately, because a reject changed what the model did and is worth
  opening, while a recommendation usually is not.
"""

from __future__ import annotations

from typing import Any, Callable

from infinidev.ui.controls.message_widgets import RenderResult
from infinidev.ui.theme import (
    CRITIC_INFO,
    CRITIC_RECOMMEND,
    CRITIC_REJECT,
    TEXT_MUTED,
)

# Glyphs mirror the tool group's vocabulary so the two read as one system.
_ARROW_COLLAPSED = "▸"
_ARROW_EXPANDED = "▾"
_ICON = "◇"
_ICON_REJECT = "◆"

_SUMMARY_FG = TEXT_MUTED
_DIM_FG = TEXT_MUTED

_SEVERITY: dict[str, tuple[str, str]] = {
    # action → (label, colour)
    "reject": ("REJECT", CRITIC_REJECT),
    "recommendation": ("RECOMMEND", CRITIC_RECOMMEND),
    "information": ("INFO", CRITIC_INFO),
}


def _severity(msg: dict[str, Any]) -> tuple[str, str]:
    """(label, colour) for one verdict, tolerant of an unknown action."""
    action = str(msg.get("critic_action") or "information")
    return _SEVERITY.get(action, _SEVERITY["information"])


def _wrap(text: str, width: int) -> list[str]:
    """Soft-wrap a verdict body, preserving its blank-line structure."""
    import textwrap

    out: list[str] = []
    for paragraph in text.splitlines():
        if not paragraph.strip():
            out.append("")
            continue
        out.extend(textwrap.wrap(paragraph, max(20, width)) or [""])
    return out


def _oneliner(msg: dict[str, Any], width: int) -> str:
    """First sentence-ish of a verdict, for the expanded index line."""
    text = " ".join((msg.get("text") or "").split())
    if len(text) <= width:
        return text
    return text[: max(8, width - 1)].rstrip() + "…"


def build_critic_group(
    messages: list[dict[str, Any]],
    *,
    collapsed: bool,
    expanded_set: set,
    width: int,
    on_toggle_group: Callable[[], None],
    on_toggle_item: Callable[[int], None],
) -> RenderResult:
    """Render a run of critic verdicts as one compact, collapsible group.

    ``clickable_offsets`` are wired like the tool group's: offset 0 toggles
    the whole group, and each index line toggles that verdict's body.
    """
    n = len(messages)
    n_reject = sum(
        1 for m in messages if str(m.get("critic_action")) == "reject"
    )

    icon = _ICON_REJECT if n_reject else _ICON
    icon_style = CRITIC_REJECT if n_reject else _DIM_FG
    unit = "note" if n == 1 else "notes"
    reject_note = f"   ·  {n_reject} reject" if n_reject else ""
    arrow = _ARROW_COLLAPSED if collapsed else _ARROW_EXPANDED

    lines: list[list[tuple[str, str]]] = []
    clickable: dict[int, Callable[[], None]] = {}

    # ── Summary line (offset 0) — click toggles the whole group ──────────
    lines.append([
        ("", "  "),
        (icon_style, icon + " "),
        (_SUMMARY_FG, f"Critic · {n} {unit}{reject_note}"),
        ("", "  "),
        (_DIM_FG, arrow),
    ])
    clickable[0] = on_toggle_group

    # ── Expanded: one index line per verdict (+ optional full body) ──────
    if not collapsed:
        for i, msg in enumerate(messages):
            label, colour = _severity(msg)
            is_exp = i in expanded_set
            twirl = _ARROW_EXPANDED if is_exp else _ARROW_COLLAPSED
            model = str(msg.get("critic_model") or "")
            source = str(msg.get("critic_source") or "")
            meta = f" · re: {source}" if source and source != "tools" else ""
            head = f"{label}{meta}"
            preview = _oneliner(msg, max(12, width - len(head) - 16))

            off = len(lines)
            lines.append([
                ("", "     "),
                (colour, head),
                (_DIM_FG, f"  {preview}"),
                ("", "  "),
                (_DIM_FG, twirl),
            ])
            clickable[off] = (lambda idx=i: on_toggle_item(idx))

            if is_exp:
                if model:
                    lines.append([("", "        "), (_DIM_FG, f"({model})")])
                for body_line in _wrap(msg.get("text") or "", width - 10):
                    lines.append([("", "        "), (_SUMMARY_FG, body_line)])

    # trailing chat separator
    lines.append([("", "")])
    return RenderResult(lines=lines, clickable_offsets=clickable)
