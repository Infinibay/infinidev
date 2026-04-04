"""Findings detail control — right panel of the findings browser."""

from __future__ import annotations

from infinidev.ui.theme import TEXT, TEXT_MUTED
from infinidev.ui.controls.scrollable_text import ScrollableTextControl
from infinidev.ui.dialogs.findings_list_control import FindingsListControl


class FindingsDetailControl(ScrollableTextControl):
    """Detail view for the selected finding.

    Subclasses ScrollableTextControl for mouse-wheel scrolling and
    automatic line wrapping via the parent Window's wrap_lines=True.
    """

    def __init__(self, list_ctrl: FindingsListControl) -> None:
        self._list = list_ctrl
        super().__init__(self._get_fragments)

    def _get_fragments(self):
        finding = self._list.get_selected()
        if not finding:
            return [(f"{TEXT_MUTED}", " Select a finding")]

        fragments = [
            (f"{TEXT} bold", f" {finding.get('topic', '?')}\n"),
            (f"{TEXT_MUTED}",
             f" Type: {finding.get('finding_type', '?')} | "
             f"Confidence: {finding.get('confidence', '?')} | "
             f"Status: {finding.get('status', '?')}\n"),
            ("", "\n"),
        ]
        content = finding.get("content", "")
        for line in content.split("\n"):
            fragments.append((f"{TEXT}", f" {line}\n"))
        return fragments


