"""Findings detail control — right panel of the findings browser."""

from __future__ import annotations

from prompt_toolkit.layout.controls import FormattedTextControl

from infinidev.ui.theme import TEXT, TEXT_MUTED
from infinidev.ui.dialogs.findings_list_control import FindingsListControl


class FindingsDetailControl(FormattedTextControl):
    """Detail view for the selected finding.

    Uses FormattedTextControl so wrap_lines=True on the parent Window
    works correctly. Focusable is False to avoid stealing focus from
    the list panel.
    """

    def __init__(self, list_ctrl: FindingsListControl) -> None:
        self._list = list_ctrl
        super().__init__(self._build_fragments, focusable=False)

    def _build_fragments(self):
        finding = self._list.get_selected()
        if not finding:
            return [(f"{TEXT_MUTED}", " Select a finding")]

        fragments = [
            (f"{TEXT} bold", f" {finding.get('topic', '?')}"),
            ("", "\n"),
            (f"{TEXT_MUTED}",
             f" Type: {finding.get('finding_type', '?')} | "
             f"Confidence: {finding.get('confidence', '?')} | "
             f"Status: {finding.get('status', '?')}"),
            ("", "\n\n"),
        ]
        content = finding.get("content", "")
        for line in content.split("\n"):
            fragments.append((f"{TEXT}", f" {line}"))
            fragments.append(("", "\n"))
        return fragments
