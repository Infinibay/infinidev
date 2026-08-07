"""Thinking transcript messages are visible only after explicit opt-in."""

from __future__ import annotations

from infinidev.config.settings import Settings, settings
from infinidev.ui.controls.chat_history import ChatHistoryControl
from infinidev.ui.dialogs.settings_editor_state import SETTINGS_SECTIONS


def _rendered_text(control: ChatHistoryControl, width: int = 80) -> str:
    lines, line_count, custom_get_line = control._build_lines(width)

    def get_line(index: int) -> list[tuple[str, str]]:
        if custom_get_line is not None:
            return custom_get_line(index)
        return lines[index]

    return "\n".join(
        "".join(text for _style, text in get_line(index))
        for index in range(line_count)
    )


def test_show_thinking_in_chat_is_disabled_by_default() -> None:
    field = Settings.model_fields["UI_SHOW_THINKING_IN_CHAT"]

    assert field.default is False


def test_thinking_section_exposes_chat_visibility_toggle() -> None:
    rows = {key: setting_type for key, _label, setting_type in SETTINGS_SECTIONS["Thinking"]}

    assert rows["UI_SHOW_THINKING_IN_CHAT"] == "bool"


def test_chat_hides_thinking_until_setting_is_enabled(monkeypatch) -> None:
    messages = [
        {"sender": "Thinking", "text": "private scratchpad", "type": "think"},
        {"sender": "Assistant", "text": "visible answer", "type": "agent"},
    ]
    control = ChatHistoryControl(messages)

    monkeypatch.setattr(settings, "UI_SHOW_THINKING_IN_CHAT", False)
    hidden = _rendered_text(control)
    assert "private scratchpad" not in hidden
    assert "visible answer" in hidden

    # Changing /settings invalidates the app, but not necessarily this control's
    # line cache. The setting must therefore participate in the cache key.
    monkeypatch.setattr(settings, "UI_SHOW_THINKING_IN_CHAT", True)
    visible = _rendered_text(control)
    assert "private scratchpad" in visible
    assert "visible answer" in visible
