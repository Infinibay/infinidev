"""Tests for the Infinidev TUI — ChatInput widget and autocomplete."""

from unittest.mock import patch, MagicMock
from textual.app import App, ComposeResult
from textual.widgets import Static, OptionList
from textual.widgets.option_list import Option


# ---------------------------------------------------------------------------
# Patch heavy imports before loading tui module
# ---------------------------------------------------------------------------
with patch("infinidev.db.service.init_db"), \
     patch("infinidev.engine.loop_engine.LoopEngine"), \
     patch("infinidev.agents.base.InfinidevAgent"):
    from infinidev.cli.tui import ChatInput, InfinidevTUI, COMMANDS


# ---------------------------------------------------------------------------
# Minimal test app that only mounts ChatInput + OptionList
# ---------------------------------------------------------------------------

class ChatInputTestApp(App):
    """Minimal app for testing ChatInput in isolation."""

    def compose(self) -> ComposeResult:
        yield OptionList(id="autocomplete-menu")
        yield ChatInput(id="chat-input")

    def on_mount(self) -> None:
        self.query_one("#chat-input").focus()


# ---------------------------------------------------------------------------
# Full TUI with deps stubbed out
# ---------------------------------------------------------------------------

class FullTUITestApp(InfinidevTUI):
    """InfinidevTUI with heavy deps mocked."""

    def on_mount(self) -> None:
        self.query_one("#chat-input").focus()
        self.add_message("System", "Test mode", "system")
        self.session_id = "test-session"
        self.engine = MagicMock()
        self.agent = MagicMock()


# ============================= SPACE / CHARACTERS ============================

async def test_space_inserts_space_character():
    """Pressing space should insert a space into the text area."""
    app = ChatInputTestApp()
    async with app.run_test() as pilot:
        await pilot.press("h", "e", "l", "l", "o")
        await pilot.press("space")
        await pilot.press("w", "o", "r", "l", "d")
        widget = app.query_one("#chat-input", ChatInput)
        assert widget.text == "hello world"


async def test_multiple_spaces():
    """Multiple consecutive spaces should all be inserted."""
    app = ChatInputTestApp()
    async with app.run_test() as pilot:
        await pilot.press("a", "space", "space", "space", "b")
        widget = app.query_one("#chat-input", ChatInput)
        assert widget.text == "a   b"


async def test_space_at_start():
    """Space at the beginning of input should work."""
    app = ChatInputTestApp()
    async with app.run_test() as pilot:
        await pilot.press("space", "h", "i")
        widget = app.query_one("#chat-input", ChatInput)
        assert widget.text == " hi"


async def test_regular_characters():
    """Letters and digits should be inserted normally."""
    app = ChatInputTestApp()
    async with app.run_test() as pilot:
        await pilot.press("a", "b", "c", "1", "2", "3")
        widget = app.query_one("#chat-input", ChatInput)
        assert widget.text == "abc123"


async def test_space_in_full_tui():
    """Space should work in the full TUI app too."""
    app = FullTUITestApp()
    async with app.run_test() as pilot:
        await pilot.press("h", "e", "l", "l", "o", "space", "w", "o", "r", "l", "d")
        widget = app.query_one("#chat-input", ChatInput)
        assert widget.text == "hello world"


# ================================= ENTER ====================================

async def test_enter_submits_and_clears():
    """Pressing Enter on non-empty input should clear the widget."""
    app = ChatInputTestApp()
    async with app.run_test() as pilot:
        await pilot.press("h", "i")
        await pilot.press("enter")
        widget = app.query_one("#chat-input", ChatInput)
        assert widget.text == ""


async def test_enter_on_empty_does_nothing():
    """Enter on empty input should not change anything."""
    app = ChatInputTestApp()
    async with app.run_test() as pilot:
        widget = app.query_one("#chat-input", ChatInput)
        await pilot.press("enter")
        assert widget.text == ""


async def test_enter_on_whitespace_does_not_submit():
    """Enter on whitespace-only should not clear (strip is empty)."""
    app = ChatInputTestApp()
    async with app.run_test() as pilot:
        await pilot.press("space", "space")
        await pilot.press("enter")
        widget = app.query_one("#chat-input", ChatInput)
        # strip() is empty so it should NOT submit/clear
        assert widget.text == "  "


# ============================== HISTORY (UP/DOWN) ============================

async def test_up_recalls_previous():
    """After submit, Up should recall the last message."""
    app = ChatInputTestApp()
    async with app.run_test() as pilot:
        widget = app.query_one("#chat-input", ChatInput)
        await pilot.press("h", "e", "l", "l", "o")
        await pilot.press("enter")
        assert widget.text == ""

        await pilot.press("up")
        assert widget.text == "hello"


async def test_down_returns_to_draft():
    """After Up, Down should restore the draft."""
    app = ChatInputTestApp()
    async with app.run_test() as pilot:
        widget = app.query_one("#chat-input", ChatInput)

        # Submit two messages
        await pilot.press("a", "enter")
        await pilot.press("b", "enter")

        # Type draft
        await pilot.press("d")

        # Up twice
        await pilot.press("up")
        assert widget.text == "b"
        await pilot.press("up")
        assert widget.text == "a"

        # Down back
        await pilot.press("down")
        assert widget.text == "b"
        await pilot.press("down")
        assert widget.text == "d"


async def test_up_no_history():
    """Up with no history should not change text."""
    app = ChatInputTestApp()
    async with app.run_test() as pilot:
        widget = app.query_one("#chat-input", ChatInput)
        await pilot.press("x")
        await pilot.press("up")
        assert widget.text == "x"


# ============================== AUTOCOMPLETE =================================

async def test_slash_shows_autocomplete():
    """Typing '/' in full TUI should show autocomplete menu."""
    app = FullTUITestApp()
    async with app.run_test() as pilot:
        menu = app.query_one("#autocomplete-menu", OptionList)
        assert not menu.has_class("-visible")

        await pilot.press("slash")
        assert menu.has_class("-visible")
        assert menu.option_count > 0


async def test_slash_h_filters_to_help():
    """Typing '/h' should filter autocomplete to include /help."""
    app = FullTUITestApp()
    async with app.run_test() as pilot:
        menu = app.query_one("#autocomplete-menu", OptionList)
        await pilot.press("slash", "h")

        assert menu.has_class("-visible")
        options = [str(menu.get_option_at_index(i).prompt) for i in range(menu.option_count)]
        assert any("/help" in o for o in options)


async def test_slash_m_filters_to_models():
    """Typing '/m' should filter to /models commands."""
    app = FullTUITestApp()
    async with app.run_test() as pilot:
        menu = app.query_one("#autocomplete-menu", OptionList)
        await pilot.press("slash", "m")

        assert menu.has_class("-visible")
        options = [str(menu.get_option_at_index(i).prompt) for i in range(menu.option_count)]
        assert any("/models" in o for o in options)


async def test_regular_text_hides_autocomplete():
    """Regular text (not /) should hide autocomplete."""
    app = FullTUITestApp()
    async with app.run_test() as pilot:
        menu = app.query_one("#autocomplete-menu", OptionList)

        # Show it first
        await pilot.press("slash")
        assert menu.has_class("-visible")

        # Clear and type regular text
        widget = app.query_one("#chat-input", ChatInput)
        widget.text = ""
        await pilot.press("h", "e", "l", "l", "o")
        assert not menu.has_class("-visible")


async def test_escape_hides_autocomplete():
    """Escape should dismiss autocomplete menu."""
    app = ChatInputTestApp()
    async with app.run_test() as pilot:
        menu = app.query_one("#autocomplete-menu", OptionList)
        menu.add_class("-visible")
        assert menu.has_class("-visible")

        await pilot.press("escape")
        assert not menu.has_class("-visible")


# ======================== FULL TUI COMMANDS ==================================

async def test_help_command_adds_message():
    """Submitting /help should add a system message to chat."""
    app = FullTUITestApp()
    async with app.run_test() as pilot:
        await pilot.press(*list("/help"))
        await pilot.press("enter")

        history = app.query_one("#chat-history")
        messages = history.query(Static)
        # welcome + user "/help" + help response = at least 3
        assert len(messages) >= 3


async def test_submit_adds_user_message():
    """Submitting text should add a user message to chat history."""
    app = FullTUITestApp()
    async with app.run_test() as pilot:
        await pilot.press("h", "i")
        await pilot.press("enter")

        history = app.query_one("#chat-history")
        messages = history.query(Static)
        # welcome + "hi" user msg = at least 2 messages
        assert len(messages) >= 2


# ================ SPACE REGRESSION — THOROUGH COVERAGE ====================
# These tests ensure space works in every context where it could break.

async def test_space_between_words_full_tui():
    """Space between words in the full TUI with Header/Footer/Sidebar."""
    app = FullTUITestApp()
    async with app.run_test() as pilot:
        await pilot.press("f", "i", "x", "space", "b", "u", "g")
        widget = app.query_one("#chat-input", ChatInput)
        assert widget.text == "fix bug"


async def test_space_after_slash_command_filter():
    """Space typed after autocomplete was shown and dismissed."""
    app = FullTUITestApp()
    async with app.run_test() as pilot:
        widget = app.query_one("#chat-input", ChatInput)
        menu = app.query_one("#autocomplete-menu", OptionList)

        # Type slash to trigger autocomplete, then clear
        await pilot.press("slash")
        assert menu.has_class("-visible")

        # Backspace to remove slash, autocomplete should hide
        await pilot.press("backspace")
        assert not menu.has_class("-visible")

        # Now type text with space
        await pilot.press("h", "i", "space", "t", "h", "e", "r", "e")
        assert widget.text == "hi there"


async def test_space_not_doubled():
    """Space must not be inserted twice (MRO dispatch regression)."""
    app = ChatInputTestApp()
    async with app.run_test() as pilot:
        await pilot.press("a", "space", "b")
        widget = app.query_one("#chat-input", ChatInput)
        assert widget.text == "a b", f"Expected 'a b', got {widget.text!r}"


async def test_space_not_doubled_full_tui():
    """Space must not be doubled in the full TUI context."""
    app = FullTUITestApp()
    async with app.run_test() as pilot:
        await pilot.press("x", "space", "y")
        widget = app.query_one("#chat-input", ChatInput)
        assert widget.text == "x y", f"Expected 'x y', got {widget.text!r}"


async def test_letters_not_doubled():
    """Regular letters must not be doubled either (same MRO concern)."""
    app = ChatInputTestApp()
    async with app.run_test() as pilot:
        await pilot.press("a", "b", "c")
        widget = app.query_one("#chat-input", ChatInput)
        assert widget.text == "abc", f"Expected 'abc', got {widget.text!r}"


async def test_space_preserved_after_submit_and_retype():
    """Space works after submitting a message and typing again."""
    app = ChatInputTestApp()
    async with app.run_test() as pilot:
        widget = app.query_one("#chat-input", ChatInput)

        # Submit first message
        await pilot.press("h", "i", "enter")
        assert widget.text == ""

        # Type new message with space
        await pilot.press("n", "e", "w", "space", "m", "s", "g")
        assert widget.text == "new msg"


async def test_space_in_recalled_history_then_edit():
    """Recall a message with space via Up, then add more text."""
    app = ChatInputTestApp()
    async with app.run_test() as pilot:
        widget = app.query_one("#chat-input", ChatInput)

        # Submit message with space
        await pilot.press(*list("hello world"))
        # 'hello world' = h e l l o space w o r l d
        # But pilot.press with list("hello world") sends individual chars
        # including the literal space character ' ' — let's use "space" name
        widget.text = ""
        await pilot.press("h", "e", "l", "l", "o", "space", "w", "o", "r", "l", "d")
        await pilot.press("enter")
        assert widget.text == ""

        # Recall
        await pilot.press("up")
        assert widget.text == "hello world"

        # Add more
        await pilot.press("space", "2")
        assert widget.text.endswith(" 2")


async def test_special_characters_not_swallowed():
    """Punctuation and symbols should be inserted like letters."""
    app = ChatInputTestApp()
    async with app.run_test() as pilot:
        await pilot.press("a", "exclamation_mark", "space", "b", "question_mark")
        widget = app.query_one("#chat-input", ChatInput)
        assert widget.text == "a! b?"


async def test_enter_does_not_insert_newline():
    """Enter must submit, not insert a newline character."""
    app = ChatInputTestApp()
    async with app.run_test() as pilot:
        widget = app.query_one("#chat-input", ChatInput)
        await pilot.press("a")
        await pilot.press("enter")
        # Widget should be cleared (submitted), not contain "a\n"
        assert "\n" not in widget.text
        assert widget.text == ""


async def test_tab_does_not_insert_when_menu_hidden():
    """Tab with hidden autocomplete menu should not insert a tab char."""
    app = ChatInputTestApp()
    async with app.run_test() as pilot:
        menu = app.query_one("#autocomplete-menu", OptionList)
        assert not menu.has_class("-visible")

        await pilot.press("a", "tab", "b")
        widget = app.query_one("#chat-input", ChatInput)
        # Tab should NOT insert a \t — it should be handled by TextArea
        # default behavior (focus next or indent, but not literal \t
        # unless tab_behavior is "indent")
        assert "\t" not in widget.text


async def test_multiple_submits_with_spaces():
    """Submit multiple messages with spaces, all should work."""
    app = ChatInputTestApp()
    async with app.run_test() as pilot:
        widget = app.query_one("#chat-input", ChatInput)
        messages_sent = []

        def on_submit(event):
            messages_sent.append(event.value)

        # Submit 3 messages with spaces
        for msg_keys in [
            ["h", "e", "l", "l", "o", "space", "w", "o", "r", "l", "d"],
            ["f", "o", "o", "space", "b", "a", "r"],
            ["a", "space", "b", "space", "c"],
        ]:
            await pilot.press(*msg_keys)
            # Verify text before submit
            expected = "".join(k if k != "space" else " " for k in msg_keys)
            assert widget.text == expected, f"Before submit: expected {expected!r}, got {widget.text!r}"
            await pilot.press("enter")
            assert widget.text == ""
