"""Tests for contextual FOOTER_HINTS filtering and sidebar toggle keybinding."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from infinidev.ui.keybindings import FOOTER_HINTS, get_active_contexts


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_app_state(active_tab: str = "chat", active_dialog=None) -> SimpleNamespace:
    """Create a minimal app_state mock."""
    return SimpleNamespace(active_tab=active_tab, active_dialog=active_dialog)


def _hint_keys_for_context(contexts: frozenset[str]) -> set[str]:
    """Return the shortcut keys visible for a given set of active contexts."""
    return {key for key, _desc, ctx in FOOTER_HINTS if ctx & contexts}


# ── get_active_contexts ───────────────────────────────────────────────────


class TestGetActiveContexts:
    """Tests for get_active_contexts()."""

    def test_default_is_always_only(self):
        """With default app_state (chat tab, no dialog), only 'always' is active."""
        state = _make_app_state()
        result = get_active_contexts(state)
        assert result == frozenset({"always"})

    def test_file_tab_adds_file_context(self):
        """When active_tab is not 'chat', the 'file' context is active."""
        state = _make_app_state(active_tab="src/main.py")
        result = get_active_contexts(state)
        assert "file" in result
        assert "always" in result

    def test_active_dialog_adds_modal_context(self):
        """When active_dialog is set, the 'modal' context is active."""
        state = _make_app_state(active_dialog="search")
        result = get_active_contexts(state)
        assert "modal" in result
        assert "always" in result

    def test_file_and_modal_combined(self):
        """Both file and modal contexts can be active simultaneously."""
        state = _make_app_state(active_tab="README.md", active_dialog="confirm")
        result = get_active_contexts(state)
        assert result == frozenset({"always", "file", "modal"})

    def test_missing_attributes_use_defaults(self):
        """If app_state lacks attributes, defaults are used (chat tab, no dialog)."""
        state = SimpleNamespace()  # no active_tab, no active_dialog
        result = get_active_contexts(state)
        assert result == frozenset({"always"})


# ── FOOTER_HINTS structure ────────────────────────────────────────────────


class TestFooterHintsStructure:
    """Validate FOOTER_HINTS data structure."""

    def test_all_entries_have_three_elements(self):
        for entry in FOOTER_HINTS:
            assert len(entry) == 3, f"Entry {entry!r} must be (key, desc, contexts)"

    def test_contexts_are_frozenset(self):
        for _key, _desc, ctx in FOOTER_HINTS:
            assert isinstance(ctx, frozenset), f"Context must be frozenset, got {type(ctx)}"

    def test_always_context_exists(self):
        """At least one hint must use 'always' so the footer is never empty."""
        always_hints = [k for k, _d, ctx in FOOTER_HINTS if "always" in ctx]
        assert len(always_hints) > 0, "Need at least one 'always' hint"

    def test_sidebar_hint_present(self):
        """Ctrl+. (Sidebar) must be in FOOTER_HINTS."""
        keys = {k for k, _d, _ctx in FOOTER_HINTS}
        assert "Ctrl+." in keys


# ── Context-based filtering ───────────────────────────────────────────────


class TestContextFiltering:
    """Test that hints are correctly filtered based on context."""

    # Keys that should always be visible
    ALWAYS_KEYS = {
        "Ctrl+C", "Ctrl+O", "Ctrl+E", "Ctrl+.", "Ctrl+G", "F2", "Esc",
    }

    # Keys only visible in file context
    FILE_ONLY_KEYS = {"F6", "Ctrl+W", "Ctrl+S", "Ctrl+F"}

    def test_chat_context_shows_only_always_hints(self):
        """In chat mode (no file, no modal), only 'always' hints appear."""
        state = _make_app_state(active_tab="chat")
        contexts = get_active_contexts(state)
        visible = _hint_keys_for_context(contexts)
        assert visible == self.ALWAYS_KEYS

    def test_file_context_shows_file_hints(self):
        """With a file tab open, file-only hints are also visible."""
        state = _make_app_state(active_tab="src/app.py")
        contexts = get_active_contexts(state)
        visible = _hint_keys_for_context(contexts)
        assert self.FILE_ONLY_KEYS.issubset(visible)
        assert self.ALWAYS_KEYS.issubset(visible)

    def test_file_only_hints_hidden_in_chat(self):
        """File-only hints (Save, Close, Find, Line#) are hidden in chat mode."""
        state = _make_app_state(active_tab="chat")
        contexts = get_active_contexts(state)
        visible = _hint_keys_for_context(contexts)
        for key in self.FILE_ONLY_KEYS:
            assert key not in visible, f"{key} should not be visible in chat mode"

    def test_all_hints_visible_in_file_context(self):
        """In file context, all hints (always + file) are visible."""
        state = _make_app_state(active_tab="file.py")
        contexts = get_active_contexts(state)
        visible = _hint_keys_for_context(contexts)
        expected = self.ALWAYS_KEYS | self.FILE_ONLY_KEYS
        assert visible == expected

    def test_modal_does_not_hide_hints(self):
        """Modal context doesn't remove any hints (intersection logic)."""
        state_no_modal = _make_app_state(active_tab="file.py")
        state_modal = _make_app_state(active_tab="file.py", active_dialog="search")
        visible_no_modal = _hint_keys_for_context(get_active_contexts(state_no_modal))
        visible_modal = _hint_keys_for_context(get_active_contexts(state_modal))
        # Modal can only add, never remove
        assert visible_no_modal.issubset(visible_modal)


# ── FooterControl integration ─────────────────────────────────────────────


class TestFooterControl:
    """Tests for FooterControl text generation."""

    def test_no_app_state_shows_all_hints(self):
        """Without app_state, all hints are shown (fallback)."""
        from infinidev.ui.controls.footer_control import FooterControl

        ctrl = FooterControl(app_state=None)
        text = ctrl._get_text()
        # Every hint key should appear in the formatted text
        text_str = "".join(frag[1] for frag in text)
        for key, _desc, _ctx in FOOTER_HINTS:
            assert key in text_str, f"{key} missing from fallback text"

    def test_chat_mode_hides_file_hints(self):
        """In chat mode, file-only hints are not rendered."""
        from infinidev.ui.controls.footer_control import FooterControl

        state = _make_app_state(active_tab="chat")
        ctrl = FooterControl(app_state=state)
        text = ctrl._get_text()
        text_str = "".join(frag[1] for frag in text)
        # File-only keys should NOT appear
        for key in ("F6", "Ctrl+W", "Ctrl+S", "Ctrl+F"):
            assert key not in text_str, f"{key} should be hidden in chat mode"

    def test_file_mode_shows_all_hints(self):
        """With a file tab open, all hints appear."""
        from infinidev.ui.controls.footer_control import FooterControl

        state = _make_app_state(active_tab="src/main.py")
        ctrl = FooterControl(app_state=state)
        text = ctrl._get_text()
        text_str = "".join(frag[1] for frag in text)
        for key, _desc, _ctx in FOOTER_HINTS:
            assert key in text_str, f"{key} missing in file mode"

    def test_context_change_updates_display(self):
        """Switching from chat to file tab updates the displayed hints."""
        from infinidev.ui.controls.footer_control import FooterControl

        state = _make_app_state(active_tab="chat")
        ctrl = FooterControl(app_state=state)

        # Chat mode: no file hints
        text_chat = ctrl._get_text()
        chat_str = "".join(frag[1] for frag in text_chat)
        assert "Ctrl+S" not in chat_str

        # Switch to file tab
        state.active_tab = "file.py"
        text_file = ctrl._get_text()
        file_str = "".join(frag[1] for frag in text_file)
        assert "Ctrl+S" in file_str
