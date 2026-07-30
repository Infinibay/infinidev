"""Credentials must never render in full anywhere the user can see them.

A key on screen is a key in scrollback, in screenshots, in recordings and
in whatever the screen-share is showing. These tests cover the three
paths that reach a screen: the settings panel, the `/settings` command,
and any text that reaches the transcript at all.
"""

from __future__ import annotations

import pytest

from infinidev.config.secrets import (
    is_secret,
    mask_if_secret,
    mask_secret,
    redact,
)

ANTHROPIC = "sk-ant-api03-Zx9QhH2LmNbVc7aRt4eYw1PfKj6GdSu8-Wq3Er5Ty"
OPENAI = "sk-proj-aB3dEf5GhJ7kLm9NpQ2rS4tU6vW8xY0z"
SHORT = "abcd1234"


# ── which fields count as secret ──────────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    [
        "LLM_API_KEY",
        "ASSISTANT_LLM_API_KEY",
        "BEHAVIOR_LLM_API_KEY",
        "OPENAI_API_KEY",
        "GITHUB_TOKEN",
        "DB_PASSWORD",
        "CLIENT_SECRET",
        "PRIVATE_KEY_PATH",
    ],
)
def test_credential_names_are_recognised(name):
    assert is_secret(name) is True


@pytest.mark.parametrize(
    "name", ["LLM_MODEL", "LLM_BASE_URL", "LOOP_MAX_ITERATIONS", "KEEP_ALIVE"]
)
def test_ordinary_settings_are_not_masked(name):
    assert is_secret(name) is False
    assert mask_if_secret(name, "visible-value") == "visible-value"


def test_a_new_providers_key_is_masked_without_registering_it():
    """Name-based detection, so nobody has to remember to opt in."""
    assert is_secret("SOME_FUTURE_PROVIDER_API_KEY") is True


# ── the shape that gets shown ─────────────────────────────────────────────


def test_mask_keeps_the_public_prefix_and_a_short_tail():
    masked = mask_secret(ANTHROPIC)
    assert masked.startswith("sk-ant-api")
    assert masked.endswith(ANTHROPIC[-4:])
    assert ANTHROPIC not in masked


def test_masked_output_is_not_usable():
    masked = mask_secret(ANTHROPIC)
    body = masked.replace("sk-ant-api", "").replace("•", "")
    assert len(body) <= 4, "at most a four-character tail may survive"


def test_two_keys_from_different_providers_stay_distinguishable():
    """The point of keeping the prefix: knowing *which* key is set."""
    assert mask_secret(ANTHROPIC) != mask_secret(OPENAI)
    assert mask_secret(OPENAI).startswith("sk-proj-")


def test_short_values_are_masked_whole():
    masked = mask_secret(SHORT)
    assert SHORT not in masked
    assert masked.strip("•") == ""


def test_unset_key_says_so_instead_of_showing_dots():
    assert mask_secret("") == "(not set)"
    assert mask_secret(None) == "(not set)"


def test_local_backend_placeholders_are_left_readable():
    """`ollama` is not a secret; hiding it only confuses."""
    assert mask_secret("ollama") == "ollama"


# ── redaction of arbitrary text ───────────────────────────────────────────


@pytest.fixture
def configured(monkeypatch):
    from infinidev.config import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "LLM_API_KEY", ANTHROPIC)
    return ANTHROPIC


def test_redact_strips_a_key_embedded_in_an_error_message(configured):
    raw = f"401 Unauthorized: https://api.x.com/v1?key={configured}&model=m"
    cleaned = redact(raw)
    assert configured not in cleaned
    assert "401 Unauthorized" in cleaned


def test_redact_leaves_unrelated_text_alone(configured):
    assert redact("nothing secret here") == "nothing secret here"


def test_redact_handles_empty_input(configured):
    assert redact("") == ""


# ── the actual UI surfaces ────────────────────────────────────────────────


def test_settings_panel_renders_the_key_masked(monkeypatch):
    from infinidev.config import settings as settings_mod
    from infinidev.ui.dialogs.settings_control import SettingsControl
    from infinidev.ui.dialogs.settings_editor_state import SettingsEditorState

    monkeypatch.setattr(settings_mod.settings, "LLM_API_KEY", ANTHROPIC)
    state = SettingsEditorState()
    state.section_cursor = 0
    control = SettingsControl(state)

    content = control.create_content(80, 40)
    flat = "".join(
        text
        for i in range(content.line_count)
        for _, text in content.get_line(i)
    )
    if "LLM_API_KEY" not in flat:
        pytest.skip("LLM_API_KEY is not in the default section")
    assert ANTHROPIC not in flat, "the panel must never render the raw key"
    assert "•" in flat


def test_editing_a_key_still_loads_the_real_value(monkeypatch):
    """Masking is a display concern — the edit buffer holds the truth,
    otherwise pressing Enter would save the mask as the new key."""
    from infinidev.config import settings as settings_mod
    from infinidev.ui.dialogs.settings_editor_state import SettingsEditorState

    monkeypatch.setattr(settings_mod.settings, "LLM_API_KEY", ANTHROPIC)
    state = SettingsEditorState()
    for index, (key, _desc, _stype) in enumerate(state.current_settings):
        if key == "LLM_API_KEY":
            state.setting_cursor = index
            break
    else:
        pytest.skip("LLM_API_KEY is not in the default section")

    state.activate()
    assert state.editing is True
    assert state.edit_buffer.text == ANTHROPIC


def test_slash_settings_does_not_echo_the_key(monkeypatch):
    from infinidev.config import settings as settings_mod
    from infinidev.ui.handlers.commands import handle_settings

    monkeypatch.setattr(settings_mod.settings, "LLM_API_KEY", ANTHROPIC)

    class _App:
        def __init__(self):
            self.messages = []

        def add_message(self, sender, text, kind="agent"):
            self.messages.append(text)

        def _update_status_bar(self):
            pass

    app = _App()
    handle_settings(app, ["/settings", "llm_api_key"])
    assert app.messages
    assert ANTHROPIC not in app.messages[-1]
    assert "•" in app.messages[-1]


def test_transcript_redacts_whatever_reaches_it(monkeypatch):
    """Belt and braces: even text nobody audited gets scrubbed."""
    import asyncio

    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from infinidev.config import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "LLM_API_KEY", ANTHROPIC)

    class _Out(DummyOutput):
        def get_size(self):
            return Size(rows=24, columns=80)

    async def _run():
        with create_pipe_input() as pipe, create_app_session(
            input=pipe, output=_Out()
        ):
            from infinidev.ui.app import InfinidevApp

            app = InfinidevApp()
            app.add_message("System", f"Error fetching models: key={ANTHROPIC}")
            return app.chat_messages[-1]["text"]

    text = asyncio.run(_run())
    assert ANTHROPIC not in text
    assert "Error fetching models" in text
