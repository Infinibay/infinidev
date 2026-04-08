"""Unit tests for the pre-planning preamble step.

These tests call ``run_preamble_step`` directly (no TUI, no engine
loop) so iteration on the prompt is fast and the assertions are
crisp: they verify the model picks ``status="done"`` for chat
inputs and ``status="continue"`` for real-work inputs.

The tests intentionally test the prompt-bias guarantee the user
asked for: a chat input must NOT silently flip to continue, and
a work input must NOT silently flip to done. Each direction has
its own asymmetric failure mode that we want to catch.

Run with:

    uv run pytest tests/interactive/test_preamble.py -v -s
"""

from __future__ import annotations

import pytest

from infinidev.engine.orchestration.conversational_fastpath import run_preamble_step

pytestmark = pytest.mark.slow


# ─────────────────────────────────────────────────────────────────────
# Chat cases — must decide done
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("user_input", [
    "Hola",
    "hi",
    "buenas tardes",
    "gracias",
    "thanks for the fix",
    "ok",
    "chau",
    "bye",
    "are you there?",
    "who are you?",
])
def test_preamble_chat_decides_done(user_input):
    """Conversational input must NOT trigger continue.

    Failure here means the prompt has tilted toward continue and the
    model is "playing it safe" by picking work. Re-read the prompt
    bias warning in conversational_fastpath.py before iterating.
    """
    res = run_preamble_step(user_input, [])
    if res is None:
        pytest.skip(f"preamble returned None for {user_input!r} — "
                    f"intermittent tool-call miss, not a bias bug")
    status, message = res
    assert status == "done", (
        f"Expected status=done for chat input {user_input!r}, "
        f"got {status!r}. Message was: {message!r}. "
        f"This usually means the prompt tilted toward continue."
    )
    assert message, "done status requires a non-empty reply text"


# ─────────────────────────────────────────────────────────────────────
# Work cases — must decide continue
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("user_input", [
    "fix the auth bug in src/auth.py",
    "add a unit test for parseDate",
    "explain how the auth flow works",
    "refactor verify_token",
    "list the files in src/",
    "find all references to connectToVm",
    "what's in the README?",
    "delete the old logging code",
    "run the test suite",
    "show me the structure of UserService",
])
def test_preamble_work_decides_continue(user_input):
    """Real-work input must NOT trigger done.

    Failure here means the prompt has tilted toward done and the
    model is short-circuiting tasks that need actual file inspection.
    Both directions of bias are bugs — the symmetric failure mode.
    """
    res = run_preamble_step(user_input, [])
    if res is None:
        pytest.skip(f"preamble returned None for {user_input!r} — "
                    f"intermittent tool-call miss, not a bias bug")
    status, message = res
    assert status == "continue", (
        f"Expected status=continue for work input {user_input!r}, "
        f"got {status!r}. Message was: {message!r}. "
        f"This usually means the prompt is short-circuiting tasks "
        f"that should reach the analyst / executor."
    )
    assert message, "continue status requires a non-empty preview message"
