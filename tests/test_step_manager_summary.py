"""The structured step summariser must actually run.

``summarize_and_record`` calls ``_summarize_step`` inside a broad
``except Exception``. That swallowed a plain ``NameError`` — the symbol
was never imported into ``step_manager`` — so every step silently fell
back to the bare summary and ``changes_made`` / ``discovered_context`` /
``files_to_preload`` / ``anti_patterns`` were always empty in production.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from infinidev.engine.loop import step_manager
from infinidev.engine.loop.models import LoopState, StepResult


def test_summarize_step_is_resolvable_from_step_manager():
    """A regression net for the import itself."""
    assert callable(step_manager._summarize_step)


class _Engine:
    _summarizer_override = True
    _last_state = None
    _hooks = None


@pytest.fixture
def ctx(tmp_path):
    return SimpleNamespace(
        state=LoopState(),
        desc="do the thing",
        llm_params={},
        is_small=False,
        agent=SimpleNamespace(project_id=1, agent_id="agent-1"),
        project_id=1,
        agent_id="agent-1",
        session_id="session-1",
    )


def test_structured_fields_reach_the_action_record(monkeypatch, ctx, tmp_path):
    from infinidev.code_intel import _db as ci_db
    from infinidev.config import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "DB_PATH", str(tmp_path / "s.db"))
    ci_db._conn_cache.__dict__.clear()

    captured = {}

    def _fake_summarize(messages, desc, state, step_result, llm_params):
        captured["called"] = True
        return {
            "summary": "read the client and patched the retry",
            "files_to_preload": [],
            "changes_made": "src/http/client.py: added backoff",
            "discovered": "retries were unbounded",
            "pending": "add a test",
            "anti_patterns": "",
        }

    monkeypatch.setattr(step_manager, "_summarize_step", _fake_summarize)

    manager = step_manager.StepManager(_Engine())
    manager.summarize_and_record(
        ctx, StepResult(summary="raw", status="continue"), [], 2, 0
    )

    assert captured.get("called"), "the summariser must be invoked, not skipped"
    record = ctx.state.history[-1]
    assert record.summary == "read the client and patched the retry"
    assert record.changes_made == "src/http/client.py: added backoff"
    assert record.discovered_context == "retries were unbounded"
    assert record.pending_items == "add a test"


def test_step_close_archives_context_into_working_memory(monkeypatch, ctx, tmp_path):
    from infinidev.code_intel import _db as ci_db
    from infinidev.config import settings as settings_mod
    from infinidev.engine.working_memory import (
        get_working_memory,
        reset_working_memory,
    )

    monkeypatch.setattr(settings_mod.settings, "DB_PATH", str(tmp_path / "wm.db"))
    monkeypatch.setattr(settings_mod.settings, "LOOP_SUMMARIZER_ENABLED", False)
    ci_db._conn_cache.__dict__.clear()
    reset_working_memory()

    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"file_path": "src/http/client.py"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": "class HttpClient:\n    def request(self, url):\n        ...\n"
            "        # no retry handling anywhere in this method\n",
        },
    ]

    engine = _Engine()
    engine._summarizer_override = False
    manager = step_manager.StepManager(engine)
    manager.summarize_and_record(
        ctx, StepResult(summary="looked at the client", status="continue"), messages, 1, 0
    )

    memory = get_working_memory("session-1")
    hits = memory.search("http client retry handling", limit=3)
    assert hits, "closing a step must archive its raw output"
    assert "HttpClient" in hits[0].content
