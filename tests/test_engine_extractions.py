"""Characterization tests for the modules extracted from LoopEngine.

``guardrail_runner.apply_guardrail`` and ``UserMessageInjector`` were
lifted verbatim out of ``engine.loop.engine``. These tests pin their
behavior directly (the engine delegates to them) so the extraction is
provably behavior-preserving and stays that way.
"""

from types import SimpleNamespace

from infinidev.engine.loop.guardrail_runner import apply_guardrail
from infinidev.engine.loop.user_message_injector import UserMessageInjector


def _ctx():
    return SimpleNamespace(file_tracker=None, project_id=1, agent_id="agent-1")


class TestUserMessageInjector:
    def test_inject_and_drain_fifo(self):
        inj = UserMessageInjector()
        inj.inject("first")
        inj.inject("second")
        assert inj.drain() == ["first", "second"]
        assert inj.drain() == []  # queue emptied

    def test_inject_mid_step_appends_urgent_user_turns(self):
        inj = UserMessageInjector()
        inj.inject("please stop")
        messages: list[dict] = []
        inj.inject_mid_step(_ctx(), messages)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "please stop" in messages[0]["content"]
        assert "URGENT" in messages[0]["content"]

    def test_inject_mid_step_noop_when_empty(self):
        inj = UserMessageInjector()
        messages: list[dict] = [{"role": "user", "content": "x"}]
        inj.inject_mid_step(_ctx(), messages)
        assert messages == [{"role": "user", "content": "x"}]  # untouched

    def test_reject_returns_false_when_queue_empty(self):
        inj = UserMessageInjector()
        messages: list[dict] = []
        assert inj.reject_step_complete_on_late_message(_ctx(), messages, "sc1") is False
        assert messages == []

    def test_reject_overwrites_existing_step_complete_result(self):
        inj = UserMessageInjector()
        inj.inject("wait, one more thing")
        messages = [
            {"role": "assistant", "content": "done"},
            {"role": "tool", "tool_call_id": "sc1", "content": '{"status": "acknowledged"}'},
        ]
        fired = inj.reject_step_complete_on_late_message(_ctx(), messages, "sc1")
        assert fired is True
        tool_msg = messages[1]
        assert tool_msg["tool_call_id"] == "sc1"
        assert "REJECTED" in tool_msg["content"]
        assert "wait, one more thing" in tool_msg["content"]
        # No duplicate tool result appended (Anthropic one-result-per-id rule).
        assert sum(1 for m in messages if m.get("tool_call_id") == "sc1") == 1

    def test_reject_appends_when_no_prior_result(self):
        inj = UserMessageInjector()
        inj.inject("late")
        messages: list[dict] = [{"role": "assistant", "content": "done"}]
        assert inj.reject_step_complete_on_late_message(_ctx(), messages, "sc9") is True
        assert messages[-1]["role"] == "tool"
        assert messages[-1]["tool_call_id"] == "sc9"
        assert "late" in messages[-1]["content"]

    def test_overwrite_static_rewrites_in_place(self):
        messages = [{"role": "tool", "tool_call_id": "id", "content": "old"}]
        UserMessageInjector._overwrite_step_complete_tool_result(messages, "id", "new")
        assert messages[0]["content"] == "new"
        assert len(messages) == 1


class TestApplyGuardrail:
    def test_none_guardrail_returns_result_unchanged(self):
        out = apply_guardrail(_ctx(), "OUTPUT", None, 3, {}, "sys", "d", "e", None, [], {})
        assert out == "OUTPUT"

    def test_bool_guardrail_true_returns_result(self):
        out = apply_guardrail(_ctx(), "OUTPUT", lambda r: True, 3, {}, "sys", "d", "e", None, [], {})
        assert out == "OUTPUT"

    def test_tuple_guardrail_success_returns_result(self):
        out = apply_guardrail(_ctx(), "OUTPUT", lambda r: (True, r), 3, {}, "sys", "d", "e", None, [], {})
        assert out == "OUTPUT"

    def test_guardrail_exception_is_fail_open_and_logged(self, caplog):
        def boom(_r):
            raise RuntimeError("guardrail blew up")

        with caplog.at_level("ERROR"):
            out = apply_guardrail(_ctx(), "ORIG", boom, 2, {}, "sys", "d", "e", None, [], {})
        assert out == "ORIG"  # unvalidated result shipped (fail-open)
        assert "UNVALIDATED" in caplog.text

    def test_failing_tuple_guardrail_retries_then_returns_original(self):
        calls = {"n": 0}

        def always_fail(_r):
            calls["n"] += 1
            return (False, "nope")

        # max_per_action=0 -> the LLM retry loop body never runs, so the
        # result is unchanged and the guardrail is re-checked each attempt.
        out = apply_guardrail(_ctx(), "ORIG", always_fail, 3, {}, "sys", "d", "e", None, [], {}, max_per_action=0)
        assert out == "ORIG"
        assert calls["n"] == 3  # one check per retry attempt

    def test_failing_tuple_guardrail_reprompts_via_llm(self, monkeypatch):
        import infinidev.engine.loop.guardrail_runner as gr

        fake_resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="CORRECTED", tool_calls=None))]
        )
        monkeypatch.setattr(gr, "_call_llm", lambda *a, **k: fake_resp)

        calls = {"n": 0}

        def guard(r):
            calls["n"] += 1
            return (False, "needs work") if calls["n"] == 1 else (True, r)

        out = apply_guardrail(
            _ctx(), "ORIG", guard, 3, {"model": "x"}, "sys", "d", "e", None, [], {}, max_per_action=2,
        )
        assert out == "CORRECTED"  # picked up the re-prompted answer
        assert calls["n"] == 2
