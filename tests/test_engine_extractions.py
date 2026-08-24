"""Characterization tests for the modules extracted from LoopEngine.

``guardrail_runner.apply_guardrail`` and ``UserMessageInjector`` were
lifted verbatim out of ``engine.loop.engine``. These tests pin their
behavior directly (the engine delegates to them) so the extraction is
provably behavior-preserving and stays that way.
"""

from types import SimpleNamespace

from infinidev.engine.loop.guardrail_runner import apply_guardrail
from infinidev.engine.loop.loop_state import LoopState
from infinidev.engine.loop.user_message_injector import UserMessageInjector


def _ctx():
    return SimpleNamespace(file_tracker=None, project_id=1, agent_id="agent-1")


class TestUserMessageInjector:
    def test_notifies_context_change_for_mid_step_and_late_messages(self):
        notifications: list[list[str]] = []
        inj = UserMessageInjector(notifications.append)
        inj.inject("change target")
        inj.inject_mid_step(_ctx(), [])
        inj.inject("also change tests")
        messages = [
            {"role": "assistant", "tool_calls": [{}]},
            {"role": "tool", "tool_call_id": "sc1", "content": "acknowledged"},
        ]
        inj.reject_step_complete_on_late_message(_ctx(), messages, "sc1")

        assert notifications == [["change target"], ["also change tests"]]

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
        drained = inj.inject_mid_step(_ctx(), messages)
        assert drained == ["please stop"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "please stop" in messages[0]["content"]
        assert "URGENT" in messages[0]["content"]

    def test_inject_mid_step_noop_when_empty(self):
        inj = UserMessageInjector()
        messages: list[dict] = [{"role": "user", "content": "x"}]
        assert inj.inject_mid_step(_ctx(), messages) == []
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
        """FC mode: the assistant announced the call, so answer on that channel.

        The assistant turn carries ``tool_calls`` because that is what
        ``ToolRunner.append_assistant_message`` builds in FC mode — and it
        is what makes appending a ``role: "tool"`` message valid here.
        """
        inj = UserMessageInjector()
        inj.inject("late")
        messages: list[dict] = [{
            "role": "assistant", "content": "done",
            "tool_calls": [{
                "id": "sc9", "type": "function",
                "function": {"name": "step_complete", "arguments": "{}"},
            }],
        }]
        assert inj.reject_step_complete_on_late_message(_ctx(), messages, "sc9") is True
        assert messages[-1]["role"] == "tool"
        assert messages[-1]["tool_call_id"] == "sc9"
        assert "late" in messages[-1]["content"]

    def test_reject_speaks_prose_when_the_transcript_has_no_tool_channel(self):
        """Manual mode: the same assistant turn, but as prose.

        There is no tool call to answer, so a ``role: "tool"`` message
        here answers something nobody asked and invalidates the request.
        """
        inj = UserMessageInjector()
        inj.inject("late")
        messages: list[dict] = [{"role": "assistant", "content": "done"}]
        assert inj.reject_step_complete_on_late_message(_ctx(), messages, "sc9") is True
        assert messages[-1]["role"] == "user"
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

    def test_unlimited_tool_budget_still_reprompts_via_llm(self, monkeypatch):
        import infinidev.engine.loop.guardrail_runner as gr

        fake_resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="CORRECTED",
                        tool_calls=None,
                    )
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=3,
                total_tokens=14,
            ),
        )
        llm_calls = {"n": 0}

        def call_llm(*args, **kwargs):
            llm_calls["n"] += 1
            return fake_resp

        monkeypatch.setattr(gr, "_call_llm", call_llm)
        guard_calls = {"n": 0}

        def guard(result):
            guard_calls["n"] += 1
            return (
                (False, "needs work")
                if guard_calls["n"] == 1
                else (True, result)
            )

        ctx = _ctx()
        ctx.state = LoopState()
        out = apply_guardrail(
            ctx,
            "ORIG",
            guard,
            3,
            {"model": "x"},
            "sys",
            "d",
            "e",
            ctx.state,
            [],
            {},
            max_per_action=0,
        )

        assert out == "CORRECTED"
        assert llm_calls["n"] == 1
        assert guard_calls["n"] == 2
        assert ctx.state.total_prompt_tokens == 11
        assert ctx.state.total_completion_tokens == 3
        assert ctx.state.total_tokens == 14

    def test_last_corrected_result_is_revalidated(self, monkeypatch):
        import infinidev.engine.loop.guardrail_runner as gr

        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="CORRECTED",
                        tool_calls=None,
                    )
                )
            ]
        )
        llm_calls = {"n": 0}

        def call_llm(*args, **kwargs):
            llm_calls["n"] += 1
            return response

        monkeypatch.setattr(gr, "_call_llm", call_llm)
        validated: list[str] = []

        def guard(result):
            validated.append(result)
            return False, "still invalid"

        out = apply_guardrail(
            _ctx(),
            "ORIG",
            guard,
            1,
            {"model": "x"},
            "sys",
            "d",
            "e",
            None,
            [],
            {},
            max_per_action=0,
        )

        assert out == "CORRECTED"
        assert llm_calls["n"] == 1
        assert validated == ["ORIG", "CORRECTED"]

    def test_guardrail_tool_calls_count_toward_loop_budget(self, monkeypatch):
        import infinidev.engine.loop.guardrail_runner as gr

        tool_call = SimpleNamespace(
            id="tc1",
            function=SimpleNamespace(name="read_file", arguments="{}"),
        )
        responses = iter([
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="",
                            tool_calls=[tool_call],
                        )
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="CORRECTED",
                            tool_calls=None,
                        )
                    )
                ]
            ),
        ])
        monkeypatch.setattr(gr, "_call_llm", lambda *args, **kwargs: next(responses))
        monkeypatch.setattr(gr, "_capture_pre_content", lambda *args, **kwargs: None)
        monkeypatch.setattr(gr, "_maybe_emit_file_change", lambda *args, **kwargs: None)
        executed: list[str] = []

        def execute(_dispatch, name, _arguments):
            executed.append(name)
            return "ok"

        monkeypatch.setattr(gr, "execute_tool_call", execute)
        state = SimpleNamespace(total_tool_calls=4)
        guard_calls = {"n": 0}

        def guard(result):
            guard_calls["n"] += 1
            return (
                (False, "inspect first")
                if guard_calls["n"] == 1
                else (True, result)
            )

        out = apply_guardrail(
            _ctx(),
            "ORIG",
            guard,
            3,
            {"model": "x"},
            "sys",
            "d",
            "e",
            state,
            [{"type": "function"}],
            {"read_file": object()},
            max_per_action=2,
        )

        assert out == "CORRECTED"
        assert executed == ["read_file"]
        assert state.total_tool_calls == 5

    def test_guardrail_respects_remaining_global_tool_budget(self, monkeypatch):
        import infinidev.engine.loop.guardrail_runner as gr

        tool_call = SimpleNamespace(
            id="tc1",
            function=SimpleNamespace(name="read_file", arguments="{}"),
        )
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[tool_call],
                    )
                )
            ]
        )
        monkeypatch.setattr(gr, "_call_llm", lambda *args, **kwargs: response)
        monkeypatch.setattr(gr, "_capture_pre_content", lambda *args, **kwargs: None)
        monkeypatch.setattr(gr, "_maybe_emit_file_change", lambda *args, **kwargs: None)
        executed: list[str] = []
        monkeypatch.setattr(
            gr,
            "execute_tool_call",
            lambda _dispatch, name, _arguments: executed.append(name) or "ok",
        )
        ctx = _ctx()
        ctx.max_total_calls = 4
        state = SimpleNamespace(total_tool_calls=4)

        out = apply_guardrail(
            ctx,
            "ORIG",
            lambda _result: (False, "inspect first"),
            1,
            {"model": "x"},
            "sys",
            "d",
            "e",
            state,
            [{"type": "function"}],
            {"read_file": object()},
            max_per_action=2,
        )

        assert out == "ORIG"
        assert executed == []
        assert state.total_tool_calls == 4

    def test_guardrail_does_not_cross_exhausted_prompt_budget(self, monkeypatch):
        import infinidev.engine.loop.guardrail_runner as gr

        llm_calls = {"n": 0}
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="CORRECTED",
                        tool_calls=None,
                    )
                )
            ]
        )

        def call_llm(*args, **kwargs):
            llm_calls["n"] += 1
            return response

        monkeypatch.setattr(gr, "_call_llm", call_llm)
        ctx = _ctx()
        ctx.state = LoopState(total_prompt_tokens=100)
        ctx.max_prompt_tokens = 100

        out = apply_guardrail(
            ctx,
            "ORIG",
            lambda _result: (False, "needs work"),
            3,
            {"model": "x"},
            "sys",
            "d",
            "e",
            ctx.state,
            [],
            {},
            max_per_action=0,
        )

        assert out == "ORIG"
        assert llm_calls["n"] == 0
        assert ctx.state.total_prompt_tokens == 100

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
