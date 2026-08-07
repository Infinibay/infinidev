"""The gate that makes it safe to let the model rewrite an approved plan.

Loosening ``user_approved`` from an operation-gate to a field-gate hands the
model the wording of a step the user asked for. The counterweight is here: it
can say what a step *means*, it cannot decide the step no longer needs doing,
and ending the run is where that difference becomes observable.

Two properties are load-bearing and easy to get wrong:

* **The closing step is still ``active``.** ``blocks()`` runs while the model's
  ``step_complete`` is being adjudicated; ``advance_plan`` does not mark the
  step done until afterwards. A gate that counted it would refuse every
  correct close of the last step, which is every successful run.
* **``blocked`` discharges a step.** A step closed as blocked was attempted and
  reported. Treating it as undone would punish the honest outcome and reward
  the model for claiming success instead.

The gate replaces ``StepManager.auto_split``, which only fired when
``final_answer`` was empty — so ``step_complete(status="done",
final_answer="…")`` walked past every pending step in a single call.
"""

from __future__ import annotations

from types import SimpleNamespace

from infinidev.engine.loop.engine import LoopEngine
from infinidev.engine.loop.models import LoopState
from infinidev.engine.loop.plan_step import PlanStep
from infinidev.engine.loop.step_complete_gate import _SCOPE_GATE_MAX_ATTEMPTS
from infinidev.engine.loop.step_manager import StepManager
from infinidev.engine.loop.step_result import StepResult


def _plan_state(*steps: PlanStep) -> LoopState:
    state = LoopState()
    state.plan.steps = list(steps)
    return state


def _ctx(state: LoopState) -> SimpleNamespace:
    return SimpleNamespace(state=state, project_id=1, agent_id="t", workspace_path=".")


def _call(status: str = "done", final_answer: str = "all set") -> SimpleNamespace:
    return SimpleNamespace(
        id="sc1",
        function=SimpleNamespace(
            arguments=(
                '{"summary":"x","status":"%s","final_answer":"%s"}'
                % (status, final_answer)
            ),
        ),
    )


def _messages() -> list[dict]:
    return [{"role": "tool", "tool_call_id": "sc1", "content": "ok"}]


def _approved_run() -> LoopState:
    """Step 1 closing, steps 2 and 3 never started."""
    return _plan_state(
        PlanStep(index=1, title="Read auth.py", user_approved=True, status="active"),
        PlanStep(index=2, title="Add the exp check", user_approved=True),
        PlanStep(index=3, title="Run the auth tests", user_approved=True),
    )


class TestItRefusesToAbandonScope:
    def test_done_with_approved_steps_pending_is_blocked(self):
        eng = LoopEngine()
        ctx, messages = _ctx(_approved_run()), _messages()
        assert eng._step_gate._scope_open(ctx, _call(), messages) is True

    def test_a_final_answer_does_not_buy_a_pass(self):
        """The hole auto_split left: it only fired on an empty final_answer."""
        eng = LoopEngine()
        ctx, messages = _ctx(_approved_run()), _messages()
        call = _call(final_answer="I fixed the authentication bug.")
        assert eng._step_gate._scope_open(ctx, call, messages) is True

    def test_the_refusal_names_the_steps_and_reaches_the_model(self):
        eng = LoopEngine()
        ctx, messages = _ctx(_approved_run()), _messages()
        eng._step_gate._scope_open(ctx, _call(), messages)
        tool_msg = next(m for m in messages if m.get("tool_call_id") == "sc1")
        assert "BLOCKED" in tool_msg["content"]
        assert "Add the exp check" in tool_msg["content"]
        assert "Run the auth tests" in tool_msg["content"]
        assert "attempt 1/" in tool_msg["content"]

    def test_developer_steps_must_be_explicitly_discharged(self):
        """A rolling horizon cannot be abandoned by declaring Task done."""
        eng = LoopEngine()
        state = _plan_state(
            PlanStep(index=1, title="Explore", status="active"),
            PlanStep(index=2, title="Write the fix"),
        )
        ctx, messages = _ctx(state), _messages()
        assert eng._step_gate._scope_open(ctx, _call(), messages) is True



class TestItLetsCorrectRunsClose:
    def test_the_step_being_closed_is_not_counted(self):
        """It is still 'active' at gate time — advance_plan has not run yet."""
        eng = LoopEngine()
        state = _plan_state(
            PlanStep(index=1, title="Read", user_approved=True, status="done"),
            PlanStep(index=2, title="Fix", user_approved=True, status="done"),
            PlanStep(index=3, title="Test", user_approved=True, status="active"),
        )
        ctx, messages = _ctx(state), _messages()
        assert eng._step_gate._scope_open(ctx, _call(), messages) is False

    def test_a_blocked_step_discharges_the_plan(self):
        eng = LoopEngine()
        state = _plan_state(
            PlanStep(index=1, title="Read", user_approved=True, status="done"),
            PlanStep(index=2, title="Fix", user_approved=True, status="blocked"),
            PlanStep(index=3, title="Test", user_approved=True, status="active"),
        )
        ctx, messages = _ctx(state), _messages()
        assert eng._step_gate._scope_open(ctx, _call(), messages) is False

    def test_blocked_is_recorded_and_the_next_step_activates(self):
        state = _plan_state(
            PlanStep(index=1, title="Could not edit", status="active"),
            PlanStep(index=2, title="Report the limitation"),
        )
        manager = StepManager(SimpleNamespace(_hooks=None))

        manager.advance_plan(
            SimpleNamespace(state=state),
            StepResult(summary="Permission denied", status="blocked"),
        )

        assert state.plan.steps[0].status == "blocked"
        assert state.plan.steps[1].status == "active"

    def test_continue_is_never_gated(self):
        eng = LoopEngine()
        ctx, messages = _ctx(_approved_run()), _messages()
        assert eng._step_gate._scope_open(ctx, _call(status="continue"), messages) is False

    def test_blocked_close_is_never_gated(self):
        """Giving up loudly is the outcome this gate exists to make available."""
        eng = LoopEngine()
        ctx, messages = _ctx(_approved_run()), _messages()
        assert eng._step_gate._scope_open(ctx, _call(status="blocked"), messages) is False

    def test_an_empty_plan_is_not_gated(self):
        eng = LoopEngine()
        ctx, messages = _ctx(_plan_state()), _messages()
        assert eng._step_gate._scope_open(ctx, _call(), messages) is False

    def test_unapproved_steps_consume_the_task_completion_budget(self):
        eng = LoopEngine()
        state = _plan_state(
            PlanStep(index=1, title="Explore", status="active"),
            PlanStep(index=2, title="Optional follow-up"),
        )
        ctx, messages = _ctx(state), _messages()
        assert eng._step_gate._scope_open(ctx, _call(), messages) is True
        assert eng._step_gate._scope_attempts == 1


class TestItGivesUpRatherThanSpin:
    def test_the_close_is_honoured_after_the_cap(self):
        eng = LoopEngine()
        ctx, messages = _ctx(_approved_run()), _messages()
        for _ in range(_SCOPE_GATE_MAX_ATTEMPTS):
            assert eng._step_gate._scope_open(ctx, _call(), messages) is True
        assert eng._step_gate._scope_open(ctx, _call(), messages) is False

    def test_the_drop_is_recorded_where_the_next_turn_reads_it(self):
        eng = LoopEngine()
        state = _approved_run()
        ctx, messages = _ctx(state), _messages()
        for _ in range(_SCOPE_GATE_MAX_ATTEMPTS + 1):
            eng._step_gate._scope_open(ctx, _call(), messages)
        assert any("not executed" in n for n in state.notes)
        assert any("Add the exp check" in n for n in state.notes)

    def test_the_budget_is_per_run_not_per_step(self):
        """Otherwise every step is a fresh pair of attempts at ending early."""
        eng = LoopEngine()
        state = _approved_run()
        ctx, messages = _ctx(state), _messages()
        assert eng._step_gate._scope_open(ctx, _call(), messages) is True

        # The model moves on, closes step 1, and tries again from step 2.
        state.plan.steps[0].status = "done"
        state.plan.steps[1].status = "active"
        assert eng._step_gate._scope_open(ctx, _call(), messages) is True
        assert eng._step_gate._scope_open(ctx, _call(), messages) is False

    def test_reset_run_clears_the_budget(self):
        eng = LoopEngine()
        ctx, messages = _ctx(_approved_run()), _messages()
        for _ in range(_SCOPE_GATE_MAX_ATTEMPTS + 1):
            eng._step_gate._scope_open(ctx, _call(), messages)
        eng._step_gate.reset_run()
        assert eng._step_gate._scope_open(ctx, _call(), messages) is True


class TestItRunsBeforeTheExpensiveGates:
    def test_scope_is_the_first_link_in_the_chain(self):
        """A scope-dropping close must not pay for a critic call to be refused."""
        eng = LoopEngine()
        ctx, messages = _ctx(_approved_run()), _messages()
        called: list[str] = []
        eng._critic = SimpleNamespace(
            review_step_complete=lambda *a, **k: called.append("critic")
            or SimpleNamespace(blocked=False),
        )
        assert eng._step_gate.blocks(
            ctx, _call(), messages, action_tool_calls=3, reasoning=None,
        ) is True
        assert called == [], "the critic was consulted before scope was checked"
