"""Tests for user-configured lifecycle hooks (engine/user_hooks/).

Covers the three things that make the feature safe to ship enabled: the
config file is parsed forgivingly but never trusted, a misbehaving hook
costs its output and nothing else, and the two end-of-step hooks land on
opposite sides of the summarisation boundary — which is the entire point
of there being two.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from infinidev.config.settings import settings
from infinidev.engine.loop.action_record import ActionRecord
from infinidev.engine.loop.context import build_iteration_prompt
from infinidev.engine.loop.loop_state import LoopState
from infinidev.engine.loop.step_complete_gate import StepCompleteGate
from infinidev.engine.user_hooks import (
    UserHookEvent,
    get_hooks,
    invalidate_cache,
    load_hooks_config,
    run_hooks,
)


@pytest.fixture
def hooks_home(tmp_path, monkeypatch):
    """Isolate both config locations and the hook cache.

    ``get_base_dir()`` is cwd-relative and the user-level file is
    ``~/.infinidev/hooks.json``, so a test that did not move both would
    read the developer's own hooks and fail on their machine only.
    """
    workspace = tmp_path / "workspace"
    home = tmp_path / "home"
    (workspace / ".infinidev").mkdir(parents=True)
    (home / ".infinidev").mkdir(parents=True)
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(settings, "HOOKS_ENABLED", True)
    invalidate_cache()
    yield SimpleNamespace(workspace=workspace, home=home)
    invalidate_cache()


def _write(directory, payload: dict) -> None:
    path = directory / ".infinidev" / "hooks.json"
    path.write_text(json.dumps(payload), encoding="utf-8")


# ── config parsing ───────────────────────────────────────────────────────────


class TestConfigParsing:
    """hooks.json is hand-edited, so it is read forgivingly."""

    def test_no_config_means_no_hooks(self, hooks_home):
        """The feature is inert until someone writes the file."""
        assert load_hooks_config() == {}
        assert get_hooks(UserHookEvent.TASK_START) == []

    def test_literal_prompt_hook(self, hooks_home):
        """A `prompt` entry needs no subprocess."""
        _write(hooks_home.workspace, {
            "hooks": {"step_end_instruction": [{"prompt": "review it"}]},
        })
        specs = get_hooks(UserHookEvent.STEP_END_INSTRUCTION)
        assert len(specs) == 1
        assert specs[0].is_literal
        assert specs[0].prompt == "review it"

    def test_bare_string_is_a_prompt(self, hooks_home):
        """`["do the thing"]` is accepted as shorthand for a prompt hook."""
        _write(hooks_home.workspace, {"hooks": {"task_start": ["context here"]}})
        specs = get_hooks(UserHookEvent.TASK_START)
        assert [s.prompt for s in specs] == ["context here"]

    def test_single_object_is_wrapped_into_a_list(self, hooks_home):
        """Writing one hook without a list is a papercut, not an error."""
        _write(hooks_home.workspace, {"hooks": {"task_start": {"prompt": "x"}}})
        assert len(get_hooks(UserHookEvent.TASK_START)) == 1

    def test_top_level_hooks_key_is_optional(self, hooks_home):
        """A file whose keys are already event names works as-is."""
        _write(hooks_home.workspace, {"step_start": [{"prompt": "go"}]})
        assert len(get_hooks(UserHookEvent.STEP_START)) == 1

    def test_unknown_event_is_ignored_not_fatal(self, hooks_home):
        """A typo'd event name costs that entry, not the whole config."""
        _write(hooks_home.workspace, {
            "hooks": {
                "step_endd_summary": [{"prompt": "typo"}],
                "step_end_summary": [{"prompt": "real"}],
            },
        })
        assert [s.prompt for s in get_hooks(UserHookEvent.STEP_END_SUMMARY)] == ["real"]

    def test_entry_with_neither_command_nor_prompt_is_dropped(self, hooks_home):
        """A hook that declares nothing to run is not a hook."""
        _write(hooks_home.workspace, {"hooks": {"task_start": [{"timeout": 5}]}})
        assert get_hooks(UserHookEvent.TASK_START) == []

    def test_disabled_entry_is_dropped(self, hooks_home):
        """`enabled: false` keeps a hook in the file but out of the run."""
        _write(hooks_home.workspace, {
            "hooks": {"task_start": [{"prompt": "x", "enabled": False}]},
        })
        assert get_hooks(UserHookEvent.TASK_START) == []

    def test_malformed_json_costs_only_the_hooks(self, hooks_home):
        """A broken file yields no hooks rather than an exception."""
        path = hooks_home.workspace / ".infinidev" / "hooks.json"
        path.write_text("{ not json", encoding="utf-8")
        assert load_hooks_config() == {}

    def test_settings_switch_turns_everything_off(self, hooks_home, monkeypatch):
        """HOOKS_ENABLED=False disables hooks a user already wrote."""
        _write(hooks_home.workspace, {"hooks": {"task_start": [{"prompt": "x"}]}})
        assert get_hooks(UserHookEvent.TASK_START)
        monkeypatch.setattr(settings, "HOOKS_ENABLED", False)
        assert get_hooks(UserHookEvent.TASK_START) == []


class TestConfigMerge:
    """Workspace and user-level files merge per event, not per entry."""

    def test_user_level_applies_when_workspace_is_silent(self, hooks_home):
        """A global hook fires in a project that never mentions the event."""
        _write(hooks_home.home, {"hooks": {"task_start": [{"prompt": "global"}]}})
        _write(hooks_home.workspace, {"hooks": {"step_start": [{"prompt": "local"}]}})
        assert [s.prompt for s in get_hooks(UserHookEvent.TASK_START)] == ["global"]
        assert [s.prompt for s in get_hooks(UserHookEvent.STEP_START)] == ["local"]

    def test_workspace_owns_an_event_it_declares(self, hooks_home):
        """Declaring an event locally replaces the global hooks for it."""
        _write(hooks_home.home, {"hooks": {"task_start": [{"prompt": "global"}]}})
        _write(hooks_home.workspace, {"hooks": {"task_start": [{"prompt": "local"}]}})
        assert [s.prompt for s in get_hooks(UserHookEvent.TASK_START)] == ["local"]

    def test_empty_list_switches_a_global_hook_off(self, hooks_home):
        """`"task_start": []` is how a project opts out of a global hook."""
        _write(hooks_home.home, {"hooks": {"task_start": [{"prompt": "global"}]}})
        _write(hooks_home.workspace, {"hooks": {"task_start": []}})
        assert get_hooks(UserHookEvent.TASK_START) == []


class TestConfigReload:
    """An edited hooks.json takes effect without a restart."""

    def test_edit_is_picked_up(self, hooks_home):
        """The cache is keyed on mtime/size, not on process lifetime."""
        _write(hooks_home.workspace, {"hooks": {"task_start": [{"prompt": "first"}]}})
        assert [s.prompt for s in get_hooks(UserHookEvent.TASK_START)] == ["first"]
        _write(hooks_home.workspace, {
            "hooks": {"task_start": [{"prompt": "second"}, {"prompt": "third"}]},
        })
        assert [s.prompt for s in get_hooks(UserHookEvent.TASK_START)] == [
            "second", "third",
        ]


# ── running hooks ────────────────────────────────────────────────────────────


class TestHookExecution:
    """What a hook prints, and what happens when it misbehaves."""

    def test_literal_hook_returns_its_text(self, hooks_home):
        _write(hooks_home.workspace, {"hooks": {"task_start": [{"prompt": "hello"}]}})
        assert run_hooks(UserHookEvent.TASK_START).text == "hello"

    def test_command_stdout_is_the_output(self, hooks_home):
        _write(hooks_home.workspace, {
            "hooks": {"task_start": [{"command": "echo from-command"}]},
        })
        assert run_hooks(UserHookEvent.TASK_START).text == "from-command"

    def test_outputs_are_concatenated_in_declaration_order(self, hooks_home):
        _write(hooks_home.workspace, {
            "hooks": {"task_start": [{"prompt": "one"}, {"command": "echo two"}]},
        })
        assert run_hooks(UserHookEvent.TASK_START).text == "one\n\ntwo"

    def test_payload_arrives_on_stdin_as_json(self, hooks_home):
        """The full structure is available to hooks that want it."""
        _write(hooks_home.workspace, {"hooks": {"task_start": [{"command": "cat"}]}})
        output = run_hooks(UserHookEvent.TASK_START, {"session_id": "s-42"})
        assert json.loads(output.text)["session_id"] == "s-42"

    def test_scalars_arrive_as_environment_variables(self, hooks_home):
        """The shape one-liners actually use."""
        _write(hooks_home.workspace, {
            "hooks": {"step_start": [
                {"command": "echo $INFINIDEV_HOOK_EVENT/$INFINIDEV_HOOK_STEP_INDEX"},
            ]},
        })
        output = run_hooks(UserHookEvent.STEP_START, {"step_index": 7})
        assert output.text == "step_start/7"

    def test_booleans_become_one_and_zero(self, hooks_home):
        """`[ "$X" = 1 ]` is what people write; "True" would break it."""
        _write(hooks_home.workspace, {
            "hooks": {"task_end_summary": [{"command": "echo $INFINIDEV_HOOK_CHANGED"}]},
        })
        assert run_hooks(
            UserHookEvent.TASK_END_SUMMARY, {"changed": True},
        ).text == "1"

    def test_nonzero_exit_discards_the_output(self, hooks_home):
        """A failing hook must not put half-written text into the prompt."""
        _write(hooks_home.workspace, {
            "hooks": {"task_start": [{"command": "echo partial; exit 3"}]},
        })
        output = run_hooks(UserHookEvent.TASK_START)
        assert output.text == ""
        assert output.failed == 1

    def test_timeout_discards_the_output(self, hooks_home):
        """A hanging hook is killed rather than freezing the loop."""
        _write(hooks_home.workspace, {
            "hooks": {"task_start": [{"command": "sleep 5", "timeout": 0.3}]},
        })
        output = run_hooks(UserHookEvent.TASK_START)
        assert output.text == ""
        assert output.failed == 1

    def test_missing_binary_is_survivable(self, hooks_home):
        """A typo'd command is a warning, not a crashed run."""
        _write(hooks_home.workspace, {
            "hooks": {"task_start": [{"command": "definitely-not-a-real-binary-xyz"}]},
        })
        assert run_hooks(UserHookEvent.TASK_START).text == ""

    def test_one_failing_hook_does_not_silence_the_others(self, hooks_home):
        _write(hooks_home.workspace, {
            "hooks": {"task_start": [{"command": "exit 1"}, {"prompt": "survivor"}]},
        })
        output = run_hooks(UserHookEvent.TASK_START)
        assert output.text == "survivor"
        assert output.failed == 1

    def test_silent_hook_contributes_nothing(self, hooks_home):
        """Printing nothing is how a conditional hook says 'not this time'."""
        _write(hooks_home.workspace, {"hooks": {"task_start": [{"command": "true"}]}})
        assert not run_hooks(UserHookEvent.TASK_START)

    def test_output_is_capped(self, hooks_home):
        """A hook that cats a log file cannot eat the context window."""
        from infinidev.engine.user_hooks.runner import MAX_OUTPUT_CHARS

        _write(hooks_home.workspace, {
            "hooks": {"task_start": [
                {"command": f"{sys.executable} -c \"print('x' * {MAX_OUTPUT_CHARS * 2})\""},
            ]},
        })
        output = run_hooks(UserHookEvent.TASK_START)
        assert "truncated" in output.text
        assert len(output.text) < MAX_OUTPUT_CHARS * 2

    def test_command_runs_in_the_workspace(self, hooks_home):
        """cwd is the workspace, so `git`-shaped hooks see the project."""
        (hooks_home.workspace / "marker.txt").write_text("here", encoding="utf-8")
        _write(hooks_home.workspace, {"hooks": {"task_start": [{"command": "ls"}]}})
        output = run_hooks(
            UserHookEvent.TASK_START, workspace_path=str(hooks_home.workspace),
        )
        assert "marker.txt" in output.text


# ── the end-of-step gate ─────────────────────────────────────────────────────


class _FakeEngine:
    """Just the one method the hook gate calls back into."""

    @staticmethod
    def _overwrite_step_complete_tool_result(messages, call_id, body):
        for message in reversed(messages):
            if message.get("tool_call_id") == call_id:
                message["content"] = body
                return
        messages.append(
            {"role": "tool", "tool_call_id": call_id, "content": body}
        )


def _fake_call(call_id: str = "call_1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(arguments='{"status": "continue"}'),
    )


def _fake_ctx(workspace, step_index: int = 3):
    state = LoopState()
    state.plan.steps.append(_step(step_index))
    return SimpleNamespace(
        state=state,
        workspace_path=str(workspace),
        project_id=1,
        agent_id="agent-1",
        agent_name="developer",
        desc="do the thing",
    )


def _step(index: int):
    from infinidev.engine.loop.plan_step import PlanStep

    return PlanStep(index=index, title=f"step {index}", status="active")


class TestStepEndInstructionGate:
    """The end-of-step hook holds the step open for exactly one more pass."""

    def test_no_hook_lets_the_step_close(self, hooks_home):
        gate = StepCompleteGate(_FakeEngine())
        assert gate._user_hook_holds(
            _fake_ctx(hooks_home.workspace), _fake_call(), [],
        ) is False

    def test_hook_output_holds_the_step(self, hooks_home):
        """The tool result is overwritten, which is how every gate refuses."""
        _write(hooks_home.workspace, {
            "hooks": {"step_end_instruction": [{"prompt": "review the diff"}]},
        })
        gate = StepCompleteGate(_FakeEngine())
        messages = [{"role": "tool", "tool_call_id": "call_1", "content": "ok"}]
        assert gate._user_hook_holds(
            _fake_ctx(hooks_home.workspace), _fake_call(), messages,
        ) is True
        assert "review the diff" in messages[0]["content"]
        assert "step_complete" in messages[0]["content"]

    def test_fires_only_once_per_step(self, hooks_home):
        """A hook that re-fired on the retry would hold the step forever."""
        _write(hooks_home.workspace, {
            "hooks": {"step_end_instruction": [{"prompt": "review the diff"}]},
        })
        gate = StepCompleteGate(_FakeEngine())
        ctx = _fake_ctx(hooks_home.workspace)
        assert gate._user_hook_holds(ctx, _fake_call(), []) is True
        assert gate._user_hook_holds(ctx, _fake_call(), []) is False

    def test_each_step_gets_its_own_turn(self, hooks_home):
        """Firing is keyed by step index, not by a single run-wide flag."""
        _write(hooks_home.workspace, {
            "hooks": {"step_end_instruction": [{"prompt": "review"}]},
        })
        gate = StepCompleteGate(_FakeEngine())
        first = _fake_ctx(hooks_home.workspace, step_index=1)
        second = _fake_ctx(hooks_home.workspace, step_index=2)
        assert gate._user_hook_holds(first, _fake_call(), []) is True
        assert gate._user_hook_holds(second, _fake_call(), []) is True

    def test_silent_hook_still_burns_its_one_shot(self, hooks_home):
        """Otherwise a sometimes-silent hook could fire twice in one step."""
        _write(hooks_home.workspace, {
            "hooks": {"step_end_instruction": [{"command": "true"}]},
        })
        gate = StepCompleteGate(_FakeEngine())
        ctx = _fake_ctx(hooks_home.workspace)
        assert gate._user_hook_holds(ctx, _fake_call(), []) is False
        assert gate._hook_fired == {3}

    def test_reset_run_forgets_which_steps_fired(self, hooks_home):
        gate = StepCompleteGate(_FakeEngine())
        gate._hook_fired.add(1)
        gate.reset_run()
        assert gate._hook_fired == set()


# ── surviving (or not) the summary ───────────────────────────────────────────


def _prompt_with(records, **kwargs) -> str:
    state = LoopState()
    state.history.extend(records)
    return build_iteration_prompt("task", "expected", state, **kwargs)


class TestHookNotesSurviveTheSummary:
    """step_end_summary output rides the record the prompt is rebuilt from."""

    def test_hook_note_renders_in_previous_actions(self):
        prompt = _prompt_with([
            ActionRecord(step_index=1, summary="did a thing", hook_notes="COVERAGE 91%"),
        ])
        assert "COVERAGE 91%" in prompt

    def test_absent_note_adds_nothing(self):
        prompt = _prompt_with([ActionRecord(step_index=1, summary="did a thing")])
        assert "Hook:" not in prompt

    def test_note_survives_the_retention_collapse(self, monkeypatch):
        """Older records collapse to one line — the hook note comes along."""
        monkeypatch.setattr(settings, "WORKING_MEMORY_VERBATIM_STEPS", 1)
        prompt = _prompt_with([
            ActionRecord(step_index=1, summary="old", hook_notes="EARLY NOTE"),
            ActionRecord(step_index=2, summary="new", hook_notes="LATE NOTE"),
        ])
        assert "EARLY NOTE" in prompt
        assert "LATE NOTE" in prompt

    def test_collapsed_note_is_clipped(self, monkeypatch):
        """Clipped, not dropped: reachable forever, not full-width forever."""
        from infinidev.engine.loop.context import _COLLAPSED_HOOK_CHARS

        monkeypatch.setattr(settings, "WORKING_MEMORY_VERBATIM_STEPS", 1)
        prompt = _prompt_with([
            ActionRecord(step_index=1, summary="old", hook_notes="N" * 500),
            ActionRecord(step_index=2, summary="new"),
        ])
        assert "N" * 500 not in prompt
        assert "N" * _COLLAPSED_HOOK_CHARS in prompt  # the clip keeps a prefix
        assert "…" in prompt


# ── the task-level hooks ─────────────────────────────────────────────────────


class TestTaskHookHelpers:
    """The pipeline-side helpers, including the anti-recursion switch."""

    def test_task_start_wraps_output_in_an_attributed_block(self, hooks_home):
        from infinidev.engine.orchestration.pipeline import _run_task_start_hook

        _write(hooks_home.workspace, {
            "hooks": {"task_start": [{"prompt": "branch is frozen"}]},
        })
        block = _run_task_start_hook(
            user_input="hi", session_id="s1", skip=False,
        )
        assert "branch is frozen" in block
        assert 'event="task_start"' in block

    def test_task_start_is_skipped_on_a_reentered_turn(self, hooks_home):
        """A re-entered turn is the same turn continuing, not a new one."""
        from infinidev.engine.orchestration.pipeline import _run_task_start_hook

        _write(hooks_home.workspace, {"hooks": {"task_start": [{"prompt": "x"}]}})
        assert _run_task_start_hook(
            user_input="hi", session_id="s1", skip=True,
        ) == ""

    def test_task_end_instruction_returns_the_text(self, hooks_home):
        from infinidev.engine.orchestration.pipeline import _task_end_hook

        _write(hooks_home.workspace, {
            "hooks": {"task_end_instruction": [{"prompt": "now review it"}]},
        })
        assert _task_end_hook(
            UserHookEvent.TASK_END_INSTRUCTION,
            user_input="hi", session_id="s1", result="done",
            files_changed=True, status="done",
        ) == "now review it"

    def test_skip_stops_the_second_reentry(self, hooks_home):
        """This is what bounds the follow-up to exactly one extra pass."""
        from infinidev.engine.orchestration.pipeline import _task_end_hook

        _write(hooks_home.workspace, {
            "hooks": {"task_end_instruction": [{"prompt": "again"}]},
        })
        assert _task_end_hook(
            UserHookEvent.TASK_END_INSTRUCTION,
            user_input="hi", session_id="s1", result="done",
            files_changed=True, status="done", skip=True,
        ) == ""

    def test_task_instruction_is_attributed_to_the_hook(self):
        """The model must not read a hook's text as the user's own words."""
        from infinidev.engine.user_hooks import task_instruction

        rendered = task_instruction("review everything")
        assert "review everything" in rendered
        assert "hook" in rendered.lower()

    def test_changed_files_probe_tolerates_a_trackerless_engine(self):
        """The legacy PhaseEngine has no tracker; the finish path still runs."""
        from infinidev.engine.orchestration.pipeline import _turn_changed_files

        assert _turn_changed_files(object()) is False
        assert _turn_changed_files(
            SimpleNamespace(has_file_changes=lambda: True),
        ) is True
