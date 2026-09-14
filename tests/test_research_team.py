"""Regression coverage for durable team coordination and scoped workers."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from infinidev.engine.loop.engine import LoopEngine
from infinidev.engine.team.runtime import ROOT, TeamRuntime, clone_tools
from infinidev.prompts.profiles import EffectivePromptConfiguration
from infinidev.prompts.team import build_team_identity
from infinidev.tools.base.context import bind_tools_to_agent, set_context
from infinidev.tools.file import CreateFileTool, ReadFileTool
from infinidev.tools.team import build_team_tools


def _finish(team, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with team._lock:
            if not team._active:
                return
        team.wait(0.01)
    pytest.fail("Workers did not finish")


@pytest.fixture
def team(temp_db, workspace_dir):
    teams = []

    def create(**kwargs):
        def runner(runtime, member, ticket, assignment):
            runtime.poll(member["id"])
            return "Observed sample.txt:1; read only", "completed"

        runtime = TeamRuntime(
            session_id=kwargs.pop("session_id", "research-session"), project_id=1,
            workspace_path=str(workspace_dir), root_agent_id="root-agent",
            catalog=[ReadFileTool(), CreateFileTool()], worker_runner=kwargs.pop("runner", runner),
            **kwargs,
        )
        teams.append(runtime)
        return runtime

    yield create
    for runtime in teams:
        if runtime.store.snapshot().get("running"):
            runtime.close()


def _ticket(team, **kwargs):
    return team.create_ticket(ROOT, title="Check cache", objective="Does the cache detach?",
                              acceptance=["Cite inspected code and limitations"], constraints=[],
                              dependencies=kwargs.get("dependencies", []))


def _delegate(team, ticket, **kwargs):
    return team.delegate(ROOT, ticket_id=ticket["id"], name=kwargs.pop("name", "Auditor"),
                         system_prompt="Audit the cache path and cite source evidence.",
                         tools=kwargs.pop("tools", ["read_file"]), **kwargs)


def test_member_name_and_display_role_reach_messages_and_notes(team):
    runtime = team()
    member = _delegate(runtime, _ticket(runtime), name="Lucía", role="Researcher")
    _finish(runtime)
    assert member["role"] == "Researcher"
    assert runtime.read()["agents"][1]["role"] == "Researcher"
    message = runtime.send(ROOT, recipient="Lucía", content="Check the gradients")
    assert message["recipient"] == member["id"]
    assert message["recipient_label"] == "Lucía · Researcher"
    assert message["author_label"] == "Orchestrator"
    note = runtime.write_note(member["id"], content="Cache inspected", kind="observation", refs=[])
    assert note["author_label"] == "Lucía · Researcher"
    _finish(runtime)
    runtime.close()
    resumed = team()
    assert resumed.read()["agents"][1]["role"] == "Researcher"
    assert resumed.read(view="notes")["events"][0]["author_label"] == "Lucía · Researcher"


def test_delegation_schema_requires_a_descriptive_role():
    from infinidev.tools.team.tools import DelegateInput

    payload = dict(ticket_id="t_1", name="Lucía", system_prompt="Inspect source", tools=[])
    with pytest.raises(ValidationError):
        DelegateInput(**payload)
    with pytest.raises(ValidationError):
        DelegateInput(**payload, role="   ")
    assert DelegateInput(**payload, role="Researcher").role == "Researcher"


def test_reports_require_review_and_dependencies_require_acceptance(team):
    runtime = team()
    first = _ticket(runtime)
    dependent = _ticket(runtime, dependencies=[first["id"]])
    with pytest.raises(ValueError, match="Dependencies"):
        _delegate(runtime, dependent)
    _delegate(runtime, first)
    _finish(runtime)
    assert runtime.store.snapshot()["tickets"][first["id"]]["status"] == "review"
    assert first["id"] in runtime.completion_blocker()
    runtime.review(ROOT, ticket_id=first["id"], decision="accepted", reason="Checked source")
    _delegate(runtime, dependent, name="Second auditor")
    _finish(runtime)


def test_note_provenance_supersession_and_session_isolation(team):
    runtime = team()
    member = _delegate(runtime, _ticket(runtime))
    _finish(runtime)
    first = runtime.write_note(member["id"], content="Cache is detached", kind="observation",
                               refs=["cache.py:42"])
    replacement = runtime.write_note(ROOT, content="Only the inference cache is detached",
                                     kind="observation", refs=["cache.py:51"],
                                     supersedes=first["id"])
    assert first["author"] == member["id"] and first["created_at"]
    assert runtime.store.event(first["id"])["superseded_by"] == replacement["id"]
    assert runtime.store.event(first["id"])["content"] == first["content"]
    with pytest.raises(ValueError, match="current note"):
        runtime.write_note(ROOT, content="Lost race", kind="decision", refs=[],
                           supersedes=first["id"])
    other = team(session_id="other")
    assert other.read(view="notes")["events"] == []
    with pytest.raises(ValueError, match="Unknown event"):
        other.store.event(first["id"])


def test_concurrent_notes_are_not_lost_and_pages_are_stable(team):
    runtime = team()
    with ThreadPoolExecutor(max_workers=6) as pool:
        notes = list(pool.map(lambda n: runtime.write_note(
            ROOT, content=f"Observation {n}", kind="observation", refs=[]), range(18)))
    first = runtime.read(view="notes", limit=7)
    second = runtime.read(view="notes", after=first["next_after"], limit=100)
    assert {e["id"] for e in first["events"] + second["events"]} == {n["id"] for n in notes}


def test_idle_peer_wakes_for_question_and_reply_is_threaded_without_ping_pong(team):
    runs = []

    def runner(runtime, member, ticket, assignment):
        updates = json.loads(runtime.poll(member["id"]) or "[]")
        runs.append((member["id"], assignment))
        for message in updates:
            if message["kind"] == "message" and message["reply_to"] is None:
                runtime.send(member["id"], recipient=message["author"], content="Yes: cache.py:42",
                             reply_to=message["id"])
        return "Source checked", "completed"

    runtime = team(runner=runner)
    alice = _delegate(runtime, _ticket(runtime), name="Alice")
    bob = _delegate(runtime, _ticket(runtime), name="Bob")
    _finish(runtime)
    question = runtime.send(alice["id"], recipient="Bob", content="Does the cache detach?")
    _finish(runtime)
    messages = runtime.read(view="messages")["events"]
    assert messages[-1]["reply_to"] == question["id"]
    assert messages[-1]["author"] == bob["id"]
    assert runs.count((bob["id"], False)) == 1
    assert runs.count((alice["id"], False)) == 1
    assert runtime.store.snapshot()["tickets"][bob["ticket_id"]]["result"] == "Source checked"


def test_late_request_is_not_stranded_when_worker_becomes_idle(team):
    entered, release = threading.Event(), threading.Event()
    questions = []

    def runner(runtime, member, ticket, assignment):
        updates = json.loads(runtime.poll(member["id"]) or "[]")
        questions.extend(e for e in updates if e["kind"] == "message")
        if assignment:
            entered.set()
            assert release.wait(2)
        return "Report", "completed"

    runtime = team(runner=runner)
    member = _delegate(runtime, _ticket(runtime))
    assert entered.wait(2)
    request = runtime.send(ROOT, recipient=member["id"], content="Also identify the caller")
    release.set()
    _finish(runtime)
    assert [e["id"] for e in questions] == [request["id"]]


def test_worker_cannot_delegate_or_grant_unknown_tools(team):
    runtime = team()
    ticket = _ticket(runtime)
    with pytest.raises(ValueError, match="Unavailable tools"):
        _delegate(runtime, ticket, tools=["execute_command"])
    member = _delegate(runtime, ticket)
    _finish(runtime)
    with pytest.raises(ValueError, match="Only the orchestrator"):
        runtime.create_ticket(member["id"], title="x", objective="x", acceptance=["x"],
                              constraints=[], dependencies=[])
    tools = build_team_tools(runtime, member["id"], orchestrator=False)
    assert {t.name for t in tools} == {
        "team_read", "team_write_note", "team_send_message", "team_idle", "team_wait",
    }


def test_bound_authorship_and_argument_validation(team):
    runtime = team()
    set_context(agent_id="root-agent", project_id=1, session_id=runtime.session_id)
    tool = next(t for t in build_team_tools(runtime, "root-agent", orchestrator=True)
                if t.name == "team_write_note")
    note = json.loads(tool.run(content="Measured result", kind="observation", author="someone-else"))
    assert note["author"] == ROOT
    with pytest.raises(ValidationError):
        tool.run(content="   ", kind="observation")


def test_tool_copies_keep_independent_agent_bindings():
    original = ReadFileTool()
    bind_tools_to_agent([original], "root")
    child = clone_tools([original], "child")[0]
    assert original._bound_agent_id == "root"
    assert child._bound_agent_id == "child"


def test_restore_keeps_history_and_rejects_two_live_owners(team):
    runtime = team()
    note = runtime.write_note(ROOT, content="Prior decision", kind="decision", refs=[])
    with pytest.raises(ValueError, match="already has a running team"):
        team()
    runtime.close()
    restored = team()
    assert restored.store.event(note["id"])["author"] == ROOT


def test_dead_owner_recovers_running_ticket_as_interrupted(team, monkeypatch):
    runtime = team()
    ticket = _ticket(runtime)
    runtime.close()
    runtime.store.update(lambda state, emit: (
        state.update(running=True, owner_pid=1234567),
        state["tickets"][ticket["id"]].update(status="running"),
    ))
    monkeypatch.setattr("infinidev.engine.team.runtime._pid_alive", lambda pid: False)
    restored = team()
    assert restored.store.snapshot()["tickets"][ticket["id"]]["status"] == "interrupted"


def test_workspace_writers_are_serialized(team):
    lock = threading.Lock()
    active, peak = 0, 0

    def runner(runtime, member, ticket, assignment):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        runtime.poll(member["id"])
        threading.Event().wait(0.03)
        with lock:
            active -= 1
        return "Changed file", "completed"

    runtime = team(runner=runner)
    _delegate(runtime, _ticket(runtime), name="Writer A", tools=["create_file"])
    _delegate(runtime, _ticket(runtime), name="Writer B", tools=["create_file"])
    _finish(runtime)
    assert peak == 1


def test_idle_releases_worker_capacity_and_writer_lease_for_a_peer_reply(team):
    sleeping = threading.Event()
    resumed = []

    def runner(runtime, member, ticket, assignment):
        runtime.poll(member["id"])
        if member["name"] == "Alice":
            sleeping.set()
            result = runtime.idle(member["id"], events=["message"], sender="Bob", timeout=2)
            resumed.append(result)
        else:
            runtime.send(member["id"], recipient="Alice", content="Source checked")
        return "Report", "completed"

    runtime = team(runner=runner, max_workers=1)
    # Register Bob while Alice still owns the sole execution slot.
    with runtime._lock:
        _delegate(runtime, _ticket(runtime), name="Alice", tools=["create_file"])
        _delegate(runtime, _ticket(runtime), name="Bob", tools=["create_file"])
    assert sleeping.wait(2)
    _finish(runtime)
    assert resumed[0]["reason"] == "event"
    assert resumed[0]["events"][0]["content"] == "Source checked"


def test_idle_filters_events_and_user_guidance_always_wakes(team):
    runtime = team()
    member = _delegate(runtime, _ticket(runtime))
    _finish(runtime)
    runtime.poll(ROOT)
    with ThreadPoolExecutor(max_workers=1) as pool:
        waiting = pool.submit(runtime.idle, ROOT, events=["report"], timeout=2)
        runtime.send(member["id"], recipient=ROOT, content="FYI", message_type="info")
        runtime.write_note(member["id"], content="Observation", kind="observation", refs=[])
        assert not waiting.done()
        runtime.forward_user_guidance("Stop the experiment and inspect the source")
        result = waiting.result(2)
    assert result["reason"] == "user_guidance"
    assert runtime.store.snapshot()["agents"][ROOT]["status"] == "running"


def test_idle_status_callback_does_not_hold_the_team_lock(team):
    entered, sent = threading.Event(), threading.Event()
    delivered_during_callback = []

    def on_status(level, message):
        if "idle —" in message:
            entered.set()
            delivered_during_callback.append(sent.wait(2))

    runtime = team(on_status=on_status)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(runtime.idle, ROOT, events=["message"])
        assert entered.wait(2)
        runtime.forward_user_guidance("Continue with the new evidence")
        sent.set()
        assert pending.result(2)["reason"] == "user_guidance"
    assert delivered_during_callback == [True]


def test_principal_discards_cached_source_after_sleep_without_resetting_global_diff(team):
    from infinidev.engine.loop.loop_state import LoopState
    from infinidev.tools.base.context import set_loop_state

    sleeping = threading.Event()
    runtime = team(on_status=lambda level, message: sleeping.set())
    state = LoopState()
    state.cache_file("cache.py", "stale source")
    state.read_delivery_revisions["cache.py"] = "old revision"
    set_loop_state("root-agent", state)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(runtime.idle, ROOT, events=["message"])
        assert sleeping.wait(2)
        runtime.forward_user_guidance("Inspect the updated cache")
        assert pending.result(2)["reason"] == "user_guidance"
    assert not state.opened_files
    assert not state.read_delivery_revisions


def test_reply_subscription_handles_reply_before_wait_and_reports_delivery(team):
    runtime = team()
    member = _delegate(runtime, _ticket(runtime))
    _finish(runtime)
    request = runtime.send(ROOT, recipient=member["id"], content="Check the caller")
    _finish(runtime)
    reply = runtime.send(member["id"], recipient=ROOT, content="caller.py:12",
                         reply_to=request["id"])
    runtime.poll(ROOT)
    result = runtime.idle(ROOT, events=["message"], reply_to=request["id"], timeout=0.2)
    assert result["reason"] == "event"
    assert result["events"][0]["id"] == reply["id"]
    thread = runtime.read(view="messages", thread_id=request["id"])["events"]
    assert [e["id"] for e in thread] == [request["id"], reply["id"]]
    assert thread[0]["delivery"] == "answered"
    assert thread[1]["delivery"] == "delivered"


def test_cancel_wakes_idle_without_waiting_for_timeout(team):
    runtime = team()
    runtime.poll(ROOT)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(runtime.idle, ROOT, events=["message"])
        runtime.cancel()
        assert pending.result(2)["reason"] == "cancelled"


def test_idle_delivery_does_not_swallow_unmatched_peer_or_user_updates(team):
    runtime = team()
    member = _delegate(runtime, _ticket(runtime))
    _finish(runtime)
    runtime.poll(ROOT)
    note = runtime.write_note(member["id"], content="Caveat", kind="observation", refs=[])
    message = runtime.send(member["id"], recipient=ROOT, content="Answer")
    result = runtime.idle(ROOT, events=["message"], timeout=0)
    assert result["events"][0]["id"] == message["id"]
    updates = json.loads(runtime.poll(ROOT))
    assert note["id"] in [e["id"] for e in updates]
    assert message["id"] not in [e["id"] for e in updates]
    assert runtime.idle(ROOT, events=["message"], timeout=0)["reason"] == "timeout"


def test_idle_tool_cancellation_and_team_close_release_a_sleeping_worker(team):
    entered = threading.Event()
    results = []

    def runner(runtime, member, ticket, assignment):
        engine = LoopEngine()
        engine._team_runtime, engine._team_actor = runtime, member["id"]
        runtime.attach_engine(member["id"], engine)
        runtime.poll(member["id"])
        entered.set()
        results.append(runtime.idle(member["id"], events=["message"]))
        return "Stopped", "cancelled"

    runtime = team(runner=runner, max_workers=1)
    member = _delegate(runtime, _ticket(runtime))
    assert entered.wait(2)
    runtime.review(ROOT, ticket_id=member["ticket_id"], decision="cancelled", reason="User stopped")
    _finish(runtime)
    assert results[0]["reason"] == "cancelled"
    assert not runtime._executing
    assert runtime._writing is None
    assert "waiting" not in runtime.store.snapshot()["agents"][member["id"]]


def test_idle_rejects_unknown_sources_and_incompatible_filters(team):
    runtime = team()
    with pytest.raises(ValueError, match="Unknown recipient"):
        runtime.idle(ROOT, sender="Missing")
    with pytest.raises(ValueError, match="Unknown background task"):
        runtime.idle(ROOT, events=["background_task"], task_ids=["bg-missing"])
    with pytest.raises(ValidationError, match="reply_to requires"):
        runtime.idle(ROOT, events=["report"], reply_to=1)


def test_information_does_not_spawn_workers_and_reply_to_reply_does_not_ping_pong(team):
    calls = []

    def runner(runtime, member, ticket, assignment):
        calls.append(member["id"])
        runtime.poll(member["id"])
        return "Report", "completed"

    runtime = team(runner=runner)
    member = _delegate(runtime, _ticket(runtime))
    _finish(runtime)
    request = runtime.send(ROOT, recipient=member["id"], content="Question")
    _finish(runtime)
    reply = runtime.send(member["id"], recipient=ROOT, content="Answer", reply_to=request["id"])
    runtime.send(ROOT, recipient=member["id"], content="Thanks", reply_to=reply["id"])
    runtime.send(ROOT, recipient=member["id"], content="An update", message_type="info")
    assert len(calls) == 2
    assert not runtime.has_active_workers


def test_legacy_conversation_migration_preserves_nested_reply_threads(temp_db):
    from infinidev.engine.team.store import TeamStore
    from infinidev.tools.base.db import execute_with_retry

    def legacy(conn):
        conn.executescript(
            "CREATE TABLE research_team_events (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " team_id TEXT, kind TEXT, author TEXT, recipient TEXT, ticket_id TEXT,"
            " reply_to INTEGER, supersedes INTEGER, content TEXT, refs TEXT, created_at TEXT);"
            "INSERT INTO research_team_events VALUES"
            " (1,'legacy','message','a','b',NULL,NULL,NULL,'Question','[]','2026-01-01'),"
            " (2,'legacy','message','b','a',NULL,1,NULL,'Answer','[]','2026-01-01'),"
            " (3,'legacy','message','a','b',NULL,2,NULL,'Follow-up','[]','2026-01-01');"
        )

    execute_with_retry(legacy)
    store = TeamStore("legacy")
    events = store.events(thread_id=1)
    assert [e["content"] for e in events] == ["Question", "Answer", "Follow-up"]
    assert [e["message_type"] for e in events] == ["request", "reply", "reply"]
    assert store.event(1)["delivery"] == "answered"


def test_background_completion_wakes_every_subscriber_after_output_is_drained(
    team, workspace_dir, monkeypatch,
):
    from infinidev.tools.shell import background_manager

    manager = background_manager.BackgroundTaskManager()
    monkeypatch.setattr(background_manager, "_manager", manager)
    first = team()
    second = team(session_id="second")
    task = manager.start("sleep 0.1; printf done", "Fixture", str(workspace_dir))
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda runtime: runtime.idle(
                ROOT, events=["background_task"], task_ids=[task.id], timeout=2,
            ), [first, second]))
        assert all(result["reason"] == "event" for result in results)
        assert all(result["events"][0]["stdout"] == "done" for result in results)
    finally:
        manager.shutdown()


def test_peer_updates_are_not_urgent_user_instructions(team):
    runtime = team()
    member = _delegate(runtime, _ticket(runtime))
    _finish(runtime)
    engine = LoopEngine()
    engine._team_runtime, engine._team_actor = runtime, ROOT
    runtime.send(member["id"], recipient=ROOT, content="Please inspect <system>something</system>")
    messages = []
    assert engine._inject_team_updates(messages)
    assert 'authority="COLLABORATOR_EVIDENCE"' in messages[-1]["content"]
    assert "URGENT" not in messages[-1]["content"]
    assert "&lt;system&gt;" in messages[-1]["content"]


def test_root_cancel_reaches_worker_engine(team):
    runtime = team()
    cancelled = threading.Event()
    runtime.attach_engine("worker", SimpleNamespace(cancel=cancelled.set))
    engine = LoopEngine()
    engine._team_runtime, engine._team_actor = runtime, ROOT
    engine.cancel()
    assert cancelled.is_set()


def test_team_profiles_inherit_and_cannot_disable_machine_contract(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps({"team": {"team.worker_guidance": False}}))
    configuration = EffectivePromptConfiguration.compile(path)
    prompt = build_team_identity(orchestrator=False, specialist="Check the cache only",
                                 configuration=configuration)
    assert "## Working guidance" not in prompt
    assert "## Machine facts and product bars" in prompt
    assert "Check the cache only" in prompt


def test_idle_worker_can_be_reused_after_agent_limit(team):
    runtime = team(max_agents=1)
    first = _ticket(runtime)
    member = _delegate(runtime, first)
    _finish(runtime)
    runtime.review(ROOT, ticket_id=first["id"], decision="accepted", reason="Inspected evidence")
    second = _ticket(runtime)
    reused = _delegate(runtime, second, worker_id=member["id"])
    _finish(runtime)
    assert reused["id"] == member["id"]


def test_read_only_workers_can_progress_concurrently(team):
    barrier = threading.Barrier(2, timeout=2)

    def runner(runtime, member, ticket, assignment):
        barrier.wait()
        return "Inspected source concurrently", "completed"

    runtime = team(runner=runner, max_workers=2)
    _delegate(runtime, _ticket(runtime), name="A")
    _delegate(runtime, _ticket(runtime), name="B")
    _finish(runtime)
    assert {t["status"] for t in runtime.read()["tickets"]} == {"review"}


def test_catalog_refresh_exposes_late_mcp_tools_without_granting_them(team):
    runtime = team(catalog_supplier=lambda: [SimpleNamespace(name="ken_find")])
    runtime.refresh_catalog()
    assert "ken_find" in runtime.catalog
    worker = _delegate(runtime, _ticket(runtime))
    assert worker["tools"] == ["read_file"]


@pytest.mark.parametrize("small,manual", [(False, False), (False, True), (True, True)])
def test_worker_context_enforces_grants_and_inherits_prompt_rules(
    team, workspace_dir, monkeypatch, small, manual,
):
    from infinidev.engine.loop import context_builder
    from infinidev.engine.team.worker import run_worker

    (workspace_dir / "AGENTS.md").write_text("Keep research result artifacts immutable.")
    monkeypatch.setattr(context_builder, "get_litellm_params",
                        lambda: {"model": "openai/gpt-6-astra"})
    monkeypatch.setattr(context_builder, "_is_small_model", lambda: small)
    monkeypatch.setattr(context_builder, "get_model_capabilities",
                        lambda: SimpleNamespace(supports_function_calling=not manual))
    captured = {}

    def execute(engine, agent, prompt, **kwargs):
        captured["context"] = context_builder.build_execution_context(
            engine, agent, prompt, verbose=False, **kwargs,
        )
        captured["kwargs"] = kwargs
        captured["prompt"] = prompt
        from infinidev.tools.base.context import get_context_for_agent
        captured["session"] = get_context_for_agent(agent.agent_id).session_id
        assert engine._tool_allowlist_locked
        engine._last_status = "done"
        return "Checked source; hypothesis rejected"

    monkeypatch.setattr(LoopEngine, "execute", execute)
    runtime = team(runner=run_worker, user_request="Comprueba el cache, no entrenes.",
                   attachments=["attachment sentinel"])
    member = _delegate(runtime, _ticket(runtime))
    _finish(runtime)
    ctx = captured["context"]
    assert {tool.name for tool in ctx.tools} == {
        "read_file", "team_send_message", "team_write_note", "team_read", "team_idle", "team_wait",
    }
    assert "create_file" not in ctx.tool_dispatch
    assert "team_delegate" not in ctx.tool_dispatch
    assert "Keep research result artifacts immutable." in ctx.system_prompt
    assert "Audit the cache path" in ctx.system_prompt
    assert "## Machine facts and product bars" in ctx.system_prompt
    assert "Comprueba el cache, no entrenes." in captured["prompt"][0]
    assert captured["session"] == f"research-session:worker:{member['id']}"
    assert captured["kwargs"]["initial_attachments"] == ["attachment sentinel"]
    assert captured["kwargs"]["prompt_configuration"] is runtime.prompt_configuration


@pytest.mark.parametrize("force_gather", [False, True])
def test_orchestrator_cannot_return_success_with_unreviewed_worker_report(
    temp_db, workspace_dir, monkeypatch, force_gather,
):
    from infinidev.engine.engines.orchestrator import OrchestratorAdapter
    from infinidev.engine.orchestration.escalation_packet import EscalationPacket
    from infinidev.engine.team.worker import TeamAgent

    captured = {}

    def gather(**kwargs):
        captured["gather"] = kwargs
        return ("Gathered evidence\n" + kwargs["task_prompt"][0], kwargs["task_prompt"][1])

    monkeypatch.setattr("infinidev.engine.orchestration.pipeline._run_gather_phase", gather)

    def worker(runtime, member, ticket, assignment):
        runtime.attach_engine(member["id"], SimpleNamespace(
            _last_state=SimpleNamespace(total_prompt_tokens=30, total_completion_tokens=7),
            _last_total_tool_calls=2,
        ))
        return "Source inspected", "completed"

    def execute(engine, agent, prompt, **kwargs):
        runtime = engine._team_runtime
        runtime._runner = worker
        assert ("Gathered evidence" in prompt[0]) is force_gather
        assert "create_file" not in {t.name for t in agent.tools}
        assert "team_delegate" in {t.name for t in agent.tools}
        _delegate(runtime, _ticket(runtime))
        _finish(runtime)
        captured["store"] = runtime.store
        engine._last_state = SimpleNamespace(total_prompt_tokens=100, total_completion_tokens=10)
        engine._last_total_tool_calls = 3
        engine._last_status = "done"
        return "Everything is done"

    monkeypatch.setattr(LoopEngine, "execute", execute)
    agent = TeamAgent("root-agent", "Lead", 1, str(workspace_dir),
                      [ReadFileTool(), CreateFileTool()])
    engine = LoopEngine()
    hooks = SimpleNamespace(on_phase=lambda *a: None, on_status=lambda *a: None)
    result = OrchestratorAdapter().run(
        agent=agent, engine=engine, hooks=hooks, project_id=1,
        session_id="review-gate", workspace_path=str(workspace_dir),
        escalation=EscalationPacket(user_request="Audit the cache", understanding=""),
        force_gather=force_gather,
    )
    assert result.status == "blocked"
    assert "Team work remains" in result.user_message
    assert not captured["store"].snapshot()["running"]
    assert engine._team_runtime is None
    assert ("gather" in captured) is force_gather
    assert result.metrics["observed_prompt_tokens"] == 130
    assert result.metrics["observed_completion_tokens"] == 17
    assert result.metrics["observed_tool_calls"] == 5


def test_legitimate_waits_do_not_trigger_repetition_but_errors_still_do():
    from infinidev.engine.loop.loop_guard import LoopGuard

    guard = LoopGuard()
    for _ in range(20):
        guard.on_tool_result("team_wait", "{}", False, awaiting_workers=True)
    assert guard.same_tool_streak == 0
    assert guard.non_progress_tool_calls == 0
    guard.on_tool_result("team_wait", "{}", True, awaiting_workers=True)
    assert guard.consecutive_tool_errors == 1
    guard.on_tool_result("team_wait", "{}", False, awaiting_workers=False)
    assert guard.non_progress_tool_calls == 2


@pytest.mark.parametrize("name", ["team_idle", "team_wait"])
def test_idle_does_not_start_an_auxiliary_model_call(name, monkeypatch):
    from infinidev.engine.loop.critic_liaison import CriticLiaison

    liaison = CriticLiaison()
    monkeypatch.setattr(liaison, "get", lambda ctx: pytest.fail("Idle must not call the critic"))
    call = SimpleNamespace(function=SimpleNamespace(name=name))
    assert liaison.review_alongside(None, [], [call], None, lambda: 7) == 7


def test_real_user_guidance_keeps_user_authority_in_worker_context(team):
    runtime = team()
    member = _delegate(runtime, _ticket(runtime))
    _finish(runtime)
    runtime.forward_user_guidance("No ejecutes entrenamiento; revisa sólo el código.")
    engine = LoopEngine()
    engine._team_runtime, engine._team_actor = runtime, member["id"]
    messages = []
    assert engine._inject_team_updates(messages)
    guidance = [m for m in messages if "No ejecutes entrenamiento" in m["content"]]
    assert len(guidance) == 1
    assert 'authority="USER_LITERAL"' in guidance[0]["content"]
    assert "URGENT" not in guidance[0]["content"]


@pytest.mark.parametrize("delegate", [False, True])
def test_normal_message_reaches_principal_without_mode_command(
    temp_db, workspace_dir, monkeypatch, delegate,
):
    from infinidev.config.settings import Settings, settings
    from infinidev.engine.orchestration.hooks import NoOpHooks
    from infinidev.engine.orchestration.pipeline import run_task
    from infinidev.engine.team.worker import TeamAgent

    # Load the ordinary configuration path, without a prior /engine selection.
    settings_file = workspace_dir / "settings.json"
    settings_file.write_text(json.dumps({"LLM_MODEL": "openai/gpt-6-astra"}))
    monkeypatch.setattr("infinidev.config.settings.SETTINGS_FILE", settings_file)
    loaded = Settings.load_user_settings()
    assert loaded.TASK_ENGINE_MODE == "task", (
        "the shipped default; the orchestrator is asserted below"
    )
    # The orchestrator is the one mode with no preliminary router: a message
    # the user addressed to the team must not be consumed by the chat agent.
    # Pinned explicitly because the default mode now routes through it first.
    monkeypatch.setattr(settings, "TASK_ENGINE_MODE", "orchestrator")
    monkeypatch.setattr(settings, "GATHER_ENABLED", False)
    monkeypatch.setattr(settings, "KEN_SESSION_ENABLED", False)
    monkeypatch.setattr("infinidev.engine.orchestration.pipeline._run_task_start_hook",
                        lambda **kwargs: "")
    monkeypatch.setattr("infinidev.engine.orchestration.pipeline._task_end_hook",
                        lambda *args, **kwargs: "")

    def unexpected_router(*args, **kwargs):
        pytest.fail("A preliminary router must not consume the principal's user message")

    monkeypatch.setattr("infinidev.engine.orchestration.chat_agent.run_chat_agent", unexpected_router)
    observed = {}
    request = "Revisa el cache y comprueba los gradientes." if delegate else "Hola."

    def execute(engine, agent, prompt, **kwargs):
        runtime = engine._team_runtime
        assert request in prompt[0]
        assert "A normal task message activates this role" in " ".join(
            kwargs["identity_override"].split(),
        )
        assert not runtime.read()["tickets"]
        available = {tool.name: tool for tool in agent.tools}
        if delegate:
            runtime._runner = lambda *args: ("Inspected cache.py:42", "completed")
            ticket = json.loads(available["team_create_ticket"]._run(
                title="Check gradients", objective="Inspect the cache write path",
                acceptance=["Cite source and execution limits"], constraints=[], dependencies=[],
            ))
            available["team_delegate"]._run(
                ticket_id=ticket["id"], name="Auditor", tools=["read_file"],
                system_prompt="Inspect the cache write path and report evidence.",
            )
            _finish(runtime)
            available["team_review_ticket"]._run(
                ticket_id=ticket["id"], decision="accepted", reason="Checked cache.py:42",
            )
        observed["board"] = runtime.read()
        engine._last_status = "done"
        return "Revisado." if delegate else "Hola."

    monkeypatch.setattr(LoopEngine, "execute", execute)
    agent = TeamAgent("root-agent", "Lead", 1, str(workspace_dir), [ReadFileTool()])
    result = run_task(agent=agent, user_input=request, session_id="ordinary-message",
                      engine=LoopEngine(), reviewer=None, hooks=NoOpHooks())
    assert result == ("Revisado." if delegate else "Hola.")
    assert len(observed["board"]["tickets"]) == int(delegate)
    assert all(ticket["status"] == "accepted" for ticket in observed["board"]["tickets"])


def test_principal_role_survives_disabled_optional_working_guidance(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps({"team": {"team.orchestrator_guidance": False}}))
    identity = build_team_identity(
        orchestrator=True, configuration=EffectivePromptConfiguration.compile(path),
    )
    assert "A normal task message activates this role" in " ".join(identity.split())
    assert "## Working guidance" not in identity


def test_an_idle_subscription_never_waits_without_a_timeout() -> None:
    """A lost wakeup must not hang the turn forever.

    ``wait_for_events`` checks the event log and then waits. A worker that
    finishes between those two steps notifies nobody the waiter has registered
    for yet, so an unbounded wait never returns: the orchestrator sat idle with
    its worker already finished and the turn never came back.
    """
    from infinidev.engine.team.waiting import _IDLE_RECHECK_SECONDS, _wait_slice

    assert _wait_slice(None) == _IDLE_RECHECK_SECONDS
    assert _wait_slice(None) is not None, "None blocks until notified, which hangs"
    # A deadline is still honoured, and never exceeds the re-check interval.
    assert _wait_slice(1.5) == 1.5
    assert _wait_slice(600.0) == _IDLE_RECHECK_SECONDS
    assert _wait_slice(0.0) == 0.0
