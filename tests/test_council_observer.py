"""Regression coverage for inspectable council and agent transcripts."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from infinidev.cli import commands
from infinidev.config.settings import Settings, reload_all
from infinidev.engine.council import observer
from infinidev.flows.event_listeners import event_bus
from infinidev.ui.app import InfinidevApp
from infinidev.ui.dialogs.agents_browser import AgentsBrowserControl


def setup_function() -> None:
    observer.clear_councils()


def teardown_function() -> None:
    observer.clear_councils()


def test_observer_keeps_combined_and_per_agent_transcripts() -> None:
    council_id = observer.start_council(
        question="Cache or recompute?",
        members=[
            {"member_id": "builder", "persona": "pragmatist", "objective": "ship"},
            {"member_id": "critic", "persona": "skeptic", "objective": "stress test"},
        ],
        project_id=7,
    )

    observer.set_member_status(council_id, "builder", "running", project_id=7, round_num=1)
    observer.add_message(
        council_id, "builder", "Cache the stable part.",
        project_id=7, round_num=1, action="post",
    )
    observer.add_message(
        council_id, "critic", "Invalidation is the risk.",
        project_id=7, round_num=1, action="conclude",
    )
    observer.finish_council(council_id, "completed", project_id=7)

    state = observer.get_council(council_id)
    assert state is not None
    assert [message["member_id"] for message in state["messages"]] == [
        "builder", "critic",
    ]
    assert state["members"]["builder"]["messages"][0]["text"] == (
        "Cache the stable part."
    )
    assert state["members"]["critic"]["messages"][0]["text"] == (
        "Invalidation is the risk."
    )
    assert state["status"] == "completed"


def test_observer_emits_lifecycle_events_with_snapshots() -> None:
    received: list[tuple[str, str]] = []

    def capture(event_type, project_id, agent_id, data):
        if event_type.startswith("council_"):
            received.append((event_type, data["council"]["status"]))

    event_bus.subscribe(capture)
    try:
        council_id = observer.start_council(
            question="Q",
            members=[{"member_id": "a", "persona": "p", "objective": "o"}],
            project_id=1,
        )
        observer.add_message(
            council_id, "a", "answer", project_id=1, round_num=1, action="post",
        )
        observer.finish_council(council_id, "completed", project_id=1)
    finally:
        event_bus.unsubscribe(capture)

    assert received == [
        ("council_started", "running"),
        ("council_agent_message", "running"),
        ("council_finished", "completed"),
    ]


def test_message_event_keeps_atomic_running_snapshot_during_concurrent_finish() -> None:
    message_listener_started = threading.Event()
    release_message_listener = threading.Event()
    received: list[tuple[str, str]] = []

    def capture(event_type, project_id, agent_id, data):
        if event_type == "council_agent_message":
            message_listener_started.set()
            assert release_message_listener.wait(1)
            received.append((event_type, data["council"]["status"]))
        elif event_type == "council_finished":
            received.append((event_type, data["council"]["status"]))

    council_id = observer.start_council(
        question="Q",
        members=[{"member_id": "a", "persona": "p", "objective": "o"}],
        project_id=1,
    )
    event_bus.subscribe(capture)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            message_added = executor.submit(
                observer.add_message,
                council_id,
                "a",
                "answer",
                project_id=1,
                round_num=1,
                action="post",
            )
            assert message_listener_started.wait(1)
            observer.finish_council(council_id, "completed", project_id=1)
            release_message_listener.set()
            message_added.result(1)
    finally:
        release_message_listener.set()
        event_bus.unsubscribe(capture)

    assert ("council_agent_message", "running") in received
    assert ("council_finished", "completed") in received


def test_retention_uses_completion_order_and_preserves_active_councils(monkeypatch) -> None:
    monkeypatch.setattr(observer.settings, "COUNCIL_HISTORY_LIMIT", 1)
    first = observer.start_council(question="first", members=[], project_id=1)
    active = observer.start_council(question="active", members=[], project_id=1)
    last = observer.start_council(question="last", members=[], project_id=1)

    observer.finish_council(last, "completed", project_id=1)
    observer.finish_council(first, "completed", project_id=1)

    assert observer.get_council(last) is None
    assert observer.get_council(first) is not None
    assert observer.get_council(active) is not None


def test_read_boundaries_apply_lowered_runtime_retention_limit(monkeypatch) -> None:
    monkeypatch.setattr(observer.settings, "COUNCIL_HISTORY_LIMIT", None)
    first = observer.start_council(question="first", members=[], project_id=1)
    active = observer.start_council(question="active", members=[], project_id=1)
    second = observer.start_council(question="second", members=[], project_id=1)

    observer.finish_council(first, "completed", project_id=1)
    observer.finish_council(second, "completed", project_id=1)
    monkeypatch.setattr(observer.settings, "COUNCIL_HISTORY_LIMIT", 1)

    assert [state["id"] for state in observer.list_councils()] == [active, second]
    assert observer.get_council(first) is None
    assert observer.get_council(active) is not None
    assert observer.council_eviction_count() == 1


def test_runtime_zero_retention_preserves_active_councils(monkeypatch) -> None:
    monkeypatch.setattr(observer.settings, "COUNCIL_HISTORY_LIMIT", None)
    terminal = observer.start_council(question="terminal", members=[], project_id=1)
    active = observer.start_council(question="active", members=[], project_id=1)
    observer.finish_council(terminal, "completed", project_id=1)

    monkeypatch.setattr(observer.settings, "COUNCIL_HISTORY_LIMIT", 0)

    assert [state["id"] for state in observer.list_councils()] == [active]
    assert observer.get_council(terminal) is None
    assert observer.get_council(active) is not None


def test_runtime_unlimited_retention_does_not_reconcile_history(monkeypatch) -> None:
    monkeypatch.setattr(observer.settings, "COUNCIL_HISTORY_LIMIT", None)
    terminal_ids = [
        observer.start_council(question=str(index), members=[], project_id=1)
        for index in range(3)
    ]
    for council_id in terminal_ids:
        observer.finish_council(council_id, "completed", project_id=1)

    assert [state["id"] for state in observer.list_councils()] == list(
        reversed(terminal_ids)
    )
    assert observer.council_eviction_count() == 0


def test_settings_reload_reconciles_history_through_shared_observer(monkeypatch) -> None:
    original = observer.settings.model_dump()
    monkeypatch.setattr(observer.settings, "COUNCIL_HISTORY_LIMIT", None)
    first = observer.start_council(question="first", members=[], project_id=1)
    second = observer.start_council(question="second", members=[], project_id=1)
    observer.finish_council(first, "completed", project_id=1)
    observer.finish_council(second, "completed", project_id=1)
    reloaded = observer.settings.model_copy(update={"COUNCIL_HISTORY_LIMIT": 1})
    monkeypatch.setattr(Settings, "load_user_settings", classmethod(lambda cls: reloaded))

    try:
        reload_all()

        assert observer.settings.COUNCIL_HISTORY_LIMIT == 1
        assert [state["id"] for state in observer.list_councils()] == [second]
        assert observer.get_council(first) is None
    finally:
        for key, value in original.items():
            setattr(observer.settings, key, value)


def test_listing_prioritises_active_then_orders_terminal_by_completion_recency() -> None:
    first = observer.start_council(question="first", members=[], project_id=1)
    active = observer.start_council(question="active", members=[], project_id=1)
    last = observer.start_council(question="last", members=[], project_id=1)

    observer.finish_council(last, "completed", project_id=1)
    observer.finish_council(first, "completed", project_id=1)

    assert [state["id"] for state in observer.list_councils()] == [active, first, last]
    assert [
        state["id"] for state in observer.list_councils(include_messages=False)
    ] == [active, first, last]


def test_concurrent_completions_retain_last_finished_and_every_active(monkeypatch) -> None:
    history_limit = 3
    terminal_count = 8
    monkeypatch.setattr(observer.settings, "COUNCIL_HISTORY_LIMIT", history_limit)
    active_ids = {
        observer.start_council(question=f"active-{index}", members=[], project_id=1)
        for index in range(2)
    }
    terminal_ids = [
        observer.start_council(question=f"terminal-{index}", members=[], project_id=1)
        for index in range(terminal_count)
    ]
    barrier = threading.Barrier(terminal_count)
    finished_order: list[str] = []

    def capture(event_type, project_id, agent_id, data):
        if event_type == "council_finished":
            finished_order.append(data["council_id"])

    def finish(council_id: str) -> None:
        barrier.wait()
        observer.finish_council(council_id, "completed", project_id=1)

    event_bus.subscribe(capture)
    try:
        with ThreadPoolExecutor(max_workers=terminal_count) as executor:
            list(executor.map(finish, terminal_ids))
    finally:
        event_bus.unsubscribe(capture)

    retained = {council["id"] for council in observer.list_councils()}
    retained_terminal_ids = retained - active_ids

    assert len(finished_order) == terminal_count
    assert set(finished_order) == set(terminal_ids)
    assert active_ids <= retained
    assert len(retained_terminal_ids) == history_limit
    assert retained_terminal_ids <= set(terminal_ids)


def test_slow_listener_does_not_reorder_terminal_retention(monkeypatch) -> None:
    monkeypatch.setattr(observer.settings, "COUNCIL_HISTORY_LIMIT", 1)
    first = observer.start_council(question="first", members=[], project_id=1)
    second = observer.start_council(question="second", members=[], project_id=1)
    first_listener_started = threading.Event()
    release_first_listener = threading.Event()

    def delay_first(event_type, project_id, agent_id, data):
        if event_type == "council_finished" and data["council_id"] == first:
            first_listener_started.set()
            assert release_first_listener.wait(1)

    event_bus.subscribe(delay_first)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            first_finished = executor.submit(
                observer.finish_council, first, "completed", project_id=1,
            )
            assert first_listener_started.wait(1)
            observer.finish_council(second, "completed", project_id=1)

            assert observer.get_council(first) is not None
            assert observer.get_council(second) is not None
            release_first_listener.set()
            first_finished.result(1)
    finally:
        release_first_listener.set()
        event_bus.unsubscribe(delay_first)

    assert observer.get_council(first) is None
    assert observer.get_council(second) is not None


def test_terminal_retention_eviction_drains_delivered_entries_after_gap(monkeypatch) -> None:
    monkeypatch.setattr(observer.settings, "COUNCIL_HISTORY_LIMIT", 1)
    first = observer.start_council(question="first", members=[], project_id=1)
    second = observer.start_council(question="second", members=[], project_id=1)
    third = observer.start_council(question="third", members=[], project_id=1)
    first_listener_started = threading.Event()
    release_first_listener = threading.Event()

    def delay_first(event_type, project_id, agent_id, data):
        if event_type == "council_finished" and data["council_id"] == first:
            first_listener_started.set()
            assert release_first_listener.wait(1)

    event_bus.subscribe(delay_first)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            first_finished = executor.submit(
                observer.finish_council, first, "completed", project_id=1,
            )
            assert first_listener_started.wait(1)
            observer.finish_council(second, "completed", project_id=1)
            observer.finish_council(third, "completed", project_id=1)

            assert all(
                observer.get_council(council_id) is not None
                for council_id in (first, second, third)
            )
            release_first_listener.set()
            first_finished.result(1)
    finally:
        release_first_listener.set()
        event_bus.unsubscribe(delay_first)

    assert observer.get_council(first) is None
    assert observer.get_council(second) is None
    assert observer.get_council(third) is not None
    assert list(observer._terminal_order) == [third]
    assert observer._delivered_terminal_events == {third}


def test_clear_during_terminal_delivery_does_not_restore_bookkeeping(monkeypatch) -> None:
    monkeypatch.setattr(observer.settings, "COUNCIL_HISTORY_LIMIT", 0)
    listener_started = threading.Event()
    release_listener = threading.Event()

    def delay_finish(event_type, project_id, agent_id, data):
        if event_type == "council_finished":
            listener_started.set()
            assert release_listener.wait(1)

    event_bus.subscribe(delay_finish)
    try:
        council_id = observer.start_council(question="Q", members=[], project_id=1)
        with ThreadPoolExecutor(max_workers=1) as executor:
            finished = executor.submit(
                observer.finish_council, council_id, "completed", project_id=1,
            )
            assert listener_started.wait(1)
            observer.clear_councils()
            release_listener.set()
            finished.result(1)
    finally:
        release_listener.set()
        event_bus.unsubscribe(delay_finish)

    assert observer.list_councils() == []
    assert list(observer._terminal_order) == []
    assert observer._delivered_terminal_events == set()


def test_terminal_listener_can_inspect_state_from_another_thread(monkeypatch) -> None:
    monkeypatch.setattr(observer.settings, "COUNCIL_HISTORY_LIMIT", 0)
    inspected: list[dict | None] = []

    def capture(event_type, project_id, agent_id, data):
        if event_type != "council_finished":
            return
        with ThreadPoolExecutor(max_workers=1) as executor:
            inspected.append(executor.submit(observer.get_council, data["council_id"]).result(1))

    event_bus.subscribe(capture)
    try:
        council_id = observer.start_council(question="Q", members=[], project_id=1)
        observer.finish_council(council_id, "completed", project_id=1)
    finally:
        event_bus.unsubscribe(capture)

    assert inspected[0] is not None
    assert inspected[0]["status"] == "completed"
    assert observer.get_council(council_id) is None


def test_terminal_council_ignores_late_member_updates_during_delivery(monkeypatch) -> None:
    monkeypatch.setattr(observer.settings, "COUNCIL_HISTORY_LIMIT", 1)
    listener_started = threading.Event()
    release_listener = threading.Event()
    received: list[str] = []

    def delay_finish(event_type, project_id, agent_id, data):
        if event_type == "council_finished":
            listener_started.set()
            assert release_listener.wait(1)
        elif event_type.startswith("council_agent_"):
            received.append(event_type)

    event_bus.subscribe(delay_finish)
    try:
        council_id = observer.start_council(
            question="Q",
            members=[{"member_id": "a", "persona": "p", "objective": "o"}],
            project_id=1,
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            finished = executor.submit(
                observer.finish_council, council_id, "completed", project_id=1,
            )
            assert listener_started.wait(1)
            observer.set_member_status(council_id, "a", "running", project_id=1)
            observer.add_message(
                council_id, "a", "late", project_id=1, round_num=2, action="post",
            )
            release_listener.set()
            finished.result(1)
    finally:
        release_listener.set()
        event_bus.unsubscribe(delay_finish)

    state = observer.get_council(council_id)
    assert state is not None
    assert state["status"] == "completed"
    assert state["members"]["a"]["status"] == "completed"
    assert state["messages"] == []
    assert received == []


def test_zero_retention_emits_once_before_eviction(monkeypatch) -> None:
    monkeypatch.setattr(observer.settings, "COUNCIL_HISTORY_LIMIT", 0)
    received: list[str] = []

    def capture(event_type, project_id, agent_id, data):
        if event_type == "council_finished":
            received.append(data["council"]["status"])

    event_bus.subscribe(capture)
    try:
        council_id = observer.start_council(question="Q", members=[], project_id=1)
        observer.finish_council(council_id, "completed", project_id=1)
        observer.finish_council(council_id, "completed", project_id=1)
    finally:
        event_bus.unsubscribe(capture)

    assert received == ["completed"]
    assert observer.get_council(council_id) is None
    assert observer.council_eviction_count() == 1


def _agents_browser_text() -> str:
    control = AgentsBrowserControl(lambda council_id, member_id: None)
    fragments = AgentsBrowserControl._fragments(control)
    return "".join(text for _, text in fragments)


def test_agents_browser_distinguishes_never_run_from_fully_evicted(monkeypatch) -> None:
    monkeypatch.setattr(observer, "list_councils", lambda **kwargs: [])
    monkeypatch.setattr(observer, "council_eviction_count", lambda: 0)

    assert "No councils have run in this process." in _agents_browser_text()

    monkeypatch.setattr(observer, "council_eviction_count", lambda: 2)

    text = _agents_browser_text()
    assert "No recent councils are retained." in text
    assert "2 older council transcript(s) were evicted." in text
    assert "No councils have run" not in text


def test_agents_browser_reports_evictions_with_retained_councils(monkeypatch) -> None:
    requested: list[bool] = []

    def list_councils(*, include_messages: bool = True):
        requested.append(include_messages)
        return [{
            "id": "council-recent",
            "status": "completed",
            "question": "Ship it?",
            "members": {},
        }]

    monkeypatch.setattr(observer, "list_councils", list_councils)
    monkeypatch.setattr(observer, "council_eviction_count", lambda: 3)

    text = _agents_browser_text()

    assert requested == [False]
    assert "council-recent" in text
    assert "3 older council transcript(s) were evicted." in text


def test_agents_listing_reports_evicted_transcripts(monkeypatch, capsys) -> None:
    monkeypatch.setattr(observer, "list_councils", lambda **kwargs: [])
    monkeypatch.setattr(observer, "council_eviction_count", lambda: 2)

    commands._render_agents_classic([])

    assert "No recent councils are retained." in capsys.readouterr().out


def test_agents_listing_requests_lightweight_council_summaries(monkeypatch, capsys) -> None:
    requested: list[bool] = []

    def list_councils(*, include_messages: bool = True):
        requested.append(include_messages)
        return [{
            "id": "council-recent",
            "status": "completed",
            "question": "Ship it?",
            "members": {
                "builder": {
                    "member_id": "builder",
                    "persona": "pragmatist",
                    "status": "completed",
                },
            },
        }]

    monkeypatch.setattr(observer, "list_councils", list_councils)
    monkeypatch.setattr(observer, "council_eviction_count", lambda: 3)

    commands._render_agents_classic([])

    output = capsys.readouterr().out
    assert requested == [False]
    assert "3 older council transcript(s) were evicted." in output
    assert "council-recent" in output
    assert "builder · pragmatist" in output


def test_agents_transcript_lookup_does_not_list_or_summarise_councils(
    monkeypatch, capsys,
) -> None:
    full_state = {
        "id": "council-recent",
        "status": "completed",
        "question": "Ship it?",
        "members": {},
        "messages": [{
            "member_id": "builder",
            "persona": "pragmatist",
            "round": 1,
            "text": "Ship it.",
        }],
    }

    def fail_list(*, include_messages: bool = True):
        raise AssertionError("transcript lookup must not list every council")

    monkeypatch.setattr(observer, "list_councils", fail_list)
    monkeypatch.setattr(observer, "get_council", lambda council_id: full_state)

    commands._render_agents_classic(["council-recent"])

    output = capsys.readouterr().out
    assert "council-recent debate" in output
    assert "Ship it." in output


def test_tui_transcript_open_handles_council_evicted_after_selection(monkeypatch) -> None:
    monkeypatch.setattr(observer, "get_council", lambda council_id: None)
    flashed: list[str] = []
    app = object.__new__(InfinidevApp)
    app._agent_tab_names = {}
    app.active_tab = "chat"
    app.active_dialog = "agents"
    app.flash_status = flashed.append

    app.open_agent_tab("council-evicted")

    assert flashed == ["Council is no longer available"]
    assert app._agent_tab_names == {}
    assert app.active_tab == "chat"
    assert app.active_dialog == "agents"


def test_tui_transcript_assigns_stable_role_headers_and_colours() -> None:
    council_id = observer.start_council(
        question="Q",
        members=[
            {"member_id": "a", "persona": "architect", "objective": "design"},
            {"member_id": "b", "persona": "reviewer", "objective": "review"},
        ],
        project_id=1,
    )
    observer.add_message(
        council_id, "a", "proposal", project_id=1, round_num=1, action="post",
    )
    observer.add_message(
        council_id, "b", "critique", project_id=1, round_num=1, action="post",
    )
    state = observer.get_council(council_id)
    assert state is not None

    app = object.__new__(InfinidevApp)
    messages = app._build_agent_transcript(state, None)

    assert messages[1]["sender"] == "a · architect"
    assert messages[2]["sender"] == "b · reviewer"
    assert messages[1]["show_sender"] is True
    assert messages[1]["sender_style"] != messages[2]["sender_style"]


def test_summary_and_running_count_do_not_copy_transcripts() -> None:
    council_id = observer.start_council(
        question="Q",
        members=[{"member_id": "a", "persona": "p", "objective": "o"}],
        project_id=1,
    )
    observer.set_member_status(council_id, "a", "running", project_id=1)
    observer.add_message(
        council_id, "a", "large" * 10_000,
        project_id=1, round_num=1, action="post",
    )
    observer.set_member_status(council_id, "a", "running", project_id=1)

    summaries = observer.list_councils(include_messages=False)

    assert observer.running_agent_count() == 1
    assert "messages" not in summaries[0]
    assert "messages" not in summaries[0]["members"]["a"]
    assert summaries[0]["message_count"] == 1
