"""Regression coverage for inspectable council and agent transcripts."""

from __future__ import annotations

from infinidev.engine.council import observer
from infinidev.flows.event_listeners import event_bus
from infinidev.ui.app import InfinidevApp


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
