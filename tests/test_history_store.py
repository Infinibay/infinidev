"""Tests for the execution event store, redaction, and run digests."""

from __future__ import annotations

import pytest

from infinidev.engine.history import events as ev
from infinidev.engine.history import store
from infinidev.engine.history.digest import (
    digest_from_outcome,
    digest_from_staged_state,
    render_digest,
)
from infinidev.engine.history.redaction import redact_payload, redact_text


class TestRedaction:
    @pytest.mark.parametrize(
        "text",
        [
            "api_key=sk-abc123def456ghi789",
            "Authorization: Bearer abcdefgh12345678",
            "password=hunter2secret",
            "postgres://user:supersecret@db:5432/app",
            "token: ghp_" + "a" * 32,
        ],
    )
    def test_secret_shapes_are_masked(self, text):
        assert "[REDACTED]" in redact_text(text)

    def test_normal_text_is_untouched(self):
        text = "Add JWT middleware to the auth endpoints and run tests."
        assert redact_text(text) == text

    def test_redact_payload_recurses(self):
        payload = {
            "note": "using api_key=sk-abc123def456ghi789",
            "nested": [{"cmd": "password=hunter2secret"}],
            "count": 3,
        }
        cleaned = redact_payload(payload)
        assert "sk-abc123" not in str(cleaned)
        assert "hunter2" not in str(cleaned)
        assert cleaned["count"] == 3


class TestStoreRuns:
    def test_create_and_finish_run(self, temp_db):
        run_id = store.create_run(
            session_id="s1", engine="staged", mode="auto",
            goal_title="Add JWT", goal_request="Add JWT to all endpoints",
            selection={"engine": "staged"},
        )
        assert run_id.startswith("run_")
        run = store.get_run(run_id)
        assert run["status"] == store.RUN_RUNNING
        assert run["selection_json"]["engine"] == "staged"

        store.finish_run(run_id, "completed", digest={"status": "completed"},
                         metrics={"stages": 2})
        run = store.get_run(run_id)
        assert run["status"] == "completed"
        assert run["digest_json"] == {"status": "completed"}
        assert run["metrics_json"] == {"stages": 2}

    def test_goal_request_is_redacted_on_write(self, temp_db):
        run_id = store.create_run(
            session_id="s1", engine="react",
            goal_title="task", goal_request="use api_key=sk-abc123def456ghi789",
        )
        run = store.get_run(run_id)
        assert "sk-abc123" not in run["goal_request"]

    def test_latest_run_for_session(self, temp_db):
        first = store.create_run(session_id="s1", engine="staged")
        second = store.create_run(session_id="s1", engine="react")
        assert store.latest_run_for_session("s1")["run_id"] == second
        assert first != second


class TestStoreEvents:
    def test_sequence_is_monotonic_per_run(self, temp_db):
        run_id = store.create_run(session_id="s1", engine="staged")
        e1 = store.append_event(run_id, "s1", ev.RUN_STARTED, {})
        e2 = store.append_event(run_id, "s1", ev.ENGINE_SELECTED, {})
        events = store.list_run_events(run_id)
        assert [e["sequence"] for e in events] == [1, 2]
        assert [e["event_id"] for e in events] == [e1, e2]

    def test_search_full_text_and_filters(self, temp_db):
        run_id = store.create_run(session_id="s1", engine="staged")
        store.append_event(run_id, "s1", ev.TASK_CLOSED,
                           {"title": "Add JWT middleware"}, node_id="t1")
        store.append_event(run_id, "s1", ev.TASK_CLOSED,
                           {"title": "Write tests"}, node_id="t2")

        hits = store.search_events(query="JWT middleware", session_id="s1")
        assert len(hits) == 1
        assert hits[0]["node_id"] == "t1"

        by_type = store.search_events(event_type=ev.TASK_CLOSED, session_id="s1")
        assert len(by_type) == 2

    def test_archive_only_hidden_by_default(self, temp_db):
        run_id = store.create_run(session_id="s1", engine="staged")
        store.append_event(run_id, "s1", ev.DIGEST_CREATED, {"x": 1},
                           visibility=ev.VISIBILITY_ARCHIVE_ONLY)
        assert store.search_events(session_id="s1") == []
        assert len(store.search_events(session_id="s1", include_archive=True)) == 1
        assert store.list_run_events(run_id) == []
        assert len(store.list_run_events(run_id, include_archive=True)) == 1

    def test_trace_chain_walks_parents(self, temp_db):
        run_id = store.create_run(session_id="s1", engine="staged")
        e1 = store.append_event(run_id, "s1", ev.RUN_STARTED, {})
        e2 = store.append_event(run_id, "s1", ev.ENGINE_SELECTED, {},
                                parent_event_id=e1)
        e3 = store.append_event(run_id, "s1", ev.TASK_CLOSED, {},
                                parent_event_id=e2)
        chain = store.trace_chain(e3)
        assert [c["event_id"] for c in chain] == [e1, e2, e3]

    def test_read_events_with_window(self, temp_db):
        run_id = store.create_run(session_id="s1", engine="staged")
        store.append_event(run_id, "s1", ev.RUN_STARTED, {})
        store.append_event(run_id, "s1", ev.ENGINE_SELECTED, {})
        e3 = store.append_event(run_id, "s1", ev.TASK_CLOSED, {})
        rows = store.read_events([e3], window_before=1, window_after=1)
        assert [r["sequence"] for r in rows] == [2, 3]


class TestDigests:
    def _state(self):
        from infinidev.engine.analysis.staged_planning import (
            EvidenceEntry,
            GoalSpec,
            GoalTerminalState,
            StageSpec,
            StageTaskSpec,
            StagedPlanningState,
        )

        state = StagedPlanningState(
            goal=GoalSpec(title="Add JWT", user_request="Add JWT to all endpoints")
        )
        spec = StageSpec(
            title="Implement", outcome="JWT works", exit_criteria=["ok"],
            tasks=[StageTaskSpec(id="t1", title="Add middleware", outcome="mw",
                                 acceptance_criteria=["mw works"])],
        )
        record = state.add_stage(spec)
        record.tasks[0].status = "completed"
        record.tasks[0].result = "done"
        record.status = "completed"
        state.add_evidence(EvidenceEntry(kind="task_result", summary="middleware added"))
        state.status = "complete"
        state.terminal = GoalTerminalState(
            kind="goal_complete", summary="done", evidence=["e1"]
        )
        return state

    def test_digest_from_staged_state(self, temp_db):
        digest = digest_from_staged_state(
            self._state(), run_id="run_x", engine_name="staged", mode="staged",
            status="completed",
        )
        assert digest["goal"]["title"] == "Add JWT"
        assert digest["completed_work"] == ["Stage 1 / Add middleware"]
        assert digest["status"] == "completed"
        assert digest["references"]["run_id"] == "run_x"

    def test_digest_from_outcome(self):
        digest = digest_from_outcome(
            run_id="run_y", engine_name="react", mode="react", status="blocked",
            goal_title="Fix bug", user_request="Fix the login bug",
        )
        assert digest["open_work"]["blocked"] == ["Fix bug"]
        assert digest["next_steps"]

    def test_render_digest(self):
        digest = digest_from_outcome(
            run_id="run_y", engine_name="react", mode="react", status="blocked",
            goal_title="Fix bug", user_request="Fix the login bug",
        )
        text = render_digest(digest)
        assert "Fix bug" in text
        assert "blocked" in text
