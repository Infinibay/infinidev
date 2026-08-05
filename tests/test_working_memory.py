"""Working memory: what leaves the prompt must remain retrievable."""

from __future__ import annotations

import json

import pytest

from infinidev.engine.working_memory import (
    MAX_NOTE_GENERATION,
    TraceableNoteError,
    TraceableNoteEnvelope,
    WorkingMemory,
    create_traceable_note,
    get_working_memory,
    reset_working_memory,
)


@pytest.fixture
def memory(tmp_path, monkeypatch):
    """A WorkingMemory backed by a throwaway database."""
    from infinidev.code_intel import _db as ci_db
    from infinidev.config import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "DB_PATH", str(tmp_path / "wm.db"))
    ci_db._conn_cache.__dict__.clear()
    reset_working_memory()
    yield WorkingMemory("session-1", embed=False)
    reset_working_memory()


def _step_messages() -> list[dict]:
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"file_path": "src/auth/jwt.py"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": (
                "def verify_token(token: str) -> Claims:\n"
                "    # RS256, 60 second clock skew tolerance\n"
                "    return jwt.decode(token, PUBLIC_KEY, algorithms=['RS256'])\n"
            ),
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_2",
                    "function": {
                        "name": "execute_command",
                        "arguments": '{"command": "pytest tests/test_auth.py"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "content": (
                "FAILED tests/test_auth.py::test_expiry - AssertionError: "
                "expected 401 but the endpoint returned 500 for an expired token"
            ),
        },
    ]


# ── archiving ─────────────────────────────────────────────────────────────


def test_step_output_is_archived_with_its_call(memory):
    stored = memory.archive_step(1, _step_messages(), summary="")
    assert len(stored) == 2
    # The titles come back so the caller can render them as recall queries.
    assert any("read_file" in title for title in stored)
    records = memory.search("verify token", limit=5)
    titles = [record.title for record in records]
    assert any("read_file" in title and "jwt.py" in title for title in titles)


def test_trivial_tool_output_is_not_archived(memory):
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "c", "function": {"name": "x", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c", "content": "OK"},
    ]
    assert memory.archive_step(1, messages, summary="") == []


def test_identical_output_is_stored_once(memory):
    memory.archive_step(1, _step_messages(), summary="")
    assert memory.archive_step(2, _step_messages(), summary="") == []


def test_step_summary_is_archived_when_substantial(memory):
    summary = (
        "Read the JWT verifier and ran the auth suite; the expiry test fails "
        "because the endpoint returns 500 instead of 401."
    )
    memory.archive_step(1, [], summary=summary)
    records = memory.search("expiry test failure", limit=3)
    assert any(record.kind == "step_summary" for record in records)


# ── recall ────────────────────────────────────────────────────────────────


def test_recall_finds_the_evicted_error_message(memory):
    memory.archive_step(1, _step_messages(), summary="")
    records = memory.search("expired token returned 500", limit=3)
    assert records, "the archived failure must be retrievable"
    assert "expected 401" in records[0].content


def test_recall_is_scoped_to_the_session_by_default(memory, tmp_path):
    memory.archive_step(1, _step_messages(), summary="")
    other = WorkingMemory("session-2", embed=False)
    assert other.search("expired token") == []
    assert other.search("expired token", all_sessions=True)


def test_recall_matches_by_meaning_not_shared_words(tmp_path, monkeypatch):
    """The whole point of the archive: recall without knowing the wording.

    Also guards the persistence contract — ``execute_with_retry`` does not
    commit, so a missing ``conn.commit()`` leaves rows invisible to the
    embedding worker's connection and silently degrades this to keyword
    scoring, which cannot match these queries at all.
    """
    from infinidev.code_intel import _db as ci_db
    from infinidev.config import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "DB_PATH", str(tmp_path / "sem.db"))
    ci_db._conn_cache.__dict__.clear()
    reset_working_memory()

    memory = WorkingMemory("semantic-session")  # embeddings ON
    memory.archive_step(1, _step_messages(), summary="")

    records = memory.search("why is the endpoint answering with a server error")
    assert records, "semantic recall must find the failure without shared words"
    assert "expected 401" in records[0].content
    # Keyword scoring cannot produce this: the query shares no token with
    # the archived text, so any score at all proves the vector path ran.
    assert records[0].score > 0


def test_recall_returns_nothing_for_unrelated_queries(memory):
    memory.archive_step(1, _step_messages(), summary="")
    assert memory.search("kubernetes ingress annotations") == []


def test_render_truncates_but_reports_the_remainder(memory):
    memory.remember("big", "x" * 5000)
    record = memory.search("big", limit=1)[0]
    rendered = record.render(max_chars=100)
    assert "more chars" in rendered
    assert "source=working-memory" in rendered
    assert "authority=advisory" in rendered
    assert len(rendered) < 400


# ── lifecycle ─────────────────────────────────────────────────────────────


def test_stats_report_what_was_offloaded(memory):
    memory.archive_step(1, _step_messages(), summary="")
    stats = memory.stats()
    assert stats["entries"] == 2
    assert stats["archived_this_run"] == 2
    assert stats["approx_tokens_offloaded"] > 0


def test_clear_removes_only_this_session(memory):
    memory.archive_step(1, _step_messages(), summary="")
    other = WorkingMemory("session-2", embed=False)
    other.remember("kept", "this entry belongs to another session entirely")
    assert memory.clear() == 2
    assert memory.search("verify token") == []
    assert other.search("another session")


def test_registry_returns_one_instance_per_session(memory):
    first = get_working_memory("abc")
    assert get_working_memory("abc") is first
    assert get_working_memory("def") is not first


def test_archiving_never_raises_when_storage_fails(monkeypatch, memory):
    def boom(*_args, **_kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(
        "infinidev.engine.working_memory.execute_with_retry", boom, raising=True
    )
    assert memory.archive_step(1, _step_messages(), summary="") == []
    assert memory.search("anything") == []


def test_failed_insert_can_be_retried(monkeypatch, memory):
    """The in-process hash cache must only change after a confirmed insert."""
    from infinidev.engine import working_memory as working_memory_mod

    real_execute = working_memory_mod.execute_with_retry
    calls = 0

    def fail_once(fn, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient write failure")
        return real_execute(fn, *args, **kwargs)

    monkeypatch.setattr(working_memory_mod, "execute_with_retry", fail_once)
    content = "retryable context that is long enough to be worth archiving"
    assert memory.remember("retry", content) is False
    assert memory.remember("retry", content) is True
    assert memory.stats()["entries"] == 1


# ── traceable notes ───────────────────────────────────────────────────────


def test_equal_note_text_keeps_distinct_occurrences_and_provenance(memory):
    first = create_traceable_note(
        "auto_note", "same derived fact", source_artifact_id=11,
        step_index=2, tool_call_id="call-a", occurrence_id="occurrence-a",
    )
    second = create_traceable_note(
        "auto_note", "same derived fact", source_artifact_id=12,
        step_index=2, tool_call_id="call-b", occurrence_id="occurrence-b",
    )

    assert memory.remember_traceable(first)
    assert memory.remember_traceable(second)
    loaded = memory.load_traceable_notes(kinds=("auto_note",))
    assert [note.occurrence_id for note in loaded] == ["occurrence-a", "occurrence-b"]
    assert [note.source_artifact_id for note in loaded] == [11, 12]
    assert [note.tool_call_id for note in loaded] == ["call-a", "call-b"]


def test_traceable_notes_survive_registry_restart(tmp_path, monkeypatch):
    from infinidev.code_intel import _db as ci_db
    from infinidev.config import settings as settings_mod

    db_path = str(tmp_path / "restart.db")
    monkeypatch.setattr(settings_mod.settings, "DB_PATH", db_path)
    ci_db._conn_cache.__dict__.clear()
    reset_working_memory()
    original = WorkingMemory("trace-session", embed=False)
    note = create_traceable_note(
        "auto_note", "fact retained after a process-local reset",
        source_artifact_id=33, step_index=4, tool_call_id="call-restart",
        occurrence_id="restart-occurrence",
    )
    assert original.remember_traceable(note)

    reset_working_memory()
    reopened = WorkingMemory("trace-session", embed=False)
    assert reopened.load_traceable_notes() == [note]


def test_compaction_is_ordered_immutable_and_idempotent(memory):
    first = create_traceable_note(
        "auto_note", "first fact", source_artifact_id=21,
        step_index=1, tool_call_id="call-1", occurrence_id="source-1",
    )
    second = create_traceable_note(
        "auto_note", "second fact", source_artifact_id=22,
        step_index=3, tool_call_id="call-2", occurrence_id="source-2",
    )
    assert memory.remember_traceable(first)
    assert memory.remember_traceable(second)

    compacted = memory.compact_traceable_notes(
        [first, second], "first and second imply a stable conclusion",
        step_index=4, tool_call_id="analysis-call",
    )
    repeated = memory.compact_traceable_notes(
        [first, second], "retry produced different prose that must not fork identity",
        step_index=99, tool_call_id="retry-call",
    )

    assert repeated == compacted
    assert compacted.parent_ids == ("source-1", "source-2")
    assert [citation.occurrence_id for citation in compacted.citations] == [
        "source-1", "source-2",
    ]
    loaded = memory.load_traceable_notes()
    assert loaded[:2] == [first, second], "source notes must remain unchanged and ordered"
    assert loaded[2:] == [compacted], "repeating a compaction must not create another row"


def test_reversing_sources_changes_compaction_identity_and_order(memory):
    first = create_traceable_note(
        "auto_note", "first", step_index=1, occurrence_id="ordered-1",
    )
    second = create_traceable_note(
        "auto_note", "second", step_index=1, occurrence_id="ordered-2",
    )
    forward = memory.compact_traceable_notes([first, second], "combined")
    reverse = memory.compact_traceable_notes([second, first], "combined")
    assert forward.occurrence_id != reverse.occurrence_id
    assert reverse.parent_ids == ("ordered-2", "ordered-1")
    assert [item.occurrence_id for item in reverse.citations] == [
        "ordered-2", "ordered-1",
    ]


def test_compaction_limits_generation_without_deleting_sources(memory):
    source = create_traceable_note(
        "auto_note", "base", step_index=1, occurrence_id="generation-0",
    )
    chain = [source]
    current = source
    for generation in range(1, MAX_NOTE_GENERATION + 1):
        sibling = create_traceable_note(
            "auto_note", f"sibling {generation}", step_index=generation,
            occurrence_id=f"generation-source-{generation}",
        )
        current = memory.compact_traceable_notes(
            [current, sibling], f"generation {generation} summary"
        )
        chain.append(current)
        assert current.generation == generation

    with pytest.raises(TraceableNoteError, match="generation"):
        memory.compact_traceable_notes([current], "one generation too far")
    assert all(note.summary for note in chain)


def test_traceable_note_json_round_trip_is_versioned_and_validated():
    note = create_traceable_note(
        "artifact_analysis", "safe summary", step_index=2,
        occurrence_id="analysis-source",
    )
    restored = TraceableNoteEnvelope.from_json(note.to_json())
    assert restored == note
    payload = restored.to_dict()
    assert payload["version"] == 2
    assert payload["claim"]["text"] == "safe summary"
    assert payload["claim"]["classification"] == "observation"
    assert payload["claim"]["evidence"]
    assert payload["claim"]["provenance"]["occurrence_id"] == "analysis-source"
    assert payload["claim"]["still_valid"] is None
    with pytest.raises(TraceableNoteError, match="unsupported traceable note version"):
        TraceableNoteEnvelope.from_json(note.to_json().replace('"version":2', '"version":99'))


def test_version_one_traceable_notes_remain_readable():
    note = create_traceable_note(
        "auto_note", "legacy fact", step_index=1, occurrence_id="legacy-source",
    )
    payload = note.to_dict()
    payload["version"] = 1
    payload.pop("claim")

    restored = TraceableNoteEnvelope.from_json(json.dumps(payload))

    assert restored.summary == "legacy fact"
    assert restored.claim_classification == "observation"
    assert restored.confidence is None


# ── prompt retention policy ───────────────────────────────────────────────


def test_old_summaries_collapse_and_point_at_recall(monkeypatch):
    from infinidev.config import settings as settings_mod
    from infinidev.engine.loop.context import build_iteration_prompt
    from infinidev.engine.loop.models import ActionRecord, LoopState

    monkeypatch.setattr(settings_mod.settings, "WORKING_MEMORY_VERBATIM_STEPS", 2)
    state = LoopState()
    for index in range(1, 6):
        state.history.append(
            ActionRecord(
                step_index=index,
                summary=f"did thing {index}",
                changes_made=f"touched file{index}.py",
            )
        )
    prompt = build_iteration_prompt(
        description="t", expected_output="e", state=state, small_model=False
    )
    assert "3 earlier step(s) shown as one line each" in prompt
    assert "recall_context" in prompt
    # Collapsed entries lose their detail lines; recent ones keep them.
    assert "touched file1.py" not in prompt
    assert "touched file5.py" in prompt


def test_verbatim_budget_zero_keeps_every_summary(monkeypatch):
    from infinidev.config import settings as settings_mod
    from infinidev.engine.loop.context import build_iteration_prompt
    from infinidev.engine.loop.models import ActionRecord, LoopState

    monkeypatch.setattr(settings_mod.settings, "WORKING_MEMORY_VERBATIM_STEPS", 0)
    state = LoopState()
    for index in range(1, 6):
        state.history.append(
            ActionRecord(
                step_index=index,
                summary=f"did thing {index}",
                changes_made=f"touched file{index}.py",
            )
        )
    prompt = build_iteration_prompt(
        description="t", expected_output="e", state=state, small_model=False
    )
    assert "touched file1.py" in prompt
    assert "shown as one line each" not in prompt
