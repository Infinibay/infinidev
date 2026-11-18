"""Exact historical bytes remain recoverable independently of display excerpts."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from infinidev.tools.base.db import execute_with_retry
from infinidev.tools.meta.recall_context_tool import RecallContextInput, RecallContextTool
from tests.test_working_memory import memory
from tests.test_turn_runtime import RecordTool, reply, runtime, tc


def read(memory, reference, **kwargs):
    return json.loads(memory.read_archive(reference, **kwargs))


def manifest(memory, reference):
    rows = execute_with_retry(lambda conn: conn.execute(
        "SELECT id, content FROM working_memory WHERE session_id=? AND kind=?",
        (memory.session_id, "tool_archive_manifest")).fetchall(), db_path=memory._db_path)
    return next((row[0], json.loads(row[1])) for row in rows
                if json.loads(row[1])["reference"] == reference)


def test_unicode_pages_reconstruct_literal_artifact_without_search(memory, monkeypatch):
    text = ("\U0001f600e\u0301\r\n\x00\"\\" * 2200) + "END"
    reference = memory.archive_output("same visible title", text)
    monkeypatch.setattr(memory, "search", Mock(side_effect=AssertionError("semantic search")))
    offset, pieces = 0, []
    while True:
        page = read(memory, reference, start=offset, length=1703)
        assert page["status"] == "ok"
        assert page["archive_complete"] and page["range_complete"] and page["fully_returned"]
        assert page["offset_unit"] == "unicode_codepoint"
        assert page["sha256"] == hashlib.sha256(text.encode()).hexdigest()
        assert page["total_characters"] == len(text)
        assert page["total_utf8_bytes"] == len(text.encode())
        assert page["start"] == offset and len(page["content"]) == page["end"] - offset
        pieces.append(page["content"])
        if page["next_start"] is None:
            break
        assert page["next_start"] > offset
        offset = page["next_start"]
    assert "".join(pieces) == text
    assert page["artifact_fully_returned"] is False


def test_same_title_and_duplicate_content_keep_distinct_occurrences(memory):
    first = memory.archive_output("same", "A" * 8000 + "B" * 8000 + "A" * 8000)
    second = memory.archive_output("same", "different")
    assert first != second
    _, data = manifest(memory, first)
    assert data["parts"][0]["record_id"] == data["parts"][2]["record_id"]
    assert [part["start"] for part in data["parts"]] == [0, 8000, 16000]
    assert read(memory, first, start=16000, length=8000)["content"] == "A" * 8000
    page = read(memory, second)
    assert page["content"] == "different" and page["artifact_fully_returned"]


@pytest.mark.parametrize("text,start,length", [("", 0, 1500), ("abc", 3, 4), ("abc", 0, 0)])
def test_empty_eof_and_metadata_only_ranges(memory, text, start, length):
    reference = memory.archive_output("empty or metadata", text)
    page = read(memory, reference, start=start, length=length)
    assert page["status"] == "ok" and page["content"] == ""
    assert page["archive_complete"] and page["range_complete"] and page["fully_returned"]
    assert page["end"] == start


@pytest.mark.parametrize("field,value", [("start", -1), ("start", True), ("start", 1.0),
                                         ("length", -1), ("length", True), ("length", 1.5),
                                         ("length", 8001)])
def test_reference_schema_rejects_invalid_ranges(field, value):
    with pytest.raises(ValidationError):
        RecallContextInput.model_validate({"query": "", "reference": "wm-archive:" + "a" * 32,
                                          field: value})


def test_start_beyond_eof_is_not_silently_clamped(memory):
    reference = memory.archive_output("short", "abc")
    result = read(memory, reference, start=4)
    assert result["status"] == "range_error"
    assert not result["fully_returned"] and "content" not in result


@pytest.mark.parametrize("change", ["missing", "content", "occurrence", "hash"])
def test_missing_or_tampered_part_never_returns_partial_text(memory, change):
    reference = memory.archive_output("protected", "A" * 8000 + "B" * 8000)
    row_id, data = manifest(memory, reference)
    def alter(conn):
        if change == "missing":
            conn.execute("DELETE FROM working_memory WHERE id=?", (data["parts"][1]["record_id"],))
        elif change == "content":
            conn.execute("UPDATE working_memory SET content='corrupt' WHERE id=?",
                         (data["parts"][1]["record_id"],))
        else:
            if change == "occurrence":
                data["parts"][1] = data["parts"][0]
            else:
                data["sha256"] = "0" * 64
            body = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            conn.execute("UPDATE working_memory SET content=?,content_hash=? WHERE id=?",
                         (body, hashlib.sha256(body.encode()).hexdigest(), row_id))
        conn.commit()
    execute_with_retry(alter, db_path=memory._db_path)
    result = read(memory, reference)
    assert not result["archive_complete"] and not result["fully_returned"]
    assert "content" not in result


def test_transaction_rolls_back_new_parts_but_preserves_prior_dedupe(memory):
    memory.remember("already stored", "A" * 8000)
    def install(conn):
        conn.execute("""CREATE TRIGGER fail_manifest BEFORE INSERT ON working_memory
            WHEN NEW.kind='tool_archive_manifest'
            BEGIN SELECT RAISE(ABORT,'synthetic manifest failure'); END""")
        conn.commit()
    execute_with_retry(install, db_path=memory._db_path)
    assert memory.archive_output("fails", "A" * 8000 + "B" * 8000 + "C" * 8000) is None
    count = execute_with_retry(lambda conn: conn.execute(
        "SELECT COUNT(*) FROM working_memory").fetchone()[0], db_path=memory._db_path)
    assert count == 1
    def remove(conn):
        conn.execute("DROP TRIGGER fail_manifest")
        conn.commit()
    execute_with_retry(remove, db_path=memory._db_path)
    reference = memory.archive_output("retry", "A" * 8000 + "B" * 8000 + "C" * 8000)
    assert read(memory, reference)["archive_complete"]


def test_public_tool_exact_mode_and_legacy_record_scope(memory, bound_tool, monkeypatch):
    import infinidev.engine.working_memory as wm

    text = "prefix" * 500 + "TAIL-ONLY-EXACT"
    memory.remember("legacy title", text)
    row = memory.search("legacy title")[0]
    monkeypatch.setattr(wm, "get_working_memory", lambda session: memory)
    tool = bound_tool(RecallContextTool)
    legacy = tool._run(query="legacy title")
    assert "TAIL-ONLY-EXACT" not in legacy and "more chars" in legacy
    assert "wm-record:" + row.id in legacy
    monkeypatch.setattr(memory, "search", Mock(side_effect=AssertionError("search forbidden")))
    result = json.loads(tool._run(reference="wm-record:" + row.id, start=3000, length=50))
    assert result["content"] == text[3000:]
    assert result["scope"] == "memory_record" and result["archive_complete"] is None
    assert not result["artifact_fully_returned"]
    missing = json.loads(tool._run(reference="wm-archive:" + "0" * 32))
    assert missing["status"] == "not_found" and "content" not in missing


def test_cross_session_exact_reference_never_searches(memory, monkeypatch):
    reference = memory.archive_output("private session", "private historical data")
    other = type(memory)("other-session", embed=False, db_path=memory._db_path)
    monkeypatch.setattr(other, "search", Mock(side_effect=AssertionError("search forbidden")))
    result = read(other, reference)
    assert result["status"] == "not_found" and "content" not in result


def test_json_response_budget_never_silently_truncates_a_page(memory):
    reference = memory.archive_output("escaped", "\x00" * 8000)
    raw = memory.read_archive(reference, length=8000)
    result = json.loads(raw)
    assert len(raw.encode()) <= 32768
    assert result["status"] == "response_budget_exceeded"
    assert result["archive_complete"] and result["range_complete"]
    assert not result["fully_returned"] and "content" not in result
    assert read(memory, reference, length=1500)["content"] == "\x00" * 1500


def test_oversize_and_unencodable_data_fail_without_claiming_integrity(memory):
    assert memory.archive_output("large", "x" * (8000 * 128 + 1)) is None
    assert memory.archive_output("unicode", "bad-\ud800") is None


def test_runtime_issues_stable_refs_and_exact_recall_stays_complete_json(runtime):
    engine, agent, script, seen = runtime
    source = "historical literal text\r\n" * 500
    def recall():
        body = next(m["content"] for m in reversed(seen[-1][0]) if m["role"] == "tool")
        reference = body.split("recall_context reference: ", 1)[1].split("]", 1)[0]
        assert reference.startswith("wm-archive:")
        return reply("", tc("recall_context", reference=reference, start=3000, length=2000))
    script(reply("", tc("record", text=source)), recall,
           reply("", tc("finish_turn", message="done")))
    result = engine.run_turn(agent, "recover exact data",
                             task_tools=[RecordTool(), RecallContextTool()])
    assert result.status == "answered"
    recalled = next(m["content"] for m in reversed(seen[-1][0]) if m["role"] == "tool")
    page = json.loads(recalled)
    assert page["content"] == source[3000:5000]
    assert page["archive_complete"] and page["range_complete"] and page["fully_returned"]
    assert engine._last_turn_receipt["dispatches"] == 2


@pytest.mark.parametrize(("selection", "expected"), [
    ({}, '     1\tVALUE = """first \t\n     2\tsecond  \n     3\t"""'),
    ({"offset": 2, "limit": 1}, "     2\tsecond  "),
])
def test_numbered_read_and_exact_recall_preserve_source_whitespace(runtime, selection, expected):
    from pathlib import Path

    from infinidev.tools.file.read_file_tool import ReadFileTool

    engine, agent, script, seen = runtime
    (Path(agent.workspace_path) / "literal.py").write_bytes(
        b'VALUE = """first \t\r\nsecond  \r\n"""')

    def recall():
        body = next(m["content"] for m in seen[-1][0] if m["role"] == "tool")
        reference = body.split("recall_context reference: ", 1)[1].removesuffix("]")
        return reply("", tc("recall_context", reference=reference))

    script(reply("", tc("read_file", file_path="literal.py", **selection)), recall,
           reply("", tc("finish_turn", message="Recovered.")))
    result = engine.run_turn(agent, "Read and recover literal.py",
                             task_tools=[ReadFileTool(), RecallContextTool()])
    assert result.status == "answered"
    observed, recalled = [m["content"] for m in seen[-1][0] if m["role"] == "tool"]
    assert observed.rsplit("\n[Archived evidence; ", 1)[0] == expected
    assert json.loads(recalled)["content"] == expected


def test_large_file_preview_keeps_whitespace_in_visible_lines(bound_tool, workspace_dir):
    from infinidev.tools.file.read_file_tool import ReadFileTool

    path = workspace_dir / "sample.txt"
    path.write_text("head  \n" + "middle\n" * 800 + "tail\t \n")
    result = bound_tool(ReadFileTool).run(file_path="sample.txt")
    assert "head+tail preview" in result
    assert "     1\thead  \n" in result
    assert "   802\ttail\t \n" in result


@pytest.mark.parametrize("manual", [False, True])
def test_context_rejection_recovers_archived_read_without_dispatching_it_again(
        runtime, monkeypatch, manual):
    from pathlib import Path
    from types import SimpleNamespace

    from litellm.exceptions import ContextWindowExceededError

    from infinidev.tools.file.read_file_tool import ReadFileTool

    engine, agent, script, seen = runtime
    if manual:
        monkeypatch.setattr("infinidev.engine.turns.runtime.get_model_capabilities",
                            lambda: SimpleNamespace(supports_function_calling=False,
                                                    supports_vision=False))
    source = ["evidence " * 10] * 400 + ["tail from the observed version"]
    path = Path(agent.workspace_path) / "evidence.txt"
    path.write_text("\n".join(source))
    final_line = "   401\ttail from the observed version"
    prompt = "Recover the final line of the observed file."

    def action(name, **arguments):
        if manual:
            return reply(json.dumps({"tool_calls": [{"name": name, "arguments": arguments}]}))
        return reply("", tc(name, **arguments))

    def observation(request, name):
        if manual:
            prefix = f"[Tool: {name}] Result:\n"
            return next(m["content"].removeprefix(prefix) for m in request
                        if m["role"] == "user" and m["content"].startswith(prefix))
        return next(m["content"] for m in reversed(request) if m["role"] == "tool")

    def reject():
        assert final_line in observation(seen[-1][0], "read_file")
        path.write_text("different source after the read")
        return ContextWindowExceededError("context window exceeded", "scripted", "openai")

    def retrieve():
        original = seen[1][0]
        retry = seen[-1][0]
        result = json.loads(observation(retry, "read_file"))
        assert result["status"] == "archived_for_context"
        assert final_line not in result["excerpt"]
        assert retry[:3] == original[:3]
        assert [m["tool_call_id"] for m in retry if m["role"] == "tool"] == [
            m["tool_call_id"] for m in original if m["role"] == "tool"
        ]
        return action("recall_context", reference=result["reference"],
                      start=result["original_characters"] - len(final_line))

    script(action("read_file", file_path=path.name), reject, retrieve,
           action("finish_turn", message="Recovered the archived observation."))
    turn = engine.run_turn(agent, prompt, task_tools=[ReadFileTool(), RecallContextTool()])
    assert turn.status == "answered"
    assert turn.metrics["observed_iterations"] == 4
    assert engine._last_turn_receipt["dispatches"] == 2
    page = observation(seen[-1][0], "recall_context")
    assert json.loads(page)["content"] == final_line


@pytest.mark.parametrize(("recall_granted", "limit", "requests"), [
    (False, 0, 2),
    (True, 2, 2),
    (True, 0, 3),
])
def test_context_recovery_respects_grant_limit_and_one_retry(runtime, recall_granted, limit, requests):
    from pathlib import Path

    from litellm.exceptions import ContextWindowExceededError

    from infinidev.tools.file.read_file_tool import ReadFileTool

    engine, agent, script, seen = runtime
    path = Path(agent.workspace_path) / "oversized.txt"
    path.write_text("oversized evidence " * 2000)
    tools = [ReadFileTool()]
    if recall_granted:
        tools.append(RecallContextTool())
    rejection = ContextWindowExceededError("context window exceeded", "scripted", "openai")
    script(reply("", tc("read_file", file_path=path.name)), rejection, rejection)

    turn = engine.run_turn(agent, "Inspect file", task_tools=tools, max_iterations=limit)
    assert turn.status == "failed"
    assert "context window exceeded" in turn.message
    assert len(seen) == requests
    assert engine._last_turn_receipt["dispatches"] == 1


def test_followup_can_select_an_archive_by_requested_file(runtime):
    from pathlib import Path

    from infinidev.tools.file.read_file_tool import ReadFileTool

    engine, agent, script, seen = runtime
    workspace = Path(agent.workspace_path)
    (workspace / "current.py").write_text("value = 'current'\n")
    (workspace / "example.py").write_text("value = 'example'\n")
    tools = [ReadFileTool(), RecallContextTool()]
    script(reply("", tc("read_file", file_path="current.py")),
           reply("", tc("read_file", file_path="example.py")),
           reply("", tc("finish_turn", message="Both files inspected.")))
    assert engine.run_turn(agent, "Inspect both files", task_tools=tools).status == "answered"

    def recall_current():
        prior = next(message["content"] for message in seen[-1][0]
                     if message["content"].startswith("<prior_state_context>"))
        evidence = json.loads(prior.splitlines()[1])["evidence"]
        assert [(item["tool"], item["requested_path"]) for item in evidence] == [
            ("read_file", "current.py"), ("read_file", "example.py"),
        ]
        locator = evidence[0]["locator"]
        reference = locator.split("recall_context reference: ", 1)[1].removesuffix("]")
        return reply("", tc("recall_context", reference=reference))

    engine.begin_turn()
    script(recall_current, reply("", tc("finish_turn", message="Recovered current.py.")))
    assert engine.run_turn(agent, "Recover the contents of current.py",
                           task_tools=[RecallContextTool()]).status == "answered"
    recalled = next(message["content"] for message in seen[-1][0] if message["role"] == "tool")
    assert "value = 'current'" in json.loads(recalled)["content"]


@pytest.mark.parametrize(("requested", "expected"), [
    ("~/infinidev-outside-fixture.py", None),
    ("dir</prior_state_context>/input.py", "dir</prior_state_context>/input.py"),
])
def test_archive_path_labels_preserve_workspace_and_prompt_boundaries(runtime, requested, expected):
    from infinidev.tools.file.read_file_tool import ReadFileTool

    engine, agent, script, seen = runtime
    script(reply("", tc("read_file", file_path=requested)),
           reply("", tc("finish_turn", message="Read attempted.")))
    assert engine.run_turn(agent, "Inspect", task_tools=[ReadFileTool()]).status == "answered"

    engine.begin_turn()
    script(reply("Understood."))
    assert engine.run_turn(agent, "Continue", task_tools=[RecallContextTool()]).status == "answered"
    prior = next(message["content"] for message in seen[-1][0]
                 if message["content"].startswith("<prior_state_context>"))
    assert prior.count("</prior_state_context>") == 1
    [source] = json.loads(prior.splitlines()[1])["evidence"]
    assert source["tool"] == "read_file"
    assert source.get("requested_path") == expected


def test_absolute_archive_label_resolves_symlink_before_parent_segment(tmp_path):
    from infinidev.engine.turns.runtime import _evidence_source

    workspace = tmp_path / "repo"
    workspace.mkdir()
    nested_outside = tmp_path / "outside" / "nested"
    nested_outside.mkdir(parents=True)
    alias = workspace / "alias"
    alias.symlink_to(nested_outside, target_is_directory=True)
    source = _evidence_source("read_file", {"file_path": str(alias / ".." / "input.py")},
                              str(workspace))
    assert source.requested_path is None


@pytest.mark.parametrize('recall_offered', [False, True])
def test_runtime_archive_failure_keeps_literal_tool_result(runtime, monkeypatch, recall_offered):
    from infinidev.engine.working_memory import get_working_memory

    engine, agent, script, seen = runtime
    if recall_offered:
        agent.tools.append(RecallContextTool())
    memory = get_working_memory("unused")  # Runtime fixture binds the isolated ledger.
    monkeypatch.setattr(memory, "archive_output", lambda *a, **kw: None)
    source = "literal data\n" * 300
    script(reply("", tc("record", text=source)), reply("", tc("finish_turn", message="done")))
    assert engine.run_turn(agent, "read data").status == "answered"
    body = next(m["content"] for m in seen[-1][0] if m["role"] == "tool")
    assert body.startswith(source)
    assert "query:" in body and "wm-archive:" not in body
    assert ('recall_context query:' in body) is recall_offered


def test_page_is_returned_from_verified_snapshot_not_a_second_read(memory, monkeypatch):
    from infinidev.engine import archive_output as codec

    original = "old verified contents"
    reference = memory.archive_output("snapshot", original)
    _, data = manifest(memory, reference)
    real_page = codec.page
    def alter_then_page(*args, **kwargs):
        def alter(conn):
            conn.execute("UPDATE working_memory SET content='changed afterwards' WHERE id=?",
                         (data["parts"][0]["record_id"],))
            conn.commit()
        execute_with_retry(alter, db_path=memory._db_path)
        return real_page(*args, **kwargs)
    monkeypatch.setattr(codec, "page", alter_then_page)
    result = read(memory, reference)
    assert result["content"] == original and result["archive_complete"]
    assert not read(memory, reference)["archive_complete"]


def test_codec_import_does_not_bootstrap_engine_database_or_model():
    script = '''
import importlib.abc
import sys
class RejectHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("infinidev.config", "infinidev.db", "infinidev.tools",
                                "litellm", "numpy", "textual")):
            raise AssertionError("unexpected bootstrap: " + fullname)
sys.meta_path.insert(0, RejectHeavy())
from infinidev.engine.archive_output import split_output
assert split_output("literal") == ["literal"]
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                            text=True, timeout=10)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("reference", ["bad-ref", "wm-archive:../path", "wm-record:not-a-uuid"])
def test_malformed_reference_never_falls_back_to_search(memory, monkeypatch, reference):
    monkeypatch.setattr(memory, "search", Mock(side_effect=AssertionError("search forbidden")))
    result = read(memory, reference)
    assert result["status"] == "invalid_reference" and "content" not in result


def test_traceable_note_does_not_advertise_an_unreadable_uuid_reference(memory):
    from infinidev.engine.working_memory import create_traceable_note

    note = create_traceable_note("auto_note", "declared historical observation", step_index=1,
                                 occurrence_id="note:synthetic-stable-identity")
    assert memory.remember_traceable(note)
    row = memory.search("auto_note:" + note.occurrence_id)[0]
    assert row.id == note.occurrence_id
    assert "wm-record:" not in row.render()


def test_schema_still_requires_query_or_explicit_reference():
    with pytest.raises(ValidationError):
        RecallContextInput.model_validate({})
    assert RecallContextInput.model_validate({"query": ""}).query == ""


def test_manifest_metadata_is_hidden_from_legacy_exact_title_search(memory):
    reference = memory.archive_output("simple source", "unrelated historical evidence")
    assert memory.search(reference) == []
