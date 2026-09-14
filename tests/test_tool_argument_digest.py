"""Eliding the payload of a tool call whose Step already closed.

The measurement behind this: on a task that writes files the serialized
``tool_calls`` reach 33 % of the request payload, because a written file's body
*is* the argument, and every later request replays it. These tests pin the two
properties that make the elision safe to consider — the JSON shape survives, and
nothing is touched that a provider still has to answer.
"""

from __future__ import annotations

import json

from infinidev.engine.loop.tool_argument_digest import (
    _ELIDE_OVER_CHARS,
    digest_arguments,
    trim_superseded_tool_arguments,
)


def _call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def test_a_long_body_is_replaced_and_a_short_argument_is_left_alone() -> None:
    body = "line\n" * 400
    digested, saved = digest_arguments(json.dumps({"file_path": "a.py", "content": body}))
    parsed = json.loads(digested)

    assert saved > 0
    # The shape and the identity survive; only the payload is gone.
    assert parsed["file_path"] == "a.py"
    assert parsed["content"].startswith("<elided by infinidev:")
    assert str(len(body)) in parsed["content"]
    # The marker may only promise what the engine can actually deliver.
    assert "read the file" in parsed["content"]


def test_a_short_argument_is_returned_byte_identical() -> None:
    original = json.dumps({"file_path": "src/pkg/mod.py", "old_string": "v + 1"})

    digested, saved = digest_arguments(original)

    assert saved == 0
    assert digested is original


def test_nothing_is_elided_at_exactly_the_threshold() -> None:
    edge = "x" * _ELIDE_OVER_CHARS
    digested, saved = digest_arguments(json.dumps({"content": edge}))

    assert saved == 0
    assert json.loads(digested)["content"] == edge

    over = "x" * (_ELIDE_OVER_CHARS + 1)
    _, saved_over = digest_arguments(json.dumps({"content": over}))
    assert saved_over > 0


def test_a_malformed_call_is_never_rewritten() -> None:
    """A call that does not parse is a fact about the run, not a payload."""
    broken = '{"content": "unterminated'

    digested, saved = digest_arguments(broken)

    assert saved == 0
    assert digested is broken


def test_nested_payloads_are_reached() -> None:
    arguments = json.dumps(
        {
            "edits": [
                {"path": "a.py", "new_text": "y" * 900},
                {"path": "b.py", "new_text": "z" * 900},
            ],
            "dry_run": False,
        }
    )

    digested, saved = digest_arguments(arguments)
    parsed = json.loads(digested)

    assert saved > 1_000
    assert parsed["dry_run"] is False
    assert [item["path"] for item in parsed["edits"]] == ["a.py", "b.py"]
    assert all("<elided" in item["new_text"] for item in parsed["edits"])


def test_only_closed_turns_lose_their_arguments() -> None:
    """The newest call is the one being answered; it stays whole."""
    body = "w" * 5_000
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_call("c1", "create_file", {"file_path": "a.py", "content": body})],
        },
        {"role": "tool", "content": "created", "tool_call_id": "c1"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_call("c2", "create_file", {"file_path": "b.py", "content": body})],
        },
    ]

    saved = trim_superseded_tool_arguments(messages)

    assert saved > 4_000
    first = json.loads(messages[1]["tool_calls"][0]["function"]["arguments"])
    newest = json.loads(messages[3]["tool_calls"][0]["function"]["arguments"])
    assert "<elided" in first["content"]
    assert newest["content"] == body


def test_the_call_ids_and_names_survive_so_the_results_still_match() -> None:
    """A provider rejects a tool result whose call it was never told about."""
    body = "q" * 3_000
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_call("call_abc", "write_file", {"path": "x.py", "text": body})],
        },
        {"role": "tool", "content": "ok", "tool_call_id": "call_abc"},
        {"role": "assistant", "content": "", "tool_calls": [_call("call_def", "read_file", {"path": "x.py"})]},
    ]

    trim_superseded_tool_arguments(messages)

    call = messages[0]["tool_calls"][0]
    assert call["id"] == "call_abc"
    assert call["function"]["name"] == "write_file"
    assert call["type"] == "function"
    # And the result it answers is still in the transcript.
    assert messages[1]["tool_call_id"] == "call_abc"


def test_a_transcript_without_assistant_turns_is_untouched() -> None:
    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "go"}]

    assert trim_superseded_tool_arguments(messages) == 0


def test_a_transcript_whose_calls_are_all_small_reports_no_saving() -> None:
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [_call("c1", "read_file", {"path": "a.py"})]},
        {"role": "tool", "content": "x", "tool_call_id": "c1"},
        {"role": "assistant", "content": "", "tool_calls": [_call("c2", "read_file", {"path": "b.py"})]},
    ]

    assert trim_superseded_tool_arguments(messages) == 0


def test_the_marker_never_promises_a_recall_that_cannot_return_the_body() -> None:
    """The archive stores a call's *result*, not its arguments.

    ``WorkingMemory._extract`` pairs an assistant call with its ``role: tool``
    result and files the *result* as the record's content; the arguments are
    used only to build the title. So an elided body is not what ``recall_context``
    returns, and a marker that offered it would send the model after evidence
    that does not exist.
    """
    body = "z" * 2_000
    digested, _ = digest_arguments(json.dumps({"file_path": "a.py", "content": body}))
    marker = json.loads(digested)["content"]

    assert "recall_context" not in marker
    # What it does promise has to hold: the result is untouched, and a file body
    # is on disk.
    assert "tool result above is unchanged" in marker
    assert "read the file" in marker


def test_the_marker_names_the_size_so_re_reading_can_be_weighed() -> None:
    body = "z" * 4_321
    digested, _ = digest_arguments(json.dumps({"path": "a.py", "text": body}))

    assert "4321" in json.loads(digested)["text"]
