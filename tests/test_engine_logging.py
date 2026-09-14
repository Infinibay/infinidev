"""Terminal run summaries must reflect the actual terminal state."""

from __future__ import annotations

from infinidev.engine import engine_logging


def test_finish_summary_does_not_label_blocked_run_completed(monkeypatch) -> None:
    lines: list[str] = []
    monkeypatch.setattr(engine_logging, "log", lines.append)

    engine_logging.log_finish("developer", "blocked", 2, 23, 252377)

    rendered = "\n".join(lines)
    assert "Blocked" in rendered
    assert "Completed" not in rendered


def test_finish_summary_labels_done_run_completed(monkeypatch) -> None:
    lines: list[str] = []
    monkeypatch.setattr(engine_logging, "log", lines.append)

    engine_logging.log_finish("developer", "done", 1, 10, 1000)

    assert "Completed" in "\n".join(lines)


def test_hallucinated_call_errors_are_separated_from_tool_failures():
    """Only invented call shapes count as hallucinations.

    A command that ran and returned a non-zero exit, or a tool that raised on
    real input, says something about the repository. A parameter that does not
    exist says something about the model, and each one costs a model round trip.
    """
    from infinidev.engine.engine_logging import is_hallucinated_call_error

    for invented in (
        "Unknown tool: read_fil. Did you mean one of: read_file",
        "Tool edit_file: unexpected kwargs {'old_text'}",
        "Tool 'read_file' is missing required parameter(s): file_path",
        "Tool 'x' argument validation failed: value: too short",
        "Invalid JSON arguments: {oops",
        "Tool 'x' EXISTS and is callable — your call was rejected only because "
        "of wrong parameter name(s)",
    ):
        assert is_hallucinated_call_error(invented) is True, invented

    for real_world in (
        "",
        "Tool 'execute_command' failed: Command exited with status 1",
        "File already exists: src/app.py",
        "Access denied: path '/etc/passwd' is outside allowed directories",
    ):
        assert is_hallucinated_call_error(real_world) is False, real_world


def test_malformed_calls_record_why_they_were_rejected() -> None:
    """A count with no evidence is not diagnosable.

    ``execute_tool_call`` dispatches POST_TOOL only at its end, so a rejection
    returned early reaches neither the transcript trace nor the UI. The state
    keeps the reason so the count can be explained.
    """
    from infinidev.engine.loop.models import LoopState

    state = LoopState()
    assert state.malformed_call_reasons == []

    for index in range(25):
        state.malformed_call_reasons = (
            state.malformed_call_reasons + [f"edit_file: reason {index}"]
        )[-20:]

    assert len(state.malformed_call_reasons) == 20
    assert state.malformed_call_reasons[-1] == "edit_file: reason 24"
    assert state.malformed_call_reasons[0] == "edit_file: reason 5"


def test_the_reason_keeps_the_arguments_the_model_actually_sent() -> None:
    """The message names the wrong parameter; the call shows what was invented."""
    from infinidev.engine.loop.models import LoopState

    state = LoopState()
    state.malformed_call_reasons = (
        state.malformed_call_reasons
        + ["edit_file({'old_string': 'a', 'new_string': 'b'}) -> missing file_path"]
    )[-20:]

    assert "old_string" in state.malformed_call_reasons[0]
    assert "missing file_path" in state.malformed_call_reasons[0]
