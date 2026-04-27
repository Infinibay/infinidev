"""Tests for :class:`Task` and its renderer.

Focus: schema validation rules and the rendering contract that the
prompt builders rely on. Renderer output is checked structurally
(presence/absence of tags) rather than as full string compares so the
test suite stays robust to minor formatting tweaks.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from infinidev.engine.orchestration.task_renderer import render_task_xml
from infinidev.engine.orchestration.task_schema import (
    SUGGESTED_TASK_KINDS,
    Task,
    is_synthesised,
    task_from_free_text,
)


# --- Schema validation ------------------------------------------------------


def _valid_kwargs(**overrides):
    base = dict(
        title="Add JWT auth",
        description="Add HS256 JWT auth on the /login endpoint.",
        kind="feature",
        acceptance_criteria=["POST /login returns a JWT on valid creds"],
    )
    base.update(overrides)
    return base


def test_valid_task_constructs():
    t = Task(**_valid_kwargs())
    assert t.title == "Add JWT auth"
    assert t.kind == "feature"


def test_short_title_rejected():
    with pytest.raises(ValidationError):
        Task(**_valid_kwargs(title="JWT"))


def test_short_description_rejected():
    with pytest.raises(ValidationError):
        Task(**_valid_kwargs(description="too short"))


def test_empty_acceptance_criteria_rejected():
    with pytest.raises(ValidationError):
        Task(**_valid_kwargs(acceptance_criteria=[]))


def test_short_acceptance_criterion_rejected():
    with pytest.raises(ValidationError):
        Task(**_valid_kwargs(acceptance_criteria=["ok"]))


def test_kind_normalisation():
    t = Task(**_valid_kwargs(kind="Bug Fix"))
    assert t.kind == "bug_fix"


def test_kind_uppercase_normalised():
    t = Task(**_valid_kwargs(kind="FEATURE"))
    assert t.kind == "feature"


def test_non_standard_kind_accepted_with_warning(caplog):
    import logging
    caplog.set_level(logging.DEBUG, logger="infinidev.engine.orchestration.task_schema")
    t = Task(**_valid_kwargs(kind="exploratory_thing"))
    assert t.kind == "exploratory_thing"


def test_title_strips_quotes_and_periods():
    t = Task(**_valid_kwargs(title='"Add JWT auth."'))
    assert t.title == "Add JWT auth"


def test_non_falsifiable_criterion_warns_but_passes(caplog):
    import logging
    caplog.set_level(logging.WARNING, logger="infinidev.engine.orchestration.task_schema")
    t = Task(**_valid_kwargs(acceptance_criteria=["The code looks good"]))
    assert t.acceptance_criteria == ["The code looks good"]
    assert any("non-falsifiable" in m for m in caplog.messages)


def test_string_lists_strip_whitespace():
    t = Task(**_valid_kwargs(
        out_of_scope=["  refresh tokens  ", ""],
        constraints=["no new deps", "  "],
    ))
    assert t.out_of_scope == ["refresh tokens"]
    assert t.constraints == ["no new deps"]


# --- task_from_free_text ----------------------------------------------------


def test_free_text_short_raises():
    with pytest.raises(ValueError, match="too short"):
        task_from_free_text("hi")


def test_free_text_builds_minimal_task():
    t = task_from_free_text("Quiero agregar autenticación JWT al endpoint /login.")
    assert t.title.startswith("Quiero")
    assert "JWT" in t.description
    assert is_synthesised(t)


def test_free_text_explicit_title():
    t = task_from_free_text(
        "Quiero agregar autenticación JWT al endpoint /login.",
        title="JWT auth on /login",
    )
    assert t.title == "JWT auth on /login"


def test_authored_task_not_synthesised():
    t = Task(**_valid_kwargs(
        acceptance_criteria=["POST /login returns 200", "Tokens expire"],
    ))
    assert not is_synthesised(t)


# --- Renderer ---------------------------------------------------------------


def test_renderer_produces_well_formed_block():
    t = Task(**_valid_kwargs())
    out = render_task_xml(t)
    assert out.startswith("<task>")
    assert out.endswith("</task>")
    assert "<title>Add JWT auth</title>" in out
    assert "<acceptance-criteria>" in out


def test_renderer_includes_kind_description_for_known_kind():
    t = Task(**_valid_kwargs(kind="feature"))
    out = render_task_xml(t)
    expected = SUGGESTED_TASK_KINDS["feature"]
    assert expected in out


def test_renderer_marks_non_standard_kind():
    t = Task(**_valid_kwargs(kind="weird_thing"))
    out = render_task_xml(t)
    assert 'non-standard="true"' in out


def test_renderer_omits_empty_optional_sections():
    t = Task(**_valid_kwargs())  # no out_of_scope, no constraints, no references
    out = render_task_xml(t)
    assert "<out-of-scope>" not in out
    assert "<constraints>" not in out
    assert "<references>" not in out


def test_renderer_includes_optional_sections_when_set():
    t = Task(**_valid_kwargs(
        out_of_scope=["refresh tokens"],
        constraints=["no new deps"],
        references=["#142"],
    ))
    out = render_task_xml(t)
    assert "<out-of-scope>" in out
    assert "- refresh tokens" in out
    assert "<constraints>" in out
    assert "<references>" in out


def test_renderer_marks_synthesised_task():
    t = task_from_free_text("Quiero agregar autenticación JWT al /login.")
    out = render_task_xml(t)
    assert "<auto-generated" in out
    assert "auto-generated" in out


def test_renderer_does_not_mark_authored_task():
    t = Task(**_valid_kwargs(
        acceptance_criteria=["POST /login returns 200", "Tokens expire after 24h"],
    ))
    out = render_task_xml(t)
    assert "<auto-generated" not in out


def test_renderer_escapes_xml_in_user_content():
    """User content with angle brackets must not break the structure."""
    t = Task(**_valid_kwargs(
        title="Fix </task> injection bug",
        description="Some user wrote </task> in their message which is fine",
    ))
    out = render_task_xml(t)
    # The closing </task> in user content must be escaped
    assert out.count("</task>") == 1  # only the outer one
    assert "&lt;/task&gt;" in out


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
