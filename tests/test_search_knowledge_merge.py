"""``search_knowledge`` after it absorbed ``read_findings``.

The two tools ran the same algorithm — full-text search over findings —
behind different names, and a model that has to choose between them has to
guess. ``search_knowledge`` took over, but only because it could take over
everything ``read_findings`` did: browsing with no query at all, the
session filter that defaults to the current session, and the type filter.

These tests pin the parts a merge quietly loses. The mode split is the one
worth stating: a snippet is an excerpt *around a query*, so browsing has to
return the finding's content instead, not an empty string.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from infinidev.tools.base.context import (
    bind_tools_to_agent,
    clear_agent_context,
    set_context,
)
from infinidev.tools.base.db import execute_with_retry
from infinidev.tools.knowledge.search_knowledge_tool import SearchKnowledgeTool


@pytest.fixture
def agent(temp_db, tmp_path):
    """A bound agent whose current session is ``s1``."""
    set_context(
        project_id=1,
        agent_id="test-agent",
        agent_run_id="run-1",
        session_id="s1",
        workspace_path=str(tmp_path),
    )
    yield
    clear_agent_context("test-agent")


@pytest.fixture
def findings(agent, temp_db):
    """Three findings across two sessions."""

    rows = [
        ("auth uses JWT RS256", "the signing key lives in vault", "lesson", "s1", 0.9),
        ("tests live in tests/", "pytest, not unittest", "project_context", "s1", 0.7),
        ("older note", "from a previous session", "observation", "s2", 0.5),
    ]

    def _insert(conn):
        for topic, content, ftype, session, conf in rows:
            conn.execute(
                """INSERT INTO findings
                   (project_id, agent_id, topic, content, finding_type,
                    session_id, confidence, status)
                   VALUES (1, 'test', ?, ?, ?, ?, ?, 'active')""",
                (topic, content, ftype, session, conf),
            )
        conn.commit()

    execute_with_retry(_insert)
    return temp_db


def _run(**kwargs):
    tool = SearchKnowledgeTool()
    bind_tools_to_agent([tool], "test-agent")
    return json.loads(tool._run(**kwargs))


def _titles(payload):
    return {r["title"] for r in payload["results"]}


# ── the browse mode read_findings used to own ────────────────────────────


def test_no_query_browses_instead_of_failing(findings):
    """``read_findings()`` with no arguments was a legitimate call; an FTS
    tool that requires a query would have silently dropped that."""
    out = _run(session_id="0")
    assert out["count"] == 3


def test_browsing_returns_content_not_an_empty_snippet(findings):
    """A snippet is an excerpt around a query. With no query there is
    nothing to excerpt, so the row must carry the finding itself."""
    results = _run(session_id="0")["results"]
    assert all("content" in r for r in results)
    assert all("snippet" not in r for r in results)


def test_searching_returns_snippets(findings):
    results = _run(query="JWT")["results"]
    assert results and all("snippet" in r for r in results)


# ── the filters read_findings used to own ────────────────────────────────


def test_session_defaults_to_the_current_one(findings):
    assert _titles(_run()) == {"auth uses JWT RS256", "tests live in tests/"}


def test_session_zero_means_every_session(findings):
    assert "older note" in _titles(_run(session_id="0"))


def test_finding_type_filters(findings):
    assert _titles(_run(finding_type="project_context")) == {"tests live in tests/"}


def test_min_confidence_filters(findings):
    assert _titles(_run(session_id="0", min_confidence=0.8)) == {"auth uses JWT RS256"}


# ── reports ──────────────────────────────────────────────────────────────


def test_browsing_does_not_ask_reports_for_an_empty_match(findings):
    """An FTS MATCH needs something to match. Reports simply sit out the
    browse mode rather than raising on an empty query."""
    out = _run(session_id="0", sources=["findings", "reports"])
    assert out["count"] == 3
    assert all(r["source_type"] == "findings" for r in out["results"])


# ── the old name ─────────────────────────────────────────────────────────


def test_read_findings_is_no_longer_a_local_alias():
    """Retired local knowledge names must not bypass the Ken migration."""
    from infinidev.engine.tool_dispatch import _TOOL_ALIASES

    assert "read_findings" not in _TOOL_ALIASES


def test_semantic_mode_is_part_of_the_canonical_tool(findings, monkeypatch):
    vectors = {
        "authentication": np.array([1.0, 0.0]),
        "auth uses JWT RS256 the signing key lives in vault": np.array([1.0, 0.0]),
        "tests live in tests/ pytest, not unittest": np.array([0.0, 1.0]),
    }

    def embed(texts):
        return [vectors.get(text, np.array([0.2, 0.2])) for text in texts]

    monkeypatch.setattr("infinidev.tools.base.dedup._get_embed_fn", lambda: embed)
    out = _run(query="authentication", mode="semantic", threshold=0.8)

    assert out["mode"] == "semantic"
    assert _titles(out) == {"auth uses JWT RS256"}
    assert "content" not in out["results"][0]

    def _stored(conn):
        return conn.execute(
            "SELECT embedding, embedding_space FROM findings "
            "WHERE session_id = 's1' ORDER BY id"
        ).fetchall()

    stored = execute_with_retry(_stored)
    assert all(row["embedding"] is not None for row in stored)
    assert len({row["embedding_space"] for row in stored}) == 1


def test_old_semantic_names_are_absent_from_the_local_schema():
    from infinidev.engine.tool_dispatch import _TOOL_ALIASES
    from infinidev.tools import get_tools_for_role

    names = {tool.name for tool in get_tools_for_role("developer", supports_vision=False)}
    assert "search_findings" not in _TOOL_ALIASES
    assert "search_knowledge" not in names
    assert "search_findings" not in names
