"""Conversation decisions must survive routing and short acknowledgement turns."""

from __future__ import annotations

import pytest

from infinidev.config.settings import settings
from infinidev.db.service import get_recent_turns_full, store_conversation_turn
from infinidev.engine.orchestration.chat_agent import _build_user_message
from infinidev.tools.base.db import execute_with_retry


@pytest.fixture(autouse=True)
def context_capacity(monkeypatch):
    monkeypatch.setattr(settings, "LLM_CONTEXT_WINDOW", 262_144)


def test_earlier_decisions_survive_short_followups(temp_db):
    store_conversation_turn("chat", "user", "Brújula: bicicletas, SQLite, sin login.")
    store_conversation_turn("chat", "assistant", "Entendido.")
    for text in ("Turnos de 40 minutos.", "Corrección: se llama Faro.", "Miércoles cerrado."):
        store_conversation_turn("chat", "user", text)
        store_conversation_turn("chat", "assistant", "OK")
        store_conversation_turn("chat", "work_summary", "Internal checkpoint.")
    current = "Repasemos lo acordado."
    store_conversation_turn("chat", "user", current)

    message = _build_user_message(current, "chat")

    for decision in ("SQLite", "sin login", "40 minutos", "Faro", "Miércoles cerrado"):
        assert decision in message
    assert message.count(current) == 1


def test_long_user_message_keeps_middle_decisions_when_capacity_allows(temp_db):
    content = "contexto " * 180 + "NO USAR SERVICIOS EXTERNOS" + " contexto" * 180
    store_conversation_turn("chat", "user", content)
    store_conversation_turn("chat", "assistant", "OK")

    assert "NO USAR SERVICIOS EXTERNOS" in _build_user_message("seguimos", "chat")


def test_history_order_is_stable_for_same_timestamp(temp_db):
    for index in range(10):
        store_conversation_turn("chat", "user", f"decision-{index}")
    execute_with_retry(lambda conn: conn.execute(
        "UPDATE conversation_turns SET created_at = '2026-11-12 10:00:00'"
    ))

    assert get_recent_turns_full("chat", limit=3) == [
        ("user", "decision-7"), ("user", "decision-8"), ("user", "decision-9"),
    ]


def test_small_context_is_bounded_and_discloses_older_history(temp_db, monkeypatch):
    monkeypatch.setattr(settings, "LLM_CONTEXT_WINDOW", 4096)
    for index in range(30):
        store_conversation_turn("chat", "user", f"old-{index}: " + "x" * 300)
    store_conversation_turn("chat", "user", "última decisión: usar SQLite")

    message = _build_user_message("seguimos", "chat")

    assert len(message) < 1800
    assert "última decisión: usar SQLite" in message
    assert "chat_history_search" in message
    assert "old-0:" not in message


def test_context_capacity_lookup_uses_the_resolved_authenticated_route(temp_db, monkeypatch):
    from infinidev.engine.session_context import render_session_context

    route = {
        "model": "custom_openai/private-local-model",
        "base_url": "http://localhost:9000/v1", "api_key": "test-local-credential",
    }
    observed = []
    monkeypatch.setattr("infinidev.config.llm.get_litellm_params", lambda: route)
    monkeypatch.setattr(
        "infinidev.engine.loop.model_context.get_model_context_window",
        lambda params: observed.append(params) or 262_144,
    )
    store_conversation_turn("chat", "user", "Use SQLite.")

    assert "SQLite" in render_session_context("chat", "continue")
    assert observed == [route]


def test_unicode_history_respects_small_context_budget(temp_db, monkeypatch):
    monkeypatch.setattr(settings, "LLM_CONTEXT_WINDOW", 4096)
    for _ in range(10):
        store_conversation_turn("chat", "user", "🚲" * 300)

    message = _build_user_message("seguimos", "chat")

    assert len(message.encode("utf-8")) < 1800
    assert "chat_history_search" in message
