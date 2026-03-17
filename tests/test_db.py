"""Tests for Infinidev DB service."""
import pytest
from infinidev.db.service import (
    init_db,
    execute_with_retry,
    store_conversation_turn,
    get_recent_summaries,
)
from infinidev.config.settings import settings
import tempfile
import os


@pytest.fixture
def temp_db_path():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def temp_db(temp_db_path):
    """Setup database with temp path."""
    original = settings.DB_PATH
    settings.DB_PATH = temp_db_path
    init_db()
    yield
    settings.DB_PATH = original


class TestDatabase:
    """Test database operations."""

    def test_init_db_creates_tables(self, temp_db):
        """Database initialization creates required tables."""
        tables = execute_with_retry(
            lambda c: c.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        )
        table_names = [t[0] for t in tables]
        assert "projects" in table_names
        assert "findings" in table_names
        assert "conversation_turns" in table_names

    def test_store_conversation_turn(self, temp_db):
        """Can store and retrieve conversation turns."""
        session_id = "test-session-1"
        store_conversation_turn(
            session_id=session_id,
            role="user",
            content="Hello, how are you?",
            summary="User greeting"
        )
        store_conversation_turn(
            session_id=session_id,
            role="assistant",
            content="I'm doing well, thank you!",
            summary="Assistant response"
        )

        summaries = get_recent_summaries(session_id, limit=10)
        assert len(summaries) == 2
        # Summaries are returned oldest-first (as stored)
        assert "user" in summaries[0].lower()
        assert "assistant" in summaries[1].lower()

    def test_conversation_turns_limit(self, temp_db):
        """Conversation summaries are limited correctly."""
        session_id = "test-session-2"
        for i in range(5):
            store_conversation_turn(
                session_id=session_id,
                role="user",
                content=f"Message {i}",
                summary=f"Summary {i}"
            )

        summaries = get_recent_summaries(session_id, limit=2)
        assert len(summaries) == 2
        # Function fetches newest rows first (ORDER BY created_at DESC) then reverses,
        # so with limit=2 we get the 2 MOST RECENT summaries in oldest-first order
        # After 5 inserts (Summary 0-4), the 2 most recent are Summary 3 and Summary 4
        assert "Summary 3" in summaries[0]
        assert "Summary 4" in summaries[1]