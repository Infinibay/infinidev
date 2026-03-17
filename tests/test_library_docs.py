"""Tests for the library documentation system."""

import json
import os
import tempfile

import pytest

from infinidev.config.settings import settings
from infinidev.db.service import execute_with_retry, init_db
from infinidev.tools.docs.delete_documentation import DeleteDocumentationTool
from infinidev.tools.docs.find_documentation import FindDocumentationTool


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


class TestLibraryDocsSchema:
    """Test that init_db creates the library_docs table and FTS."""

    def test_init_db_creates_library_docs_table(self, temp_db):
        tables = execute_with_retry(
            lambda c: c.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        )
        table_names = [t[0] for t in tables]
        assert "library_docs" in table_names
        assert "library_docs_fts" in table_names

    def test_init_db_creates_triggers(self, temp_db):
        triggers = execute_with_retry(
            lambda c: c.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'library_docs_%'"
            ).fetchall()
        )
        trigger_names = {t[0] for t in triggers}
        assert "library_docs_ai" in trigger_names
        assert "library_docs_ad" in trigger_names
        assert "library_docs_au" in trigger_names

    def test_init_db_creates_index(self, temp_db):
        indexes = execute_with_retry(
            lambda c: c.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name = 'idx_library_docs_lookup'"
            ).fetchall()
        )
        assert len(indexes) == 1


class TestFindDocumentationEmpty:
    """Test find_documentation with an empty database."""

    def test_list_sections_no_docs(self, temp_db):
        tool = FindDocumentationTool()
        result = json.loads(tool._run(library_name="nonexistent"))
        assert result["found"] is False
        assert "update_documentation" in result["message"]

    def test_search_no_docs(self, temp_db):
        tool = FindDocumentationTool()
        result = json.loads(tool._run(library_name="nonexistent", query="install"))
        # Should return error (no FTS results, no embeddings)
        assert "error" in result or "found" in result


class TestUpsertAndRead:
    """Test inserting, updating, and reading documentation sections."""

    def _insert_section(self, lib, lang, ver, title, order, content):
        def _do(conn):
            conn.execute(
                """\
                INSERT INTO library_docs
                    (library_name, language, version, section_title, section_order, content)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(library_name, language, version, section_title) DO UPDATE SET
                    section_order = excluded.section_order,
                    content       = excluded.content,
                    updated_at    = CURRENT_TIMESTAMP
                """,
                (lib, lang, ver, title, order, content),
            )
            conn.commit()
        execute_with_retry(_do)

    def test_insert_and_list_sections(self, temp_db):
        self._insert_section("requests", "python", "2.31", "Overview", 0, "HTTP library for Python.")
        self._insert_section("requests", "python", "2.31", "Installation", 1, "pip install requests")
        self._insert_section("requests", "python", "2.31", "Quick Start", 2, "import requests; r = requests.get('...')")

        tool = FindDocumentationTool()
        result = json.loads(tool._run(library_name="requests", language="python", version="2.31"))
        sections = result["sections"]
        assert len(sections) == 3
        assert sections[0]["title"] == "Overview"
        assert sections[1]["title"] == "Installation"
        assert sections[2]["title"] == "Quick Start"

    def test_upsert_updates_content(self, temp_db):
        self._insert_section("flask", "python", "3.0", "Overview", 0, "Old content")
        self._insert_section("flask", "python", "3.0", "Overview", 0, "New content")

        rows = execute_with_retry(
            lambda c: c.execute(
                "SELECT content FROM library_docs WHERE library_name = 'flask' AND section_title = 'Overview'"
            ).fetchall()
        )
        assert len(rows) == 1
        assert rows[0]["content"] == "New content"

    def test_read_section_by_title(self, temp_db):
        self._insert_section("numpy", "python", "1.26", "API Reference", 0, "numpy.array(...) creates arrays")

        tool = FindDocumentationTool()
        result = json.loads(tool._run(library_name="numpy", language="python", version="1.26", section="API"))
        assert "numpy.array" in result["content"]

    def test_read_section_not_found(self, temp_db):
        self._insert_section("numpy", "python", "1.26", "Overview", 0, "NumPy overview")

        tool = FindDocumentationTool()
        result = json.loads(tool._run(library_name="numpy", language="python", version="1.26", section="Nonexistent"))
        assert "error" in result

    def test_fts_search(self, temp_db):
        self._insert_section("pandas", "python", "2.0", "Overview", 0, "DataFrame is the primary data structure")
        self._insert_section("pandas", "python", "2.0", "API", 1, "pandas.read_csv reads CSV files into DataFrames")

        tool = FindDocumentationTool()
        result = json.loads(tool._run(library_name="pandas", language="python", version="2.0", query="DataFrame"))
        assert len(result["results"]) >= 1

    def test_version_fallback_latest(self, temp_db):
        """When version='latest', should find the most recently updated version."""
        self._insert_section("react", "javascript", "18.0", "Overview", 0, "React 18 overview")

        tool = FindDocumentationTool()
        result = json.loads(tool._run(library_name="react", language="javascript", version="latest"))
        sections = result["sections"]
        assert len(sections) == 1
        assert sections[0]["title"] == "Overview"

    def test_section_order_preserved(self, temp_db):
        """Sections should be returned in section_order, not insertion order."""
        self._insert_section("lib", "py", "1.0", "Appendix", 3, "Extra info")
        self._insert_section("lib", "py", "1.0", "Getting Started", 1, "Start here")
        self._insert_section("lib", "py", "1.0", "Intro", 0, "Introduction")
        self._insert_section("lib", "py", "1.0", "Advanced", 2, "Advanced usage")

        tool = FindDocumentationTool()
        result = json.loads(tool._run(library_name="lib", language="py", version="1.0"))
        titles = [s["title"] for s in result["sections"]]
        assert titles == ["Intro", "Getting Started", "Advanced", "Appendix"]


class TestDeleteDocumentation:
    """Test deleting documentation sections."""

    def _insert_section(self, lib, lang, ver, title, order, content):
        def _do(conn):
            conn.execute(
                """\
                INSERT INTO library_docs
                    (library_name, language, version, section_title, section_order, content)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (lib, lang, ver, title, order, content),
            )
            conn.commit()
        execute_with_retry(_do)

    def test_delete_single_section(self, temp_db):
        self._insert_section("flask", "python", "3.0", "Overview", 0, "Flask overview")
        self._insert_section("flask", "python", "3.0", "API Reference", 1, "Flask API")

        tool = DeleteDocumentationTool()
        result = json.loads(tool._run(library_name="flask", language="python", version="3.0", section="Overview"))
        assert result["deleted"] == 1

        # API Reference should still exist
        remaining = execute_with_retry(
            lambda c: c.execute("SELECT count(*) as cnt FROM library_docs WHERE library_name = 'flask'").fetchone()
        )
        assert remaining["cnt"] == 1

    def test_delete_all_sections(self, temp_db):
        self._insert_section("flask", "python", "3.0", "Overview", 0, "Flask overview")
        self._insert_section("flask", "python", "3.0", "API Reference", 1, "Flask API")
        self._insert_section("flask", "python", "3.0", "Examples", 2, "Flask examples")

        tool = DeleteDocumentationTool()
        result = json.loads(tool._run(library_name="flask", language="python", version="3.0"))
        assert result["deleted"] == 3

        remaining = execute_with_retry(
            lambda c: c.execute("SELECT count(*) as cnt FROM library_docs WHERE library_name = 'flask'").fetchone()
        )
        assert remaining["cnt"] == 0

    def test_delete_nonexistent(self, temp_db):
        tool = DeleteDocumentationTool()
        result = json.loads(tool._run(library_name="nonexistent", language="python", version="1.0"))
        assert "error" in result

    def test_delete_section_like_match(self, temp_db):
        """Section parameter uses LIKE match, same as find_documentation."""
        self._insert_section("numpy", "python", "1.26", "API Reference", 0, "numpy API")
        self._insert_section("numpy", "python", "1.26", "Overview", 1, "numpy overview")

        tool = DeleteDocumentationTool()
        result = json.loads(tool._run(library_name="numpy", language="python", version="1.26", section="API"))
        assert result["deleted"] == 1
