"""Regression tests for the schema.sql wiring of init_db().

init_db() provisions fresh databases from db/schema.sql (the single source of
truth, mirrored by the Rust crate) and then applies idempotent column
back-fills for older DBs. These tests guard against drift between init_db()'s
result and schema.sql, and against a broken seed.
"""

import sqlite3

from infinidev.db import service


def _schema(conn):
    tables = sorted(
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    )
    cols = {
        t: [(r[1], r[2], r[3], r[4]) for r in conn.execute(f"PRAGMA table_info({t})")]
        for t in tables
    }
    idx = sorted(
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
        )
    )
    trg = sorted(r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'"))
    return tables, cols, idx, trg


def _build_via_init_db(db_path, monkeypatch):
    """Run init_db()'s inner _init against a specific db file."""
    def fake(fn, *a, **k):
        conn = sqlite3.connect(db_path)
        try:
            fn(conn)
        finally:
            conn.close()
    monkeypatch.setattr(service, "execute_with_retry", fake)
    service.init_db()


class TestSchemaWiring:
    def test_init_db_matches_schema_sql_exactly(self, tmp_path, monkeypatch):
        """init_db() must produce the same tables/columns/indexes/triggers as schema.sql.

        Catches drift such as a migration added to init_db whose column is not
        also in schema.sql.
        """
        db_init = tmp_path / "init.db"
        _build_via_init_db(str(db_init), monkeypatch)

        db_sql = tmp_path / "schema.db"
        c_sql = sqlite3.connect(str(db_sql))
        c_sql.executescript(service._load_schema_sql())
        c_sql.commit()

        c_init = sqlite3.connect(str(db_init))
        s_init = _schema(c_init)
        s_sql = _schema(c_sql)
        c_init.close()
        c_sql.close()

        assert s_init[0] == s_sql[0], "table set differs"
        assert s_init[1] == s_sql[1], "column layout differs"
        assert s_init[2] == s_sql[2], "index set differs"
        assert s_init[3] == s_sql[3], "trigger set differs"

    def test_init_db_is_idempotent_and_seeds_once(self, tmp_path, monkeypatch):
        """Running init_db twice must not error and must seed exactly one default project."""
        db = tmp_path / "x.db"
        _build_via_init_db(str(db), monkeypatch)
        _build_via_init_db(str(db), monkeypatch)  # second run: existing DB
        c = sqlite3.connect(str(db))
        rows = c.execute("SELECT name FROM projects").fetchall()
        c.close()
        assert rows == [("Default Project",)]

    def test_core_tables_present(self, tmp_path, monkeypatch):
        db = tmp_path / "y.db"
        _build_via_init_db(str(db), monkeypatch)
        c = sqlite3.connect(str(db))
        tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        c.close()
        for t in (
            "projects", "findings", "artifacts", "sessions", "conversation_turns",
            "session_notes", "library_docs", "ci_files", "ci_symbols", "ci_references",
            "cr_interactions", "cr_session_scores",
        ):
            assert t in tables, f"missing core table: {t}"

    def test_migrated_columns_present_on_fresh_db(self, tmp_path, monkeypatch):
        """Columns added via _migrate_add_column must exist on a fresh DB (via schema.sql)."""
        db = tmp_path / "z.db"
        _build_via_init_db(str(db), monkeypatch)
        c = sqlite3.connect(str(db))
        findings_cols = {r[1] for r in c.execute("PRAGMA table_info(findings)")}
        ci_symbol_cols = {r[1] for r in c.execute("PRAGMA table_info(ci_symbols)")}
        c.close()
        assert {"session_id", "anchor_file", "updated_at"} <= findings_cols
        assert {"embedding", "embedding_text"} <= ci_symbol_cols
