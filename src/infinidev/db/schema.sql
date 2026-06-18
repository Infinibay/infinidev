-- Infinidev canonical database schema — SINGLE SOURCE OF TRUTH.
--
-- This file is the one authoritative definition of the on-disk SQLite schema
-- (`<workspace>/.infinidev/infinidev.db`). It is executed verbatim by BOTH
-- implementations so the database is 100% interchangeable between them:
--
--   * Python TUI  — src/infinidev/db/service.py::init_db() runs this file via
--                   `conn.executescript(...)`, then applies the additive
--                   `_migrate_add_column` upgrades for pre-existing databases.
--   * Rust/Desktop — crates/infinidev-knowledge embeds this file with
--                    `include_str!` and runs it via `conn.execute_batch(...)`.
--
-- RULES FOR EDITING:
--   * Every statement must be idempotent (`IF NOT EXISTS`).
--   * Column order matters: keep the migrated columns at the END of each table
--     in the exact order the historical `ALTER TABLE ... ADD COLUMN` migrations
--     ran, so a freshly-created DB is byte-identical to a migrated legacy DB.
--   * Only pure DDL belongs here. Seed data (the Default Project row) and
--     column back-fills for legacy databases stay in each host's init code.
--   * Requires SQLite with FTS5 (Python stdlib sqlite3 and rusqlite `bundled`
--     both ship it).

-- ─────────────────────────────────────────────────────────────────────────────
-- Core project / knowledge tables
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS projects (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    name                  TEXT NOT NULL,
    description           TEXT,
    status                TEXT NOT NULL DEFAULT 'new',
    created_at            DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS findings (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id            INTEGER NOT NULL REFERENCES projects(id),
    session_id            TEXT,
    agent_id              TEXT,
    agent_run_id          TEXT,
    topic                 TEXT,
    content               TEXT,
    status                TEXT DEFAULT 'active',
    finding_type          TEXT DEFAULT 'observation',
    confidence            REAL DEFAULT 0.5,
    sources_json          TEXT DEFAULT '[]',
    tags_json             TEXT DEFAULT '[]',
    artifact_id           INTEGER,
    validation_method     TEXT,
    reproducibility_score REAL,
    embedding             BLOB,
    created_at            DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at            DATETIME DEFAULT CURRENT_TIMESTAMP,
    -- migrated columns (anchored memory) — appended in migration order
    anchor_file           TEXT,
    anchor_symbol         TEXT,
    anchor_tool           TEXT,
    anchor_error          TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS findings_fts USING fts5(
    topic, content, content=findings, content_rowid=id
);

CREATE TRIGGER IF NOT EXISTS findings_ai AFTER INSERT ON findings BEGIN
    INSERT INTO findings_fts(rowid, topic, content)
    VALUES (new.id, new.topic, new.content);
END;
CREATE TRIGGER IF NOT EXISTS findings_ad AFTER DELETE ON findings BEGIN
    INSERT INTO findings_fts(findings_fts, rowid, topic, content)
    VALUES ('delete', old.id, old.topic, old.content);
END;
CREATE TRIGGER IF NOT EXISTS findings_au AFTER UPDATE ON findings BEGIN
    INSERT INTO findings_fts(findings_fts, rowid, topic, content)
    VALUES ('delete', old.id, old.topic, old.content);
    INSERT INTO findings_fts(rowid, topic, content)
    VALUES (new.id, new.topic, new.content);
END;

CREATE TABLE IF NOT EXISTS artifacts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id    INTEGER NOT NULL REFERENCES projects(id),
    session_id    TEXT,
    type          TEXT DEFAULT 'artifact',
    name          TEXT,
    file_path     TEXT,
    description   TEXT,
    content       TEXT,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE VIRTUAL TABLE IF NOT EXISTS artifacts_fts USING fts5(
    name, description, content, content=artifacts, content_rowid=id
);

CREATE TRIGGER IF NOT EXISTS artifacts_ai AFTER INSERT ON artifacts BEGIN
    INSERT INTO artifacts_fts(rowid, name, description, content)
    VALUES (new.id, new.name, new.description, new.content);
END;
CREATE TRIGGER IF NOT EXISTS artifacts_ad AFTER DELETE ON artifacts BEGIN
    INSERT INTO artifacts_fts(artifacts_fts, rowid, name, description, content)
    VALUES ('delete', old.id, old.name, old.description, old.content);
END;
CREATE TRIGGER IF NOT EXISTS artifacts_au AFTER UPDATE ON artifacts BEGIN
    INSERT INTO artifacts_fts(artifacts_fts, rowid, name, description, content)
    VALUES ('delete', old.id, old.name, old.description, old.content);
    INSERT INTO artifacts_fts(rowid, name, description, content)
    VALUES (new.id, new.name, new.description, new.content);
END;

CREATE TABLE IF NOT EXISTS web_cache (
    url        TEXT NOT NULL,
    format     TEXT NOT NULL DEFAULT 'markdown',
    content    TEXT,
    fetched_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(url, format)
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Conversation / session state
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS conversation_turns (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT,
    summary    TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_conv_session
    ON conversation_turns(session_id, created_at);

CREATE TABLE IF NOT EXISTS sessions (
    session_id     TEXT PRIMARY KEY,
    project_id     INTEGER,
    workspace_path TEXT,
    title          TEXT,
    turn_count     INTEGER NOT NULL DEFAULT 0,
    created_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_active_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_sessions_workspace
    ON sessions(workspace_path, last_active_at);

CREATE TABLE IF NOT EXISTS session_notes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    note_text  TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_session_notes_session
    ON session_notes(session_id, created_at);

-- ─────────────────────────────────────────────────────────────────────────────
-- Library documentation cache (FTS5)
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS library_docs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    library_name   TEXT NOT NULL,
    language       TEXT NOT NULL DEFAULT 'unknown',
    version        TEXT NOT NULL DEFAULT 'latest',
    section_title  TEXT NOT NULL,
    section_order  INTEGER NOT NULL DEFAULT 0,
    content        TEXT NOT NULL,
    embedding      BLOB,
    source_urls    TEXT DEFAULT '[]',
    created_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(library_name, language, version, section_title)
);
CREATE INDEX IF NOT EXISTS idx_library_docs_lookup
    ON library_docs(library_name, language, version);

CREATE VIRTUAL TABLE IF NOT EXISTS library_docs_fts USING fts5(
    section_title, content, content=library_docs, content_rowid=id
);
CREATE TRIGGER IF NOT EXISTS library_docs_ai AFTER INSERT ON library_docs BEGIN
    INSERT INTO library_docs_fts(rowid, section_title, content)
    VALUES (new.id, new.section_title, new.content);
END;
CREATE TRIGGER IF NOT EXISTS library_docs_ad AFTER DELETE ON library_docs BEGIN
    INSERT INTO library_docs_fts(library_docs_fts, rowid, section_title, content)
    VALUES ('delete', old.id, old.section_title, old.content);
END;
CREATE TRIGGER IF NOT EXISTS library_docs_au AFTER UPDATE ON library_docs BEGIN
    INSERT INTO library_docs_fts(library_docs_fts, rowid, section_title, content)
    VALUES ('delete', old.id, old.section_title, old.content);
    INSERT INTO library_docs_fts(rowid, section_title, content)
    VALUES (new.id, new.section_title, new.content);
END;

-- ─────────────────────────────────────────────────────────────────────────────
-- Orchestration bookkeeping
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS exploration_trees (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id       INTEGER NOT NULL REFERENCES projects(id),
    session_id       TEXT,
    agent_id         TEXT,
    problem          TEXT NOT NULL,
    tree_json        TEXT NOT NULL,
    synthesis        TEXT,
    status           TEXT DEFAULT 'running',
    total_nodes      INTEGER DEFAULT 0,
    total_tool_calls INTEGER DEFAULT 0,
    total_tokens     INTEGER DEFAULT 0,
    created_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at     DATETIME
);

CREATE TABLE IF NOT EXISTS artifact_changes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id    INTEGER NOT NULL REFERENCES projects(id),
    agent_run_id  TEXT,
    file_path     TEXT NOT NULL,
    action        TEXT NOT NULL DEFAULT 'modified',
    before_hash   TEXT,
    after_hash    TEXT,
    size_bytes    INTEGER,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS status_updates (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id    INTEGER NOT NULL REFERENCES projects(id),
    agent_id      TEXT,
    agent_run_id  TEXT,
    message       TEXT,
    progress      REAL,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS branches (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id    INTEGER NOT NULL REFERENCES projects(id),
    task_id       TEXT,
    repo_name     TEXT,
    branch_name   TEXT NOT NULL,
    base_branch   TEXT,
    status        TEXT DEFAULT 'active',
    created_by    TEXT,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Code intelligence cache
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ci_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    language TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    symbol_count INTEGER DEFAULT 0,
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- migrated columns — appended in migration order
    parser_version INTEGER DEFAULT 0,
    embedding BLOB,
    embedding_text TEXT,
    UNIQUE(project_id, file_path)
);
CREATE INDEX IF NOT EXISTS idx_ci_files_path ON ci_files(project_id, file_path);

CREATE TABLE IF NOT EXISTS ci_symbols (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    name TEXT NOT NULL,
    qualified_name TEXT DEFAULT '',
    kind TEXT NOT NULL,
    line_start INTEGER NOT NULL,
    line_end INTEGER,
    column_start INTEGER DEFAULT 0,
    signature TEXT DEFAULT '',
    type_annotation TEXT DEFAULT '',
    docstring TEXT DEFAULT '',
    parent_symbol TEXT DEFAULT '',
    visibility TEXT DEFAULT 'public',
    is_async BOOLEAN DEFAULT FALSE,
    is_static BOOLEAN DEFAULT FALSE,
    is_abstract BOOLEAN DEFAULT FALSE,
    language TEXT NOT NULL,
    -- migrated columns — appended in migration order
    embedding BLOB,
    embedding_text TEXT
);
CREATE INDEX IF NOT EXISTS idx_ci_symbols_name ON ci_symbols(name);
CREATE INDEX IF NOT EXISTS idx_ci_symbols_kind ON ci_symbols(kind, name);
CREATE INDEX IF NOT EXISTS idx_ci_symbols_file ON ci_symbols(project_id, file_path);
CREATE INDEX IF NOT EXISTS idx_ci_symbols_qualified ON ci_symbols(qualified_name);
CREATE INDEX IF NOT EXISTS idx_ci_symbols_parent ON ci_symbols(parent_symbol);

CREATE VIRTUAL TABLE IF NOT EXISTS ci_symbols_fts USING fts5(
    name, qualified_name, signature, docstring,
    content=ci_symbols, content_rowid=id
);
CREATE TRIGGER IF NOT EXISTS ci_symbols_ai AFTER INSERT ON ci_symbols BEGIN
    INSERT INTO ci_symbols_fts(rowid, name, qualified_name, signature, docstring)
    VALUES (new.id, new.name, new.qualified_name, new.signature, new.docstring);
END;
CREATE TRIGGER IF NOT EXISTS ci_symbols_ad AFTER DELETE ON ci_symbols BEGIN
    INSERT INTO ci_symbols_fts(ci_symbols_fts, rowid, name, qualified_name, signature, docstring)
    VALUES ('delete', old.id, old.name, old.qualified_name, old.signature, old.docstring);
END;
CREATE TRIGGER IF NOT EXISTS ci_symbols_au AFTER UPDATE ON ci_symbols BEGIN
    INSERT INTO ci_symbols_fts(ci_symbols_fts, rowid, name, qualified_name, signature, docstring)
    VALUES ('delete', old.id, old.name, old.qualified_name, old.signature, old.docstring);
    INSERT INTO ci_symbols_fts(rowid, name, qualified_name, signature, docstring)
    VALUES (new.id, new.name, new.qualified_name, new.signature, new.docstring);
END;

CREATE TABLE IF NOT EXISTS ci_references (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    name TEXT NOT NULL,
    line INTEGER NOT NULL,
    column_num INTEGER DEFAULT 0,
    context TEXT DEFAULT '',
    ref_kind TEXT DEFAULT 'usage',
    resolved_file TEXT DEFAULT '',
    resolved_line INTEGER,
    language TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ci_refs_name ON ci_references(name);
CREATE INDEX IF NOT EXISTS idx_ci_refs_file ON ci_references(project_id, file_path);

CREATE TABLE IF NOT EXISTS ci_imports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    source TEXT NOT NULL,
    name TEXT NOT NULL,
    alias TEXT DEFAULT '',
    line INTEGER NOT NULL,
    is_wildcard BOOLEAN DEFAULT FALSE,
    resolved_file TEXT DEFAULT '',
    language TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ci_imports_file ON ci_imports(project_id, file_path);
CREATE INDEX IF NOT EXISTS idx_ci_imports_name ON ci_imports(name);
CREATE INDEX IF NOT EXISTS idx_ci_imports_source ON ci_imports(source);

CREATE TABLE IF NOT EXISTS ci_diagnostics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    line INTEGER NOT NULL,
    severity TEXT NOT NULL,
    check_name TEXT NOT NULL,
    message TEXT NOT NULL,
    fix_suggestion TEXT DEFAULT '',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_ci_diag_file ON ci_diagnostics(project_id, file_path);

CREATE TABLE IF NOT EXISTS ci_method_bodies (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id    INTEGER NOT NULL,
    file_path     TEXT NOT NULL,
    qualified_name TEXT NOT NULL,
    kind          TEXT NOT NULL,
    line_start    INTEGER NOT NULL,
    line_end      INTEGER NOT NULL,
    body_size     INTEGER NOT NULL,
    body_hash     TEXT NOT NULL,
    body_norm     TEXT NOT NULL,
    language      TEXT NOT NULL,
    indexed_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(project_id, file_path, qualified_name)
);
CREATE INDEX IF NOT EXISTS idx_method_bodies_hash ON ci_method_bodies(project_id, body_hash);
CREATE INDEX IF NOT EXISTS idx_method_bodies_qual ON ci_method_bodies(project_id, qualified_name);
CREATE INDEX IF NOT EXISTS idx_method_bodies_file ON ci_method_bodies(project_id, file_path);

-- ─────────────────────────────────────────────────────────────────────────────
-- ContextRank — vectorized context + interaction log + scoring
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS cr_contexts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id      TEXT NOT NULL,
    session_id   TEXT NOT NULL,
    context_type TEXT NOT NULL,
    content      TEXT NOT NULL,
    embedding    BLOB,
    iteration    INTEGER,
    step_index   INTEGER,
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cr_contexts_session ON cr_contexts(session_id);
CREATE INDEX IF NOT EXISTS idx_cr_contexts_type ON cr_contexts(context_type);
CREATE INDEX IF NOT EXISTS idx_cr_contexts_created_at ON cr_contexts(created_at DESC);

CREATE TABLE IF NOT EXISTS cr_interactions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id      TEXT NOT NULL,
    session_id   TEXT NOT NULL,
    context_id   INTEGER REFERENCES cr_contexts(id),
    iteration    INTEGER NOT NULL,
    event_type   TEXT NOT NULL,
    target       TEXT NOT NULL,
    target_type  TEXT NOT NULL,
    weight       REAL NOT NULL DEFAULT 1.0,
    metadata     TEXT,
    created_at   REAL NOT NULL,
    -- migrated columns — appended in migration order
    was_error    INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_cr_interactions_target ON cr_interactions(target, target_type);
CREATE INDEX IF NOT EXISTS idx_cr_interactions_context ON cr_interactions(context_id);
CREATE INDEX IF NOT EXISTS idx_cr_interactions_session ON cr_interactions(session_id);

CREATE TABLE IF NOT EXISTS cr_session_scores (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id      TEXT NOT NULL,
    session_id   TEXT NOT NULL,
    target       TEXT NOT NULL,
    target_type  TEXT NOT NULL,
    score        REAL NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    created_at   REAL NOT NULL,
    -- migrated columns — appended in migration order
    productivity REAL DEFAULT 1.0,
    was_edited   INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_cr_scores_target ON cr_session_scores(target, target_type);
CREATE INDEX IF NOT EXISTS idx_cr_scores_session ON cr_session_scores(session_id);
CREATE INDEX IF NOT EXISTS idx_cr_session_scores_session_target
    ON cr_session_scores(session_id, target, target_type);
