//! Session notes — the agent's working memory across a task.
//!
//! Ported from the Python loop's `session_notes`/`add_note` mechanism. The
//! agent jots short, durable reminders ("the build uses `cargo test -p X`",
//! "auth lives in middleware.rs") with the `record_note` tool; the engine
//! injects the current notes back into the system context at the start of
//! every turn so they actually steer behaviour, and the desktop UI surfaces
//! them in a Notes browser. Stored alongside findings in
//! `<workspace>/.infinidev/knowledge.db`.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

use infinidev_tools::{Result, Tool, ToolContext, ToolError};

const SCHEMA: &str = "CREATE TABLE IF NOT EXISTS notes (\
    id INTEGER PRIMARY KEY AUTOINCREMENT,\
    note TEXT NOT NULL,\
    created_at TEXT NOT NULL DEFAULT (datetime('now')))";

/// One persisted note.
#[derive(Debug, Clone, Serialize)]
pub struct NoteRow {
    pub id: i64,
    pub note: String,
    pub created_at: String,
}

/// SQLite-backed notes store (sync — call from a blocking task).
pub struct Notes {
    path: PathBuf,
}

impl Notes {
    pub fn open(workspace: &Path) -> rusqlite::Result<Self> {
        let dir = workspace.join(".infinidev");
        let _ = std::fs::create_dir_all(&dir);
        let me = Self { path: dir.join("knowledge.db") };
        me.conn()?.execute(SCHEMA, [])?;
        Ok(me)
    }

    fn conn(&self) -> rusqlite::Result<Connection> {
        Connection::open(&self.path)
    }

    pub fn add(&self, note: &str) -> rusqlite::Result<i64> {
        let c = self.conn()?;
        c.execute("INSERT INTO notes (note) VALUES (?1)", params![note])?;
        Ok(c.last_insert_rowid())
    }

    pub fn list(&self, limit: i64) -> rusqlite::Result<Vec<NoteRow>> {
        let c = self.conn()?;
        let mut stmt = c.prepare(
            "SELECT id, note, created_at FROM notes ORDER BY id ASC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit], |row| {
            Ok(NoteRow { id: row.get(0)?, note: row.get(1)?, created_at: row.get(2)? })
        })?;
        rows.collect()
    }

    pub fn clear(&self) -> rusqlite::Result<usize> {
        self.conn()?.execute("DELETE FROM notes", [])
    }
}

// ── tool ────────────────────────────────────────────────────────────────

async fn open_async(ws: PathBuf) -> Result<Notes> {
    tokio::task::spawn_blocking(move || Notes::open(&ws))
        .await
        .map_err(|e| ToolError::Other(e.to_string()))?
        .map_err(|e| ToolError::Other(e.to_string()))
}

#[derive(Deserialize)]
struct RecordNoteArgs {
    note: String,
}

pub struct RecordNote;

#[async_trait]
impl Tool for RecordNote {
    fn name(&self) -> &'static str {
        "record_note"
    }
    fn description(&self) -> &'static str {
        "Jot a short note to your working memory (a fact, a decision, a thing to remember). \
         Notes persist across turns and are shown back to you at the start of each turn."
    }
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "note": { "type": "string", "description": "The note — one sentence, concrete." }
            },
            "required": ["note"]
        })
    }
    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<String> {
        let a: RecordNoteArgs =
            serde_json::from_value(args).map_err(|e| ToolError::InvalidArgs(e.to_string()))?;
        let note = a.note.trim().to_string();
        if note.is_empty() {
            return Err(ToolError::InvalidArgs("note is empty".into()));
        }
        let store = open_async(ctx.workspace.clone()).await?;
        let id = tokio::task::spawn_blocking(move || store.add(&note))
            .await
            .map_err(|e| ToolError::Other(e.to_string()))?
            .map_err(|e| ToolError::Other(e.to_string()))?;
        Ok(format!("Noted (#{id})."))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn notes_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let n = Notes::open(dir.path()).unwrap();
        n.add("build uses cargo test -p").unwrap();
        n.add("auth in middleware.rs").unwrap();
        let all = n.list(50).unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].note, "build uses cargo test -p"); // oldest first
        n.clear().unwrap();
        assert_eq!(n.list(50).unwrap().len(), 0);
    }

    #[tokio::test]
    async fn record_note_tool_writes() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolContext::new(dir.path());
        let out = RecordNote.execute(json!({"note": "remember this"}), &ctx).await.unwrap();
        assert!(out.starts_with("Noted"));
        let n = Notes::open(dir.path()).unwrap();
        assert_eq!(n.list(10).unwrap()[0].note, "remember this");
    }
}
