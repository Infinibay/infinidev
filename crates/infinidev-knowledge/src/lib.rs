//! Persistent knowledge base for the Infinidev core.
//!
//! A small SQLite-backed store of "findings" (things the agent learns about a
//! project), plus the record/search/read tools that expose it to the loop.
//! Lives under `<workspace>/.infinidev/knowledge.db`. (Semantic dedup via
//! embeddings — as in the Python engine — is a future enhancement; v1 uses
//! substring search.)

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

use infinidev_tools::{Result, Tool, ToolContext, ToolError};

mod docs;
mod notes;
pub use docs::{
    docs_tools, fetch_and_store, DocHit, DocLibrary, DocSection, Docs, FetchDocumentation,
    ReadDocumentation,
};
pub use notes::{NoteRow, Notes, RecordNote};

const SCHEMA: &str = "CREATE TABLE IF NOT EXISTS findings (\
    id INTEGER PRIMARY KEY AUTOINCREMENT,\
    topic TEXT NOT NULL,\
    content TEXT NOT NULL,\
    finding_type TEXT NOT NULL DEFAULT 'observation',\
    confidence REAL NOT NULL DEFAULT 0.5,\
    created_at TEXT NOT NULL DEFAULT (datetime('now')))";

#[derive(Debug, Clone, Serialize)]
pub struct Finding {
    pub id: i64,
    pub topic: String,
    pub content: String,
    pub finding_type: String,
    pub confidence: f64,
    pub created_at: String,
}

/// SQLite-backed findings store (sync — call from a blocking task).
pub struct Knowledge {
    path: PathBuf,
}

impl Knowledge {
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

    pub fn record(&self, topic: &str, content: &str, finding_type: &str, confidence: f64) -> rusqlite::Result<i64> {
        let c = self.conn()?;
        c.execute(
            "INSERT INTO findings (topic, content, finding_type, confidence) VALUES (?1, ?2, ?3, ?4)",
            params![topic, content, finding_type, confidence],
        )?;
        Ok(c.last_insert_rowid())
    }

    pub fn search(&self, query: &str, limit: i64) -> rusqlite::Result<Vec<Finding>> {
        let c = self.conn()?;
        let pattern = format!("%{query}%");
        let mut stmt = c.prepare(
            "SELECT id, topic, content, finding_type, confidence, created_at FROM findings \
             WHERE topic LIKE ?1 OR content LIKE ?1 ORDER BY id DESC LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![pattern, limit], row_to_finding)?;
        rows.collect()
    }

    pub fn list(&self, limit: i64) -> rusqlite::Result<Vec<Finding>> {
        let c = self.conn()?;
        let mut stmt = c.prepare(
            "SELECT id, topic, content, finding_type, confidence, created_at FROM findings ORDER BY id DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit], row_to_finding)?;
        rows.collect()
    }
}

fn row_to_finding(row: &rusqlite::Row) -> rusqlite::Result<Finding> {
    Ok(Finding {
        id: row.get(0)?,
        topic: row.get(1)?,
        content: row.get(2)?,
        finding_type: row.get(3)?,
        confidence: row.get(4)?,
        created_at: row.get(5)?,
    })
}

// ── tools ───────────────────────────────────────────────────────────────
fn obj(props: serde_json::Value, required: &[&str]) -> serde_json::Value {
    serde_json::json!({ "type": "object", "properties": props, "required": required })
}

async fn open_async(ws: PathBuf) -> Result<Knowledge> {
    tokio::task::spawn_blocking(move || Knowledge::open(&ws))
        .await
        .map_err(|e| ToolError::Other(e.to_string()))?
        .map_err(|e| ToolError::Other(e.to_string()))
}

#[derive(Deserialize)]
struct RecordArgs {
    topic: String,
    content: String,
    #[serde(default)]
    finding_type: Option<String>,
    #[serde(default)]
    confidence: Option<f64>,
}

pub struct RecordFinding;

#[async_trait]
impl Tool for RecordFinding {
    fn name(&self) -> &'static str {
        "record_finding"
    }
    fn description(&self) -> &'static str {
        "Save a durable finding about the project (a class, pattern, API, decision) to the knowledge base."
    }
    fn parameters(&self) -> serde_json::Value {
        obj(
            serde_json::json!({
                "topic": {"type": "string", "description": "Short title."},
                "content": {"type": "string", "description": "The finding."},
                "finding_type": {"type": "string", "description": "observation | pattern | api | decision (default observation)."},
                "confidence": {"type": "number", "description": "0..1 (default 0.5)."}
            }),
            &["topic", "content"],
        )
    }
    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<String> {
        let a: RecordArgs = serde_json::from_value(args).map_err(|e| ToolError::InvalidArgs(e.to_string()))?;
        let kb = open_async(ctx.workspace.clone()).await?;
        let id = tokio::task::spawn_blocking(move || {
            kb.record(
                &a.topic,
                &a.content,
                a.finding_type.as_deref().unwrap_or("observation"),
                a.confidence.unwrap_or(0.5),
            )
        })
        .await
        .map_err(|e| ToolError::Other(e.to_string()))?
        .map_err(|e| ToolError::Other(e.to_string()))?;
        Ok(format!("Recorded finding #{id}."))
    }
}

#[derive(Deserialize)]
struct SearchArgs {
    query: String,
    #[serde(default)]
    limit: Option<i64>,
}

pub struct SearchFindings;

#[async_trait]
impl Tool for SearchFindings {
    fn name(&self) -> &'static str {
        "search_findings"
    }
    fn description(&self) -> &'static str {
        "Search the project knowledge base for prior findings by keyword."
    }
    fn parameters(&self) -> serde_json::Value {
        obj(
            serde_json::json!({
                "query": {"type": "string"},
                "limit": {"type": "integer", "description": "Max results (default 20)."}
            }),
            &["query"],
        )
    }
    fn is_read_only(&self) -> bool {
        true
    }
    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<String> {
        let a: SearchArgs = serde_json::from_value(args).map_err(|e| ToolError::InvalidArgs(e.to_string()))?;
        let kb = open_async(ctx.workspace.clone()).await?;
        let found = tokio::task::spawn_blocking(move || kb.search(&a.query, a.limit.unwrap_or(20)))
            .await
            .map_err(|e| ToolError::Other(e.to_string()))?
            .map_err(|e| ToolError::Other(e.to_string()))?;
        Ok(format_findings(&found))
    }
}

#[derive(Deserialize, Default)]
struct ListArgs {
    #[serde(default)]
    limit: Option<i64>,
}

pub struct ReadFindings;

#[async_trait]
impl Tool for ReadFindings {
    fn name(&self) -> &'static str {
        "read_findings"
    }
    fn description(&self) -> &'static str {
        "List the most recent findings in the project knowledge base."
    }
    fn parameters(&self) -> serde_json::Value {
        obj(
            serde_json::json!({ "limit": {"type": "integer", "description": "Max results (default 20)."} }),
            &[],
        )
    }
    fn is_read_only(&self) -> bool {
        true
    }
    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<String> {
        let a: ListArgs = serde_json::from_value(args).unwrap_or_default();
        let kb = open_async(ctx.workspace.clone()).await?;
        let found = tokio::task::spawn_blocking(move || kb.list(a.limit.unwrap_or(20)))
            .await
            .map_err(|e| ToolError::Other(e.to_string()))?
            .map_err(|e| ToolError::Other(e.to_string()))?;
        Ok(format_findings(&found))
    }
}

fn format_findings(found: &[Finding]) -> String {
    if found.is_empty() {
        return "No findings.".to_string();
    }
    found
        .iter()
        .map(|f| format!("#{} [{}] {} — {}", f.id, f.finding_type, f.topic, f.content))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The knowledge tools, for the engine to include in its tool set.
pub fn tools() -> Vec<Box<dyn Tool>> {
    vec![
        Box::new(RecordFinding),
        Box::new(SearchFindings),
        Box::new(ReadFindings),
        Box::new(RecordNote),
        Box::new(FetchDocumentation),
        Box::new(ReadDocumentation),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn store_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let kb = Knowledge::open(dir.path()).unwrap();
        let id = kb.record("auth", "JWT in middleware.rs", "pattern", 0.9).unwrap();
        assert!(id >= 1);
        assert_eq!(kb.search("JWT", 10).unwrap().len(), 1);
        assert_eq!(kb.search("nope", 10).unwrap().len(), 0);
        assert_eq!(kb.list(10).unwrap().len(), 1);
    }

    #[tokio::test]
    async fn tools_record_and_search() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolContext::new(dir.path());
        RecordFinding
            .execute(json!({"topic": "build", "content": "use cargo test -p"}), &ctx)
            .await
            .unwrap();
        let out = SearchFindings.execute(json!({"query": "cargo"}), &ctx).await.unwrap();
        assert!(out.contains("build"));
        assert!(tools().len() == 6);
    }
}
