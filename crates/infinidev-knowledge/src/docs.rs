//! Library documentation cache.
//!
//! Ported from the Python `library_docs` table + `find_documentation` /
//! `update_documentation` tools. The agent fetches a docs page once, the engine
//! splits it into titled sections and stores them locally so later turns can
//! read or search them without re-fetching (and offline). The desktop UI
//! surfaces them in a three-panel browser. Stored in
//! `<workspace>/.infinidev/knowledge.db`.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

use infinidev_tools::{Result, Tool, ToolContext, ToolError};

const SCHEMA: &str = "CREATE TABLE IF NOT EXISTS library_docs (\
    id INTEGER PRIMARY KEY AUTOINCREMENT,\
    library TEXT NOT NULL,\
    version TEXT NOT NULL DEFAULT 'latest',\
    section_title TEXT NOT NULL,\
    section_order INTEGER NOT NULL DEFAULT 0,\
    content TEXT NOT NULL,\
    source_url TEXT NOT NULL DEFAULT '',\
    created_at TEXT NOT NULL DEFAULT (datetime('now')),\
    UNIQUE(library, version, section_title))";

/// A library that has cached docs, with how many sections.
#[derive(Debug, Clone, Serialize)]
pub struct DocLibrary {
    pub library: String,
    pub version: String,
    pub sections: i64,
}

/// One documentation section (header row — content fetched separately).
#[derive(Debug, Clone, Serialize)]
pub struct DocSection {
    pub section_title: String,
    pub section_order: i64,
    pub source_url: String,
}

/// A search hit across all cached docs.
#[derive(Debug, Clone, Serialize)]
pub struct DocHit {
    pub library: String,
    pub section_title: String,
    pub snippet: String,
}

/// SQLite-backed docs store (sync — call from a blocking task).
pub struct Docs {
    path: PathBuf,
}

impl Docs {
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

    /// Replace all sections for a library/version with `sections`. Returns the
    /// count stored.
    pub fn store(
        &self,
        library: &str,
        version: &str,
        source_url: &str,
        sections: &[(String, String)],
    ) -> rusqlite::Result<usize> {
        let mut c = self.conn()?;
        let tx = c.transaction()?;
        tx.execute(
            "DELETE FROM library_docs WHERE library = ?1 AND version = ?2",
            params![library, version],
        )?;
        for (i, (title, content)) in sections.iter().enumerate() {
            tx.execute(
                "INSERT INTO library_docs (library, version, section_title, section_order, content, source_url) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![library, version, title, i as i64, content, source_url],
            )?;
        }
        tx.commit()?;
        Ok(sections.len())
    }

    /// Distinct libraries (most-recent version each), with section counts.
    pub fn libraries(&self) -> rusqlite::Result<Vec<DocLibrary>> {
        let c = self.conn()?;
        let mut stmt = c.prepare(
            "SELECT library, version, COUNT(*) as n FROM library_docs \
             GROUP BY library, version ORDER BY library ASC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(DocLibrary { library: row.get(0)?, version: row.get(1)?, sections: row.get(2)? })
        })?;
        rows.collect()
    }

    /// Section headers for a library (ordered).
    pub fn sections(&self, library: &str) -> rusqlite::Result<Vec<DocSection>> {
        let c = self.conn()?;
        let mut stmt = c.prepare(
            "SELECT section_title, section_order, source_url FROM library_docs \
             WHERE library = ?1 ORDER BY section_order ASC",
        )?;
        let rows = stmt.query_map(params![library], |row| {
            Ok(DocSection {
                section_title: row.get(0)?,
                section_order: row.get(1)?,
                source_url: row.get(2)?,
            })
        })?;
        rows.collect()
    }

    /// The full content of one section.
    pub fn read(&self, library: &str, section_title: &str) -> rusqlite::Result<Option<String>> {
        let c = self.conn()?;
        c.query_row(
            "SELECT content FROM library_docs WHERE library = ?1 AND section_title = ?2 LIMIT 1",
            params![library, section_title],
            |row| row.get(0),
        )
        .map(Some)
        .or_else(|e| if matches!(e, rusqlite::Error::QueryReturnedNoRows) { Ok(None) } else { Err(e) })
    }

    /// Substring search across section titles + content.
    pub fn search(&self, query: &str, limit: i64) -> rusqlite::Result<Vec<DocHit>> {
        let c = self.conn()?;
        let pat = format!("%{query}%");
        let mut stmt = c.prepare(
            "SELECT library, section_title, content FROM library_docs \
             WHERE section_title LIKE ?1 OR content LIKE ?1 ORDER BY library, section_order LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![pat, limit], |row| {
            let content: String = row.get(2)?;
            Ok(DocHit {
                library: row.get(0)?,
                section_title: row.get(1)?,
                snippet: snippet_around(&content, query),
            })
        })?;
        rows.collect()
    }

    /// Delete every section of a library.
    pub fn delete(&self, library: &str) -> rusqlite::Result<usize> {
        self.conn()?.execute("DELETE FROM library_docs WHERE library = ?1", params![library])
    }
}

/// A ~160-char window of `content` around the first case-insensitive match of
/// `query` (or the head if not found).
fn snippet_around(content: &str, query: &str) -> String {
    let lc = content.to_lowercase();
    let q = query.to_lowercase();
    let idx = lc.find(&q).unwrap_or(0);
    let start = idx.saturating_sub(60);
    let collapsed: String = content.chars().skip(start).take(160).collect();
    collapsed.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ── sectioning ──────────────────────────────────────────────────────────────

/// Split a fetched docs page into `(title, content)` sections. HTML is split on
/// `<h1>`–`<h3>` headings; plain text / Markdown on `#`/`##` headings. A page
/// with no headings becomes a single "Overview" section.
pub fn to_sections(body: &str) -> Vec<(String, String)> {
    let looks_html = {
        let head = body.trim_start();
        let lower = head.get(..256).unwrap_or(head).to_lowercase();
        lower.contains("<!doctype html") || lower.contains("<html") || lower.contains("<body")
    };
    let marked = if looks_html { html_with_markers(body) } else { md_with_markers(body) };
    split_on_markers(&marked)
}

const MARK: &str = "\u{1}SECTION\u{1}";

/// Strip HTML to text, turning `<h1..3>` headings into section markers.
fn html_with_markers(html: &str) -> String {
    let script = regex::Regex::new(r"(?is)<script\b[^>]*>.*?</\s*script\s*>").unwrap();
    let style = regex::Regex::new(r"(?is)<style\b[^>]*>.*?</\s*style\s*>").unwrap();
    let comment = regex::Regex::new(r"(?s)<!--.*?-->").unwrap();
    let s = script.replace_all(html, " ");
    let s = style.replace_all(&s, " ");
    let s = comment.replace_all(&s, " ").into_owned();
    // Turn headings into "<MARK>title\n" so split_on_markers can see them.
    let heading = regex::Regex::new(r"(?is)<h[1-3][^>]*>(.*?)</\s*h[1-3]\s*>").unwrap();
    let s = heading.replace_all(&s, |c: &regex::Captures| {
        let title = strip_tags(&c[1]);
        format!("\n{MARK}{}\n", title.trim())
    });
    let stripped = strip_tags(&s);
    decode_entities(&stripped)
}

/// Turn Markdown `#`/`##`/`###` headings into section markers; pass the rest
/// through.
fn md_with_markers(text: &str) -> String {
    let mut out = String::new();
    for line in text.lines() {
        let t = line.trim_start();
        if let Some(rest) = t.strip_prefix("### ").or_else(|| t.strip_prefix("## ")).or_else(|| t.strip_prefix("# ")) {
            out.push('\n');
            out.push_str(MARK);
            out.push_str(rest.trim());
            out.push('\n');
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

fn split_on_markers(marked: &str) -> Vec<(String, String)> {
    let parts: Vec<&str> = marked.split(MARK).collect();
    let mut out: Vec<(String, String)> = Vec::new();
    // The first chunk is any preamble before the first heading.
    let preamble = clean(parts[0]);
    if !preamble.is_empty() {
        out.push(("Overview".to_string(), preamble));
    }
    for seg in &parts[1..] {
        let (title, body) = match seg.split_once('\n') {
            Some((t, rest)) => (t.trim().to_string(), clean(rest)),
            None => (seg.trim().to_string(), String::new()),
        };
        if title.is_empty() {
            continue;
        }
        out.push((truncate_title(&title), body));
    }
    if out.is_empty() {
        let whole = clean(marked);
        if !whole.is_empty() {
            out.push(("Overview".to_string(), whole));
        }
    }
    out
}

fn strip_tags(s: &str) -> String {
    let tags = regex::Regex::new(r"(?s)<[^>]+>").unwrap();
    tags.replace_all(s, " ").into_owned()
}

fn decode_entities(s: &str) -> String {
    s.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
}

/// Collapse whitespace within a section body.
fn clean(s: &str) -> String {
    let ws = regex::Regex::new(r"[ \t]+").unwrap();
    let collapsed = ws.replace_all(s, " ");
    let blank = regex::Regex::new(r"\n\s*\n\s*").unwrap();
    blank.replace_all(collapsed.trim(), "\n\n").trim().to_string()
}

fn truncate_title(t: &str) -> String {
    if t.chars().count() <= 120 {
        t.to_string()
    } else {
        t.chars().take(120).collect()
    }
}

// ── tools ──────────────────────────────────────────────────────────────────

fn obj(props: serde_json::Value, required: &[&str]) -> serde_json::Value {
    serde_json::json!({ "type": "object", "properties": props, "required": required })
}

async fn open_async(ws: PathBuf) -> Result<Docs> {
    tokio::task::spawn_blocking(move || Docs::open(&ws))
        .await
        .map_err(|e| ToolError::Other(e.to_string()))?
        .map_err(|e| ToolError::Other(e.to_string()))
}

/// Fetch + section + store a docs page. Shared by the tool and the UI command.
pub async fn fetch_and_store(ws: &Path, library: &str, url: &str, version: &str) -> Result<usize> {
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err(ToolError::InvalidArgs("url must start with http:// or https://".into()));
    }
    let client = reqwest::Client::builder()
        .user_agent(concat!("infinidev/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| ToolError::Other(e.to_string()))?;
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| ToolError::Other(format!("request failed: {e}")))?;
    if !resp.status().is_success() {
        return Err(ToolError::Other(format!("HTTP {} fetching {}", resp.status().as_u16(), url)));
    }
    let body = resp.text().await.map_err(|e| ToolError::Other(format!("reading body failed: {e}")))?;
    let sections = to_sections(&body);
    if sections.is_empty() {
        return Err(ToolError::Other("page had no readable content".into()));
    }
    let store = open_async(ws.to_path_buf()).await?;
    let (lib, ver, src) = (library.to_string(), version.to_string(), url.to_string());
    let n = tokio::task::spawn_blocking(move || store.store(&lib, &ver, &src, &sections))
        .await
        .map_err(|e| ToolError::Other(e.to_string()))?
        .map_err(|e| ToolError::Other(e.to_string()))?;
    Ok(n)
}

#[derive(Deserialize)]
struct FetchArgs {
    library: String,
    url: String,
    #[serde(default)]
    version: Option<String>,
}

pub struct FetchDocumentation;

#[async_trait]
impl Tool for FetchDocumentation {
    fn name(&self) -> &'static str {
        "fetch_documentation"
    }
    fn description(&self) -> &'static str {
        "Fetch a library's documentation page and cache it locally as searchable \
         sections, so later turns can read it without re-fetching. Use once per docs URL."
    }
    fn parameters(&self) -> serde_json::Value {
        obj(
            serde_json::json!({
                "library": {"type": "string", "description": "Library name to file the docs under (e.g. 'tokio')."},
                "url": {"type": "string", "description": "Absolute http(s) docs URL."},
                "version": {"type": "string", "description": "Version label (default 'latest')."}
            }),
            &["library", "url"],
        )
    }
    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<String> {
        let a: FetchArgs = serde_json::from_value(args).map_err(|e| ToolError::InvalidArgs(e.to_string()))?;
        let version = a.version.as_deref().unwrap_or("latest");
        let n = fetch_and_store(&ctx.workspace, &a.library, &a.url, version).await?;
        Ok(format!("Cached {n} section(s) of {} docs. Use read_documentation to read them.", a.library))
    }
}

#[derive(Deserialize, Default)]
struct ReadArgs {
    #[serde(default)]
    library: Option<String>,
    #[serde(default)]
    section: Option<String>,
    #[serde(default)]
    query: Option<String>,
}

pub struct ReadDocumentation;

#[async_trait]
impl Tool for ReadDocumentation {
    fn name(&self) -> &'static str {
        "read_documentation"
    }
    fn description(&self) -> &'static str {
        "Read locally cached library docs: with no args, list cached libraries; with a \
         library, list its sections; with library+section, read it; with a query, search."
    }
    fn parameters(&self) -> serde_json::Value {
        obj(
            serde_json::json!({
                "library": {"type": "string"},
                "section": {"type": "string"},
                "query": {"type": "string", "description": "Search text across all cached docs."}
            }),
            &[],
        )
    }
    fn is_read_only(&self) -> bool {
        true
    }
    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<String> {
        let a: ReadArgs = serde_json::from_value(args).unwrap_or_default();
        let store = open_async(ctx.workspace.clone()).await?;
        let out = tokio::task::spawn_blocking(move || -> rusqlite::Result<String> {
            if let Some(q) = a.query.as_deref().filter(|q| !q.trim().is_empty()) {
                let hits = store.search(q, 20)?;
                if hits.is_empty() {
                    return Ok(format!("No cached docs match '{q}'."));
                }
                return Ok(hits
                    .iter()
                    .map(|h| format!("[{}] {} — {}", h.library, h.section_title, h.snippet))
                    .collect::<Vec<_>>()
                    .join("\n"));
            }
            match (a.library.as_deref(), a.section.as_deref()) {
                (Some(lib), Some(sec)) => Ok(store
                    .read(lib, sec)?
                    .unwrap_or_else(|| format!("No section '{sec}' in {lib} docs."))),
                (Some(lib), None) => {
                    let secs = store.sections(lib)?;
                    if secs.is_empty() {
                        Ok(format!("No cached docs for {lib}. Fetch them with fetch_documentation."))
                    } else {
                        Ok(secs.iter().map(|s| format!("- {}", s.section_title)).collect::<Vec<_>>().join("\n"))
                    }
                }
                (None, _) => {
                    let libs = store.libraries()?;
                    if libs.is_empty() {
                        Ok("No cached documentation yet. Use fetch_documentation to add some.".into())
                    } else {
                        Ok(libs
                            .iter()
                            .map(|l| format!("- {} ({}) — {} sections", l.library, l.version, l.sections))
                            .collect::<Vec<_>>()
                            .join("\n"))
                    }
                }
            }
        })
        .await
        .map_err(|e| ToolError::Other(e.to_string()))?
        .map_err(|e| ToolError::Other(e.to_string()))?;
        Ok(out)
    }
}

/// The documentation tools, for the engine's developer set.
pub fn docs_tools() -> Vec<Box<dyn Tool>> {
    vec![Box::new(FetchDocumentation), Box::new(ReadDocumentation)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sections_split_on_html_headings() {
        let html = "<html><body><h1>Intro</h1><p>hello world</p>\
                    <h2>Usage</h2><p>call foo()</p></body></html>";
        let secs = to_sections(html);
        let titles: Vec<&str> = secs.iter().map(|(t, _)| t.as_str()).collect();
        assert!(titles.contains(&"Intro"));
        assert!(titles.contains(&"Usage"));
        let usage = secs.iter().find(|(t, _)| t == "Usage").unwrap();
        assert!(usage.1.contains("call foo()"));
    }

    #[test]
    fn sections_split_on_markdown_headings() {
        let md = "# Title\nbody text\n## Section A\nmore text";
        let secs = to_sections(md);
        assert!(secs.iter().any(|(t, _)| t == "Title"));
        assert!(secs.iter().any(|(t, b)| t == "Section A" && b.contains("more text")));
    }

    #[test]
    fn store_and_query_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let d = Docs::open(dir.path()).unwrap();
        d.store(
            "tokio",
            "latest",
            "https://docs.rs/tokio",
            &[("Spawn".into(), "tokio::spawn runs a future".into()), ("Runtime".into(), "build a runtime".into())],
        )
        .unwrap();
        assert_eq!(d.libraries().unwrap().len(), 1);
        assert_eq!(d.sections("tokio").unwrap().len(), 2);
        assert!(d.read("tokio", "Spawn").unwrap().unwrap().contains("runs a future"));
        let hits = d.search("future", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].section_title, "Spawn");
        // Re-storing replaces, not appends.
        d.store("tokio", "latest", "x", &[("Only".into(), "one".into())]).unwrap();
        assert_eq!(d.sections("tokio").unwrap().len(), 1);
        d.delete("tokio").unwrap();
        assert_eq!(d.libraries().unwrap().len(), 0);
    }
}
