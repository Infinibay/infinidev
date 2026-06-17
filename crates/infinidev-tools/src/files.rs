//! File tools: read, list, search, create, replace-lines.
//! Mirrors the contracts of the Python `tools/file` set. `replace_lines` is a
//! deterministic line-range replacement (no fuzzy text matching).

use async_trait::async_trait;
use serde::Deserialize;

use crate::context::ToolContext;
use crate::error::{Result, ToolError};
use crate::tool::Tool;

const MAX_FILE_BYTES: u64 = 5 * 1024 * 1024;
const SKIP_DIRS: &[&str] = &[
    "node_modules", ".git", "__pycache__", ".venv", "venv", "target", "dist",
    "build", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".infinidev", ".ken",
];

fn obj(props: serde_json::Value, required: &[&str]) -> serde_json::Value {
    serde_json::json!({ "type": "object", "properties": props, "required": required })
}

// ── read_file ──────────────────────────────────────────────────────────────
#[derive(Deserialize)]
struct ReadArgs {
    path: String,
    #[serde(default)]
    start_line: Option<usize>,
    #[serde(default)]
    end_line: Option<usize>,
}

pub struct ReadFile;

#[async_trait]
impl Tool for ReadFile {
    fn name(&self) -> &'static str {
        "read_file"
    }
    fn description(&self) -> &'static str {
        "Read a UTF-8 text file relative to the project root. Optionally restrict to a 1-based line range."
    }
    fn parameters(&self) -> serde_json::Value {
        obj(
            serde_json::json!({
                "path": {"type": "string", "description": "Path relative to the project root."},
                "start_line": {"type": "integer", "description": "Optional 1-based first line."},
                "end_line": {"type": "integer", "description": "Optional 1-based last line (inclusive)."}
            }),
            &["path"],
        )
    }
    fn is_read_only(&self) -> bool {
        true
    }
    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<String> {
        let a: ReadArgs = parse(args)?;
        let path = ctx.resolve(&a.path)?;
        let meta = tokio::fs::metadata(&path)
            .await
            .map_err(|_| ToolError::NotFound(a.path.clone()))?;
        if !meta.is_file() {
            return Err(ToolError::InvalidArgs(format!("not a file: {}", a.path)));
        }
        if meta.len() > MAX_FILE_BYTES {
            return Err(ToolError::InvalidArgs(format!("file too large: {}", a.path)));
        }
        let text = tokio::fs::read_to_string(&path).await?;
        match (a.start_line, a.end_line) {
            (Some(s), Some(e)) if s >= 1 && e >= s => {
                let slice: Vec<&str> = text
                    .lines()
                    .skip(s - 1)
                    .take(e - s + 1)
                    .collect();
                Ok(slice.join("\n"))
            }
            _ => Ok(text),
        }
    }
}

// ── list_directory ───────────────────────────────────────────────────────
#[derive(Deserialize, Default)]
struct ListArgs {
    #[serde(default)]
    path: Option<String>,
}

pub struct ListDirectory;

#[async_trait]
impl Tool for ListDirectory {
    fn name(&self) -> &'static str {
        "list_directory"
    }
    fn description(&self) -> &'static str {
        "List entries of a directory (relative to the project root). Skips common junk dirs."
    }
    fn parameters(&self) -> serde_json::Value {
        obj(
            serde_json::json!({
                "path": {"type": "string", "description": "Directory relative to the project root (default: root)."}
            }),
            &[],
        )
    }
    fn is_read_only(&self) -> bool {
        true
    }
    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<String> {
        let a: ListArgs = parse(args)?;
        let dir = ctx.resolve(a.path.as_deref().unwrap_or("."))?;
        let mut rd = tokio::fs::read_dir(&dir)
            .await
            .map_err(|_| ToolError::NotFound(a.path.unwrap_or_else(|| ".".into())))?;
        let mut dirs = Vec::new();
        let mut files = Vec::new();
        while let Some(entry) = rd.next_entry().await? {
            let name = entry.file_name().to_string_lossy().into_owned();
            if SKIP_DIRS.contains(&name.as_str()) {
                continue;
            }
            if entry.file_type().await.map(|t| t.is_dir()).unwrap_or(false) {
                dirs.push(format!("{name}/"));
            } else {
                files.push(name);
            }
        }
        dirs.sort();
        files.sort();
        dirs.extend(files);
        Ok(if dirs.is_empty() { "(empty)".into() } else { dirs.join("\n") })
    }
}

// ── code_search ────────────────────────────────────────────────────────────
#[derive(Deserialize)]
struct SearchArgs {
    pattern: String,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    max_results: Option<usize>,
}

pub struct CodeSearch;

#[async_trait]
impl Tool for CodeSearch {
    fn name(&self) -> &'static str {
        "code_search"
    }
    fn description(&self) -> &'static str {
        "Regex search across files (gitignore-aware) under a path. Returns path:line: text matches."
    }
    fn parameters(&self) -> serde_json::Value {
        obj(
            serde_json::json!({
                "pattern": {"type": "string", "description": "Regular expression to search for."},
                "path": {"type": "string", "description": "Subtree to search (default: project root)."},
                "max_results": {"type": "integer", "description": "Cap on matches (default 200)."}
            }),
            &["pattern"],
        )
    }
    fn is_read_only(&self) -> bool {
        true
    }
    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<String> {
        let a: SearchArgs = parse(args)?;
        let hits = search(ctx, &a.pattern, a.path.as_deref(), a.max_results.unwrap_or(200)).await?;
        Ok(if hits.is_empty() {
            "No matches.".into()
        } else {
            hits.iter()
                .map(|h| format!("{}:{}: {}", h.path, h.line, h.text))
                .collect::<Vec<_>>()
                .join("\n")
        })
    }
}

/// A single search match (structured form), reused by the desktop search UI.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SearchHit {
    pub path: String,
    pub line: usize,
    pub text: String,
}

/// Regex search across files under `path` (gitignore-aware), returning
/// structured hits. Shared by the `code_search` tool and the GUI.
pub async fn search(
    ctx: &ToolContext,
    pattern: &str,
    path: Option<&str>,
    max: usize,
) -> Result<Vec<SearchHit>> {
    let re = regex::Regex::new(pattern)
        .map_err(|e| ToolError::InvalidArgs(format!("bad regex: {e}")))?;
    let root = ctx.resolve(path.unwrap_or("."))?;
    let workspace = ctx.workspace.clone();
    tokio::task::spawn_blocking(move || {
        let mut out: Vec<SearchHit> = Vec::new();
        for entry in ignore::WalkBuilder::new(&root).hidden(false).build().flatten() {
            if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }
            let p = entry.path();
            let Ok(text) = std::fs::read_to_string(p) else { continue };
            let rel = p.strip_prefix(&workspace).unwrap_or(p).to_string_lossy().into_owned();
            for (i, line) in text.lines().enumerate() {
                if re.is_match(line) {
                    let t: String = line.chars().take(300).collect();
                    out.push(SearchHit { path: rel.clone(), line: i + 1, text: t.trim().to_string() });
                    if out.len() >= max {
                        return out;
                    }
                }
            }
        }
        out
    })
    .await
    .map_err(|e| ToolError::Other(e.to_string()))
}

// ── create_file ────────────────────────────────────────────────────────────
#[derive(Deserialize)]
struct CreateArgs {
    path: String,
    content: String,
}

pub struct CreateFile;

#[async_trait]
impl Tool for CreateFile {
    fn name(&self) -> &'static str {
        "create_file"
    }
    fn description(&self) -> &'static str {
        "Create or overwrite a file (relative to the project root), creating parent directories."
    }
    fn parameters(&self) -> serde_json::Value {
        obj(
            serde_json::json!({
                "path": {"type": "string", "description": "Path relative to the project root."},
                "content": {"type": "string", "description": "Full file contents."}
            }),
            &["path", "content"],
        )
    }
    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<String> {
        let a: CreateArgs = parse(args)?;
        let path = ctx.resolve(&a.path)?;
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(&path, a.content.as_bytes()).await?;
        Ok(format!("Wrote {} ({} bytes).", a.path, a.content.len()))
    }
}

// ── replace_lines ──────────────────────────────────────────────────────────
#[derive(Deserialize)]
struct ReplaceArgs {
    path: String,
    start_line: usize,
    end_line: usize,
    content: String,
}

pub struct ReplaceLines;

#[async_trait]
impl Tool for ReplaceLines {
    fn name(&self) -> &'static str {
        "replace_lines"
    }
    fn description(&self) -> &'static str {
        "Replace an inclusive 1-based line range [start_line, end_line] with new content. Deterministic — no text matching."
    }
    fn parameters(&self) -> serde_json::Value {
        obj(
            serde_json::json!({
                "path": {"type": "string", "description": "Path relative to the project root."},
                "start_line": {"type": "integer", "description": "1-based first line to replace."},
                "end_line": {"type": "integer", "description": "1-based last line to replace (inclusive)."},
                "content": {"type": "string", "description": "Replacement text (may be multiple lines)."}
            }),
            &["path", "start_line", "end_line", "content"],
        )
    }
    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<String> {
        let a: ReplaceArgs = parse(args)?;
        if a.start_line < 1 || a.end_line < a.start_line {
            return Err(ToolError::InvalidArgs("require 1 <= start_line <= end_line".into()));
        }
        let path = ctx.resolve(&a.path)?;
        let text = tokio::fs::read_to_string(&path)
            .await
            .map_err(|_| ToolError::NotFound(a.path.clone()))?;
        let mut segments: Vec<&str> = text.split('\n').collect();
        if a.end_line > segments.len() {
            return Err(ToolError::InvalidArgs(format!(
                "end_line {} exceeds file length {}",
                a.end_line,
                segments.len()
            )));
        }
        let replacement: Vec<&str> = a.content.split('\n').collect();
        segments.splice((a.start_line - 1)..a.end_line, replacement.iter().copied());
        tokio::fs::write(&path, segments.join("\n").as_bytes()).await?;
        Ok(format!(
            "Replaced lines {}-{} in {}.",
            a.start_line, a.end_line, a.path
        ))
    }
}

// ── multi_edit ─────────────────────────────────────────────────────────────
#[derive(Deserialize)]
struct EditOp {
    start_line: usize,
    end_line: usize,
    content: String,
}

#[derive(Deserialize)]
struct MultiEditArgs {
    path: String,
    edits: Vec<EditOp>,
}

pub struct MultiEdit;

#[async_trait]
impl Tool for MultiEdit {
    fn name(&self) -> &'static str {
        "multi_edit"
    }
    fn description(&self) -> &'static str {
        "Apply several line-range replacements to ONE file in a single call. Each \
         edit replaces an inclusive 1-based [start_line, end_line] range. Ranges \
         must not overlap; they are applied bottom-to-top so line numbers stay \
         valid. Read the file first to get exact line numbers."
    }
    fn parameters(&self) -> serde_json::Value {
        obj(
            serde_json::json!({
                "path": {"type": "string", "description": "Path relative to the project root."},
                "edits": {
                    "type": "array",
                    "description": "Non-overlapping edits to apply.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "start_line": {"type": "integer", "description": "1-based first line to replace."},
                            "end_line": {"type": "integer", "description": "1-based last line to replace (inclusive)."},
                            "content": {"type": "string", "description": "Replacement text (may be multiple lines)."}
                        },
                        "required": ["start_line", "end_line", "content"]
                    }
                }
            }),
            &["path", "edits"],
        )
    }
    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<String> {
        let a: MultiEditArgs = parse(args)?;
        if a.edits.is_empty() {
            return Err(ToolError::InvalidArgs("edits must not be empty".into()));
        }
        let path = ctx.resolve(&a.path)?;
        let text = tokio::fs::read_to_string(&path)
            .await
            .map_err(|_| ToolError::NotFound(a.path.clone()))?;
        let mut segments: Vec<String> = text.split('\n').map(str::to_string).collect();
        let len = segments.len();

        // Validate ranges and reject overlaps. Sort ascending by start to check
        // adjacency, then apply descending so earlier splices don't shift the
        // line numbers of later (higher) ones.
        let mut ops: Vec<&EditOp> = a.edits.iter().collect();
        ops.sort_by_key(|e| e.start_line);
        let mut prev_end = 0usize;
        for e in &ops {
            if e.start_line < 1 || e.end_line < e.start_line {
                return Err(ToolError::InvalidArgs(format!(
                    "invalid range {}-{}: require 1 <= start_line <= end_line",
                    e.start_line, e.end_line
                )));
            }
            if e.end_line > len {
                return Err(ToolError::InvalidArgs(format!(
                    "end_line {} exceeds file length {len}",
                    e.end_line
                )));
            }
            if e.start_line <= prev_end {
                return Err(ToolError::InvalidArgs(format!(
                    "overlapping edits near line {}",
                    e.start_line
                )));
            }
            prev_end = e.end_line;
        }

        for e in ops.iter().rev() {
            let replacement: Vec<String> = e.content.split('\n').map(str::to_string).collect();
            segments.splice((e.start_line - 1)..e.end_line, replacement);
        }
        tokio::fs::write(&path, segments.join("\n").as_bytes()).await?;
        Ok(format!("Applied {} edits to {}.", a.edits.len(), a.path))
    }
}

fn parse<T: serde::de::DeserializeOwned>(args: serde_json::Value) -> Result<T> {
    serde_json::from_value(args).map_err(|e| ToolError::InvalidArgs(e.to_string()))
}

#[cfg(test)]
mod multi_edit_tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn applies_non_overlapping_edits_bottom_up() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "a\nb\nc\nd\ne").unwrap();
        let ctx = ToolContext::new(dir.path());
        // Replace line 2 (b→B) and lines 4-5 (d,e→D). Bottom-up keeps line 2 valid.
        let out = MultiEdit
            .execute(
                json!({"path": "f.txt", "edits": [
                    {"start_line": 2, "end_line": 2, "content": "B"},
                    {"start_line": 4, "end_line": 5, "content": "D"}
                ]}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(out.contains("Applied 2 edits"));
        let result = std::fs::read_to_string(dir.path().join("f.txt")).unwrap();
        assert_eq!(result, "a\nB\nc\nD");
    }

    #[tokio::test]
    async fn rejects_overlapping_edits() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "1\n2\n3\n4").unwrap();
        let ctx = ToolContext::new(dir.path());
        let err = MultiEdit
            .execute(
                json!({"path": "f.txt", "edits": [
                    {"start_line": 1, "end_line": 2, "content": "x"},
                    {"start_line": 2, "end_line": 3, "content": "y"}
                ]}),
                &ctx,
            )
            .await;
        assert!(err.is_err());
    }
}
