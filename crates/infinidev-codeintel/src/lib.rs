//! Lightweight code intelligence: extract top-level symbols (functions,
//! types, classes) from a source file.
//!
//! v1 is a fast, dependency-light, multi-language **regex** extractor — robust
//! and good enough for "what's defined in this file / jump to it". A future
//! version can swap in tree-sitter (as the Python engine uses) for precise
//! parsing without changing the tool surface.

use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};

use infinidev_tools::{search, Result, Tool, ToolContext, ToolError};

#[derive(Debug, Clone, Serialize)]
pub struct Symbol {
    pub name: String,
    pub kind: String,
    pub line: usize,
}

fn patterns(ext: &str) -> Vec<(&'static str, Regex)> {
    let specs: &[(&str, &str)] = match ext {
        "rs" => &[
            ("fn", r"^\s*(?:pub\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)"),
            ("struct", r"^\s*(?:pub\s+)?struct\s+([A-Za-z_][A-Za-z0-9_]*)"),
            ("enum", r"^\s*(?:pub\s+)?enum\s+([A-Za-z_][A-Za-z0-9_]*)"),
            ("trait", r"^\s*(?:pub\s+)?trait\s+([A-Za-z_][A-Za-z0-9_]*)"),
            ("impl", r"^\s*impl(?:<[^>]*>)?\s+(?:[A-Za-z_][\w:]*\s+for\s+)?([A-Za-z_][\w:]*)"),
            ("mod", r"^\s*(?:pub\s+)?mod\s+([A-Za-z_][A-Za-z0-9_]*)"),
        ],
        "py" => &[
            ("def", r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)"),
            ("class", r"^\s*class\s+([A-Za-z_]\w*)"),
        ],
        "js" | "jsx" | "ts" | "tsx" | "mjs" | "cjs" => &[
            ("function", r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)"),
            ("class", r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)"),
            ("const", r"^\s*(?:export\s+)?(?:const|let)\s+([A-Za-z_$][\w$]*)\s*="),
        ],
        "go" => &[
            ("func", r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)"),
            ("type", r"^\s*type\s+([A-Za-z_]\w*)"),
        ],
        _ => &[],
    };
    specs
        .iter()
        .filter_map(|(k, p)| Regex::new(p).ok().map(|re| (*k, re)))
        .collect()
}

fn ext_of(path: &str) -> &str {
    path.rsplit('.').next().unwrap_or("")
}

/// Extract symbols from source text given the file path (used for language detection).
pub fn extract_symbols(path: &str, source: &str) -> Vec<Symbol> {
    let pats = patterns(ext_of(path));
    if pats.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (i, line) in source.lines().enumerate() {
        for (kind, re) in &pats {
            if let Some(caps) = re.captures(line) {
                if let Some(name) = caps.get(1) {
                    out.push(Symbol { name: name.as_str().to_string(), kind: (*kind).to_string(), line: i + 1 });
                }
            }
        }
    }
    out
}

pub struct ListSymbols;

#[async_trait]
impl Tool for ListSymbols {
    fn name(&self) -> &'static str {
        "list_symbols"
    }
    fn description(&self) -> &'static str {
        "List the symbols (functions, types, classes) defined in a source file, with line numbers."
    }
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": { "path": {"type": "string", "description": "Source file relative to the project root."} },
            "required": ["path"]
        })
    }
    fn is_read_only(&self) -> bool {
        true
    }
    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<String> {
        let path = args
            .get("path")
            .and_then(|p| p.as_str())
            .ok_or_else(|| ToolError::InvalidArgs("missing 'path'".into()))?
            .to_string();
        let abs = ctx.resolve(&path)?;
        let source = tokio::fs::read_to_string(&abs)
            .await
            .map_err(|_| ToolError::NotFound(path.clone()))?;
        let symbols = extract_symbols(&path, &source);
        if symbols.is_empty() {
            return Ok("No symbols found (or unsupported language).".to_string());
        }
        Ok(symbols
            .iter()
            .map(|s| format!("L{}: {} {}", s.line, s.kind, s.name))
            .collect::<Vec<_>>()
            .join("\n"))
    }
}

// ── get_symbol_code ──────────────────────────────────────────────────────
#[derive(Deserialize)]
struct GetSymbolArgs {
    path: String,
    name: String,
}

pub struct GetSymbolCode;

#[async_trait]
impl Tool for GetSymbolCode {
    fn name(&self) -> &'static str {
        "get_symbol_code"
    }
    fn description(&self) -> &'static str {
        "Return the source of a named symbol (function/type/class) in a file, with \
         1-based line numbers so you can follow up with replace_lines. The body runs \
         from the symbol's declaration to just before the next top-level symbol."
    }
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Source file relative to the project root."},
                "name": {"type": "string", "description": "Symbol name to extract."}
            },
            "required": ["path", "name"]
        })
    }
    fn is_read_only(&self) -> bool {
        true
    }
    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<String> {
        let a: GetSymbolArgs = serde_json::from_value(args)
            .map_err(|e| ToolError::InvalidArgs(e.to_string()))?;
        let abs = ctx.resolve(&a.path)?;
        let source = tokio::fs::read_to_string(&abs)
            .await
            .map_err(|_| ToolError::NotFound(a.path.clone()))?;
        let symbols = extract_symbols(&a.path, &source);
        let target = symbols
            .iter()
            .find(|s| s.name == a.name)
            .ok_or_else(|| ToolError::NotFound(format!("symbol '{}' in {}", a.name, a.path)))?;

        // End just before the next symbol that starts after this one, else EOF.
        let total = source.lines().count();
        let end = symbols
            .iter()
            .map(|s| s.line)
            .filter(|&l| l > target.line)
            .min()
            .map(|l| l - 1)
            .unwrap_or(total)
            .max(target.line);

        let body = source
            .lines()
            .enumerate()
            .skip(target.line - 1)
            .take(end - target.line + 1)
            .map(|(i, line)| format!("{:>5}  {line}", i + 1))
            .collect::<Vec<_>>()
            .join("\n");
        Ok(format!(
            "{} {} (L{}-{}):\n{body}",
            target.kind, target.name, target.line, end
        ))
    }
}

// ── find_references ────────────────────────────────────────────────────────
#[derive(Deserialize)]
struct FindRefsArgs {
    name: String,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    max_results: Option<usize>,
}

pub struct FindReferences;

#[async_trait]
impl Tool for FindReferences {
    fn name(&self) -> &'static str {
        "find_references"
    }
    fn description(&self) -> &'static str {
        "Find where a symbol name appears across the project (whole-word matches, \
         gitignore-aware). Returns path:line: text. Useful before renaming or editing \
         a symbol to see every call site."
    }
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Symbol name to find references to."},
                "path": {"type": "string", "description": "Subtree to search (default: project root)."},
                "max_results": {"type": "integer", "description": "Cap on matches (default 200)."}
            },
            "required": ["name"]
        })
    }
    fn is_read_only(&self) -> bool {
        true
    }
    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<String> {
        let a: FindRefsArgs = serde_json::from_value(args)
            .map_err(|e| ToolError::InvalidArgs(e.to_string()))?;
        let pattern = format!(r"\b{}\b", regex::escape(&a.name));
        let hits = search(ctx, &pattern, a.path.as_deref(), a.max_results.unwrap_or(200)).await?;
        Ok(if hits.is_empty() {
            format!("No references to '{}' found.", a.name)
        } else {
            hits.iter()
                .map(|h| format!("{}:{}: {}", h.path, h.line, h.text))
                .collect::<Vec<_>>()
                .join("\n")
        })
    }
}

/// Code-intelligence tools for the engine to include.
pub fn tools() -> Vec<Box<dyn Tool>> {
    vec![
        Box::new(ListSymbols),
        Box::new(GetSymbolCode),
        Box::new(FindReferences),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extracts_rust_and_python() {
        let rs = "pub fn main() {}\nstruct Foo;\nimpl Foo {}\nenum E { A }";
        let syms = extract_symbols("a.rs", rs);
        let names: Vec<_> = syms.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"main"));
        assert!(names.contains(&"Foo"));
        assert!(names.contains(&"E"));

        let py = "def hello():\n    pass\nclass Bar:\n    pass";
        let psyms = extract_symbols("b.py", py);
        assert!(psyms.iter().any(|s| s.name == "hello" && s.kind == "def"));
        assert!(psyms.iter().any(|s| s.name == "Bar" && s.kind == "class"));
    }

    #[tokio::test]
    async fn list_symbols_tool() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("x.rs"), "fn alpha() {}\nfn beta() {}").unwrap();
        let ctx = ToolContext::new(dir.path());
        let out = ListSymbols.execute(json!({"path": "x.rs"}), &ctx).await.unwrap();
        assert!(out.contains("alpha"));
        assert!(out.contains("beta"));
        assert_eq!(tools().len(), 3);
    }

    #[tokio::test]
    async fn get_symbol_code_slices_a_function() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("x.rs"),
            "fn alpha() {\n    let a = 1;\n}\nfn beta() {\n    let b = 2;\n}",
        )
        .unwrap();
        let ctx = ToolContext::new(dir.path());
        let out = GetSymbolCode
            .execute(json!({"path": "x.rs", "name": "alpha"}), &ctx)
            .await
            .unwrap();
        // Slices alpha (L1-3) and stops before beta (L4).
        assert!(out.contains("L1-3"));
        assert!(out.contains("let a = 1"));
        assert!(!out.contains("let b = 2"));
    }

    #[tokio::test]
    async fn find_references_matches_whole_words() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("x.rs"),
            "fn alpha() {}\nfn caller() { alpha(); }\nlet alphabet = 1;",
        )
        .unwrap();
        let ctx = ToolContext::new(dir.path());
        let out = FindReferences.execute(json!({"name": "alpha"}), &ctx).await.unwrap();
        assert!(out.contains("x.rs:1"));
        assert!(out.contains("x.rs:2"));
        // `alphabet` must NOT match (whole-word boundary).
        assert!(!out.contains("x.rs:3"));
    }
}
