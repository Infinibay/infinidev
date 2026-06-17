use std::path::{Component, Path, PathBuf};

use crate::error::{Result, ToolError};

/// Execution context shared by all tools. The workspace is the project root;
/// every path argument is resolved relative to it and confined within it.
#[derive(Debug, Clone)]
pub struct ToolContext {
    pub workspace: PathBuf,
}

impl ToolContext {
    pub fn new(workspace: impl Into<PathBuf>) -> Self {
        let ws = workspace.into();
        // Canonicalize so traversal checks are reliable; fall back to as-is.
        let ws = std::fs::canonicalize(&ws).unwrap_or(ws);
        Self { workspace: ws }
    }

    /// Resolve a relative path inside the workspace, rejecting traversal and
    /// absolute paths. The target need not exist (used for create).
    pub fn resolve(&self, rel: &str) -> Result<PathBuf> {
        let p = Path::new(rel);
        if p.is_absolute() {
            return Err(ToolError::InvalidArgs(format!("absolute paths not allowed: {rel}")));
        }
        let mut out = self.workspace.clone();
        for comp in p.components() {
            match comp {
                Component::Normal(c) => out.push(c),
                Component::CurDir => {}
                Component::ParentDir => {
                    if !out.pop() || !out.starts_with(&self.workspace) {
                        return Err(ToolError::InvalidArgs(format!("path escapes workspace: {rel}")));
                    }
                }
                Component::RootDir | Component::Prefix(_) => {
                    return Err(ToolError::InvalidArgs(format!("invalid path: {rel}")));
                }
            }
        }
        if !out.starts_with(&self.workspace) {
            return Err(ToolError::InvalidArgs(format!("path escapes workspace: {rel}")));
        }
        Ok(out)
    }

    /// Display a path relative to the workspace (for tool output).
    pub fn display_rel<'a>(&self, p: &'a Path) -> &'a str {
        p.strip_prefix(&self.workspace)
            .ok()
            .and_then(|r| r.to_str())
            .unwrap_or_else(|| p.to_str().unwrap_or(""))
    }
}
