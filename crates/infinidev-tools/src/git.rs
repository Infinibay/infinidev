use std::path::Path;

use async_trait::async_trait;
use serde::Deserialize;

use crate::context::ToolContext;
use crate::error::{Result, ToolError};
use crate::tool::Tool;

async fn run_git(args: &[&str], workspace: &Path) -> Result<String> {
    let output = tokio::process::Command::new("git")
        .arg("-C")
        .arg(workspace)
        .args(args)
        .output()
        .await
        .map_err(|e| ToolError::Exec(format!("git not available: {e}")))?;
    let mut buf = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.trim().is_empty() {
        buf.push_str(&stderr);
    }
    Ok(buf)
}

pub struct GitStatus;

#[async_trait]
impl Tool for GitStatus {
    fn name(&self) -> &'static str {
        "git_status"
    }
    fn description(&self) -> &'static str {
        "Show the working tree status (git status --porcelain)."
    }
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object", "properties": {} })
    }
    fn is_read_only(&self) -> bool {
        true
    }
    async fn execute(&self, _args: serde_json::Value, ctx: &ToolContext) -> Result<String> {
        let out = run_git(&["status", "--porcelain=v1", "--branch"], &ctx.workspace).await?;
        Ok(if out.trim().is_empty() { "(clean working tree)".into() } else { out })
    }
}

#[derive(Deserialize, Default)]
struct DiffArgs {
    #[serde(default)]
    path: Option<String>,
}

pub struct GitDiff;

#[async_trait]
impl Tool for GitDiff {
    fn name(&self) -> &'static str {
        "git_diff"
    }
    fn description(&self) -> &'static str {
        "Show unstaged changes (git diff), optionally scoped to a path."
    }
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Optional path to scope the diff."}
            }
        })
    }
    fn is_read_only(&self) -> bool {
        true
    }
    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<String> {
        let a: DiffArgs = serde_json::from_value(args).unwrap_or_default();
        let out = match &a.path {
            Some(p) => run_git(&["diff", "--", p], &ctx.workspace).await?,
            None => run_git(&["diff"], &ctx.workspace).await?,
        };
        Ok(if out.trim().is_empty() { "(no changes)".into() } else { out })
    }
}
