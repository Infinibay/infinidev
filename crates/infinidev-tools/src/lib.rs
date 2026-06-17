//! Infinidev agent tools — file, search, shell and git operations exposed as a
//! uniform [`Tool`] trait the engine can dispatch and advertise to the LLM.

pub mod background;
pub mod context;
pub mod error;
pub mod tool;

mod files;
mod git;
mod registry;
mod shell;
mod web;

pub use background::{
    background_tools, manager as background_manager, BackgroundStatus, RunInBackground,
    StopBackground, TaskView,
};
pub use context::ToolContext;
pub use error::{Result, ToolError};
pub use files::{
    search, CodeSearch, CreateFile, ListDirectory, MultiEdit, ReadFile, ReplaceLines, SearchHit,
};
pub use git::{GitDiff, GitStatus};
pub use registry::{default_tools, tools_for};
pub use shell::{CodeInterpreter, ExecuteCommand};
pub use tool::Tool;
pub use web::WebFetch;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ctx() -> (tempfile::TempDir, ToolContext) {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolContext::new(dir.path());
        (dir, ctx)
    }

    #[tokio::test]
    async fn file_create_read_roundtrip() {
        let (_d, c) = ctx();
        let out = CreateFile
            .execute(json!({"path": "sub/a.txt", "content": "hello\nworld"}), &c)
            .await
            .unwrap();
        assert!(out.contains("Wrote"));
        let read = ReadFile.execute(json!({"path": "sub/a.txt"}), &c).await.unwrap();
        assert_eq!(read, "hello\nworld");
        // line range
        let line2 = ReadFile
            .execute(json!({"path": "sub/a.txt", "start_line": 2, "end_line": 2}), &c)
            .await
            .unwrap();
        assert_eq!(line2, "world");
    }

    #[tokio::test]
    async fn replace_lines_is_deterministic() {
        let (_d, c) = ctx();
        CreateFile
            .execute(json!({"path": "f.txt", "content": "l1\nl2\nl3"}), &c)
            .await
            .unwrap();
        ReplaceLines
            .execute(json!({"path": "f.txt", "start_line": 2, "end_line": 2, "content": "X"}), &c)
            .await
            .unwrap();
        let read = ReadFile.execute(json!({"path": "f.txt"}), &c).await.unwrap();
        assert_eq!(read, "l1\nX\nl3");
    }

    #[tokio::test]
    async fn code_search_finds_matches() {
        let (_d, c) = ctx();
        CreateFile
            .execute(json!({"path": "code.rs", "content": "fn main() {}\nlet needle = 1;"}), &c)
            .await
            .unwrap();
        let hits = CodeSearch.execute(json!({"pattern": "needle"}), &c).await.unwrap();
        assert!(hits.contains("code.rs:2"));
        assert!(hits.contains("needle"));
    }

    #[tokio::test]
    async fn list_directory_lists_entries() {
        let (_d, c) = ctx();
        CreateFile.execute(json!({"path": "z.txt", "content": ""}), &c).await.unwrap();
        let out = ListDirectory.execute(json!({}), &c).await.unwrap();
        assert!(out.contains("z.txt"));
    }

    #[tokio::test]
    async fn execute_command_runs() {
        let (_d, c) = ctx();
        let out = ExecuteCommand
            .execute(json!({"command": "echo infinidev"}), &c)
            .await
            .unwrap();
        assert!(out.contains("infinidev"));
        assert!(out.contains("[exit code: 0]"));
    }

    #[tokio::test]
    async fn code_interpreter_runs_python() {
        // Skip gracefully where python3 isn't installed.
        if std::process::Command::new("python3").arg("--version").output().is_err() {
            return;
        }
        let (_d, c) = ctx();
        let out = CodeInterpreter
            .execute(json!({"code": "print(6 * 7)"}), &c)
            .await
            .unwrap();
        assert!(out.contains("42"));
        assert!(out.contains("[exit code: 0]"));
    }

    #[test]
    fn traversal_is_rejected() {
        let (_d, c) = ctx();
        assert!(c.resolve("../etc/passwd").is_err());
        assert!(c.resolve("/etc/passwd").is_err());
        assert!(c.resolve("ok/nested.txt").is_ok());
    }

    #[test]
    fn read_only_filter() {
        assert_eq!(default_tools().len(), 11);
        let ro = tools_for(true);
        assert_eq!(ro.len(), 5); // read_file, list_directory, code_search, git_status, git_diff
        assert!(ro.iter().all(|t| t.is_read_only()));
        // multi_edit (writes), web_fetch (network) and code_interpreter (runs
        // code) are developer-only — never exposed to the read-only chat tier.
        assert!(!ro
            .iter()
            .any(|t| matches!(t.name(), "multi_edit" | "web_fetch" | "code_interpreter")));
    }

    #[test]
    fn tool_schema_matches_name() {
        let t = ReadFile;
        let schema = t.schema();
        assert_eq!(schema.function.name, "read_file");
        assert!(t.is_read_only());
    }
}
