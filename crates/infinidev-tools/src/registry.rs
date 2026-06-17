use crate::files::{CodeSearch, CreateFile, ListDirectory, MultiEdit, ReadFile, ReplaceLines};
use crate::git::{GitDiff, GitStatus};
use crate::shell::{CodeInterpreter, ExecuteCommand};
use crate::tool::Tool;
use crate::web::WebFetch;

/// All built-in tools.
pub fn default_tools() -> Vec<Box<dyn Tool>> {
    vec![
        Box::new(ReadFile),
        Box::new(ListDirectory),
        Box::new(CodeSearch),
        Box::new(CreateFile),
        Box::new(ReplaceLines),
        Box::new(MultiEdit),
        Box::new(ExecuteCommand),
        Box::new(CodeInterpreter),
        Box::new(GitStatus),
        Box::new(GitDiff),
        Box::new(WebFetch),
    ]
}

/// Tools available to a tier. When `read_only_only` is true, only side-effect
/// free tools are returned — the security boundary for the chat/analyst tiers
/// (mirrors the Python `get_tools_for_role` filtering on `is_read_only`).
pub fn tools_for(read_only_only: bool) -> Vec<Box<dyn Tool>> {
    default_tools()
        .into_iter()
        .filter(|t| !read_only_only || t.is_read_only())
        .collect()
}
