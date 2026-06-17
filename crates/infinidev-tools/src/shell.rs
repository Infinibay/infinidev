use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use tokio::io::AsyncWriteExt;

use crate::context::ToolContext;
use crate::error::{Result, ToolError};
use crate::tool::Tool;

const MAX_OUTPUT: usize = 20_000;

/// Format a process's combined stdout/stderr + exit code into the uniform
/// `"<output>\n[exit code: N]"` string both shell tools return.
fn format_output(stdout: &[u8], stderr: &[u8], code: i32) -> String {
    let mut buf = String::new();
    buf.push_str(&String::from_utf8_lossy(stdout));
    let stderr = String::from_utf8_lossy(stderr);
    if !stderr.trim().is_empty() {
        if !buf.is_empty() {
            buf.push('\n');
        }
        buf.push_str(&stderr);
    }
    if buf.len() > MAX_OUTPUT {
        // Truncate on a char boundary so we never split a multi-byte char.
        let mut end = MAX_OUTPUT;
        while end > 0 && !buf.is_char_boundary(end) {
            end -= 1;
        }
        buf.truncate(end);
        buf.push_str("\n…(truncated)");
    }
    if buf.trim().is_empty() {
        buf = "(no output)".into();
    }
    format!("{buf}\n[exit code: {code}]")
}

#[derive(Deserialize)]
struct ExecArgs {
    command: String,
    #[serde(default)]
    timeout_secs: Option<u64>,
}

pub struct ExecuteCommand;

#[async_trait]
impl Tool for ExecuteCommand {
    fn name(&self) -> &'static str {
        "execute_command"
    }
    fn description(&self) -> &'static str {
        "Run a shell command in the project root and return its combined stdout/stderr and exit code."
    }
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to run."},
                "timeout_secs": {"type": "integer", "description": "Max seconds before the command is killed (default 120)."}
            },
            "required": ["command"]
        })
    }
    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<String> {
        let a: ExecArgs = serde_json::from_value(args).map_err(|e| ToolError::InvalidArgs(e.to_string()))?;
        let timeout = Duration::from_secs(a.timeout_secs.unwrap_or(120));

        #[cfg(windows)]
        let mut cmd = {
            let mut c = tokio::process::Command::new("cmd");
            c.args(["/C", &a.command]);
            c
        };
        #[cfg(not(windows))]
        let mut cmd = {
            let mut c = tokio::process::Command::new("sh");
            c.arg("-c").arg(&a.command);
            c
        };
        cmd.current_dir(&ctx.workspace);

        let output = tokio::time::timeout(timeout, cmd.output())
            .await
            .map_err(|_| ToolError::Exec(format!("timed out after {}s", timeout.as_secs())))??;

        Ok(format_output(&output.stdout, &output.stderr, output.status.code().unwrap_or(-1)))
    }
}

// ── code_interpreter ─────────────────────────────────────────────────────
#[derive(Deserialize)]
struct CodeArgs {
    code: String,
    #[serde(default)]
    timeout_secs: Option<u64>,
}

pub struct CodeInterpreter;

#[async_trait]
impl Tool for CodeInterpreter {
    fn name(&self) -> &'static str {
        "code_interpreter"
    }
    fn description(&self) -> &'static str {
        "Execute a Python 3 snippet in the project root (fed via stdin) and return \
         its stdout/stderr and exit code. Use for quick computations, data \
         inspection, or scripted checks. Not sandboxed — runs with your permissions."
    }
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python 3 source to execute."},
                "timeout_secs": {"type": "integer", "description": "Max seconds before the process is killed (default 120)."}
            },
            "required": ["code"]
        })
    }
    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<String> {
        let a: CodeArgs = serde_json::from_value(args).map_err(|e| ToolError::InvalidArgs(e.to_string()))?;
        let timeout = Duration::from_secs(a.timeout_secs.unwrap_or(120));

        let mut child = tokio::process::Command::new("python3")
            .arg("-")
            .current_dir(&ctx.workspace)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| ToolError::Exec(format!("failed to start python3: {e}")))?;

        // Write the code to stdin, then drop the handle to signal EOF before we
        // await output (otherwise the interpreter blocks reading stdin forever).
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(a.code.as_bytes())
                .await
                .map_err(|e| ToolError::Exec(e.to_string()))?;
        }

        let output = tokio::time::timeout(timeout, child.wait_with_output())
            .await
            .map_err(|_| ToolError::Exec(format!("timed out after {}s", timeout.as_secs())))??;

        Ok(format_output(&output.stdout, &output.stderr, output.status.code().unwrap_or(-1)))
    }
}
