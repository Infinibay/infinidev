//! Process-global registry for background shell commands.
//!
//! Ported from the Python `tools/shell/background_manager.py`. The developer
//! loop can launch long-running commands (dev servers, test watchers, builds)
//! without blocking its turn: each launch returns a short task id while the
//! command keeps running in a child process. Reader tasks drain the child's
//! stdout/stderr into a bounded in-memory buffer so the pipe never fills, and a
//! monitor task reaps the child (or kills it on request) and records the exit
//! code.
//!
//! The registry is a process-global singleton because both the tools that
//! mutate it AND the desktop UI's background-tasks browser need to reach the
//! same state, and neither has a natural place to thread an instance through.

use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::context::ToolContext;
use crate::error::{Result, ToolError};
use crate::tool::Tool;

/// Trailing output window retained per task (chars). A forgotten log-tailer
/// can emit unbounded output over its lifetime; we keep only the recent slice.
const MAX_BUFFER: usize = 64 * 1024;
/// How many trailing lines `background_status`/the UI show by default.
const TAIL_LINES: usize = 20;

/// A single background child process plus its captured output and status.
pub struct BackgroundTask {
    pub id: u64,
    pub command: String,
    pub description: String,
    output: Arc<Mutex<String>>,
    /// `None` while running; `Some(code)` once the process exits.
    exit_code: Arc<Mutex<Option<i32>>>,
    kill: Arc<AtomicBool>,
}

impl BackgroundTask {
    fn is_running(&self) -> bool {
        self.exit_code.lock().unwrap().is_none()
    }

    fn status_str(&self) -> String {
        match *self.exit_code.lock().unwrap() {
            None => "running".to_string(),
            Some(0) => "exited (0)".to_string(),
            Some(c) => format!("exited ({c})"),
        }
    }

    /// A serializable snapshot for the UI / status tool.
    fn view(&self) -> TaskView {
        let out = self.output.lock().unwrap();
        let tail: Vec<String> = out
            .lines()
            .rev()
            .take(TAIL_LINES)
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        TaskView {
            id: self.id,
            command: self.command.clone(),
            description: self.description.clone(),
            status: self.status_str(),
            running: self.is_running(),
            exit_code: *self.exit_code.lock().unwrap(),
            output_tail: tail.join("\n"),
        }
    }
}

/// Serializable view of a background task (for the desktop browser + status tool).
#[derive(Debug, Clone, Serialize)]
pub struct TaskView {
    pub id: u64,
    pub command: String,
    pub description: String,
    pub status: String,
    pub running: bool,
    pub exit_code: Option<i32>,
    pub output_tail: String,
}

/// The process-global registry.
pub struct BackgroundManager {
    tasks: Mutex<Vec<Arc<BackgroundTask>>>,
    next_id: AtomicU64,
}

impl BackgroundManager {
    fn new() -> Self {
        Self { tasks: Mutex::new(Vec::new()), next_id: AtomicU64::new(1) }
    }

    /// Spawn `command` (via `sh -c` / `cmd /C`) in `cwd`, returning its task id.
    pub fn spawn(&self, command: &str, description: &str, cwd: &std::path::Path) -> Result<u64> {
        #[cfg(target_os = "windows")]
        let mut cmd = {
            let mut c = tokio::process::Command::new("cmd");
            c.args(["/C", command]);
            c
        };
        #[cfg(not(target_os = "windows"))]
        let mut cmd = {
            let mut c = tokio::process::Command::new("sh");
            c.args(["-c", command]);
            c
        };
        cmd.current_dir(cwd)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::null());

        let mut child = cmd.spawn().map_err(|e| ToolError::Other(format!("spawn failed: {e}")))?;

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let output = Arc::new(Mutex::new(String::new()));
        let exit_code = Arc::new(Mutex::new(None));
        let kill = Arc::new(AtomicBool::new(false));

        // Reader tasks: drain each pipe line-by-line into the shared buffer.
        if let Some(stdout) = child.stdout.take() {
            spawn_reader(BufReader::new(stdout), output.clone(), "");
        }
        if let Some(stderr) = child.stderr.take() {
            spawn_reader(BufReader::new(stderr), output.clone(), "[stderr] ");
        }

        // Monitor task: poll for exit, honour the kill flag, record the code.
        let exit_for_monitor = exit_code.clone();
        let kill_for_monitor = kill.clone();
        tokio::spawn(async move {
            let mut killed = false;
            loop {
                if !killed && kill_for_monitor.load(Ordering::Relaxed) {
                    let _ = child.start_kill();
                    killed = true;
                }
                match child.try_wait() {
                    Ok(Some(status)) => {
                        *exit_for_monitor.lock().unwrap() = Some(status.code().unwrap_or(-1));
                        break;
                    }
                    Ok(None) => tokio::time::sleep(Duration::from_millis(200)).await,
                    Err(_) => {
                        *exit_for_monitor.lock().unwrap() = Some(-1);
                        break;
                    }
                }
            }
        });

        let desc = if description.trim().is_empty() {
            command.to_string()
        } else {
            description.to_string()
        };
        let task = Arc::new(BackgroundTask {
            id,
            command: command.to_string(),
            description: desc,
            output,
            exit_code,
            kill,
        });
        self.tasks.lock().unwrap().push(task);
        Ok(id)
    }

    /// Snapshot every task (oldest first).
    pub fn list(&self) -> Vec<TaskView> {
        self.tasks.lock().unwrap().iter().map(|t| t.view()).collect()
    }

    /// Snapshot one task by id.
    pub fn get(&self, id: u64) -> Option<TaskView> {
        self.tasks.lock().unwrap().iter().find(|t| t.id == id).map(|t| t.view())
    }

    /// Request a task be killed. Returns false if no such id. The actual kill +
    /// reap happens in the monitor task within ~200ms.
    pub fn stop(&self, id: u64) -> bool {
        match self.tasks.lock().unwrap().iter().find(|t| t.id == id) {
            Some(t) => {
                t.kill.store(true, Ordering::Relaxed);
                true
            }
            None => false,
        }
    }
}

/// Spawn a reader that appends each line of `reader` (with `prefix`) to `buf`,
/// trimming the buffer to its trailing window.
fn spawn_reader<R>(reader: BufReader<R>, buf: Arc<Mutex<String>>, prefix: &'static str)
where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let mut lines = reader.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let mut b = buf.lock().unwrap();
            b.push_str(prefix);
            b.push_str(&line);
            b.push('\n');
            if b.len() > MAX_BUFFER {
                // Drop from the front on a char boundary.
                let cut = b.len() - MAX_BUFFER;
                let mut start = cut;
                while start < b.len() && !b.is_char_boundary(start) {
                    start += 1;
                }
                *b = b[start..].to_string();
            }
        }
    });
}

/// The process-global background manager.
pub fn manager() -> &'static BackgroundManager {
    static M: OnceLock<BackgroundManager> = OnceLock::new();
    M.get_or_init(BackgroundManager::new)
}

// ── tools ──────────────────────────────────────────────────────────────────

fn obj(props: serde_json::Value, required: &[&str]) -> serde_json::Value {
    serde_json::json!({ "type": "object", "properties": props, "required": required })
}

#[derive(Deserialize)]
struct RunArgs {
    command: String,
    #[serde(default)]
    description: Option<String>,
}

pub struct RunInBackground;

#[async_trait]
impl Tool for RunInBackground {
    fn name(&self) -> &'static str {
        "run_in_background"
    }
    fn description(&self) -> &'static str {
        "Start a long-running shell command (dev server, watcher, build) in the background \
         without blocking. Returns a task id; use background_status to read its output and \
         stop_background to kill it."
    }
    fn parameters(&self) -> serde_json::Value {
        obj(
            serde_json::json!({
                "command": {"type": "string", "description": "The shell command to run."},
                "description": {"type": "string", "description": "Short label (e.g. 'vite dev server')."}
            }),
            &["command"],
        )
    }
    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<String> {
        let a: RunArgs = serde_json::from_value(args).map_err(|e| ToolError::InvalidArgs(e.to_string()))?;
        let id = manager().spawn(&a.command, a.description.as_deref().unwrap_or(""), &ctx.workspace)?;
        Ok(format!(
            "Started background task #{id}: {}. Use background_status({{\"id\": {id}}}) to read output.",
            a.description.as_deref().unwrap_or(&a.command)
        ))
    }
}

#[derive(Deserialize, Default)]
struct StatusArgs {
    #[serde(default)]
    id: Option<u64>,
}

pub struct BackgroundStatus;

#[async_trait]
impl Tool for BackgroundStatus {
    fn name(&self) -> &'static str {
        "background_status"
    }
    fn description(&self) -> &'static str {
        "Check background tasks: with no id, list all; with an id, show that task's status and \
         recent output."
    }
    fn parameters(&self) -> serde_json::Value {
        obj(
            serde_json::json!({ "id": {"type": "integer", "description": "Task id (omit to list all)."} }),
            &[],
        )
    }
    fn is_read_only(&self) -> bool {
        true
    }
    async fn execute(&self, args: serde_json::Value, _ctx: &ToolContext) -> Result<String> {
        let a: StatusArgs = serde_json::from_value(args).unwrap_or_default();
        match a.id {
            Some(id) => match manager().get(id) {
                Some(t) => Ok(format!(
                    "#{} [{}] {}\n--- recent output ---\n{}",
                    t.id, t.status, t.description, t.output_tail
                )),
                None => Err(ToolError::NotFound(format!("no background task #{id}"))),
            },
            None => {
                let tasks = manager().list();
                if tasks.is_empty() {
                    return Ok("No background tasks.".into());
                }
                Ok(tasks
                    .iter()
                    .map(|t| format!("#{} [{}] {}", t.id, t.status, t.description))
                    .collect::<Vec<_>>()
                    .join("\n"))
            }
        }
    }
}

#[derive(Deserialize)]
struct StopArgs {
    id: u64,
}

pub struct StopBackground;

#[async_trait]
impl Tool for StopBackground {
    fn name(&self) -> &'static str {
        "stop_background"
    }
    fn description(&self) -> &'static str {
        "Kill a background task by its id."
    }
    fn parameters(&self) -> serde_json::Value {
        obj(serde_json::json!({ "id": {"type": "integer"} }), &["id"])
    }
    async fn execute(&self, args: serde_json::Value, _ctx: &ToolContext) -> Result<String> {
        let a: StopArgs = serde_json::from_value(args).map_err(|e| ToolError::InvalidArgs(e.to_string()))?;
        if manager().stop(a.id) {
            Ok(format!("Stopping background task #{}.", a.id))
        } else {
            Err(ToolError::NotFound(format!("no background task #{}", a.id)))
        }
    }
}

/// The background-task tools, for the engine to include in its developer set.
pub fn background_tools() -> Vec<Box<dyn Tool>> {
    vec![Box::new(RunInBackground), Box::new(BackgroundStatus), Box::new(StopBackground)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn spawn_capture_and_exit() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolContext::new(dir.path());
        let out = RunInBackground
            .execute(json!({"command": "echo hello-bg", "description": "echo test"}), &ctx)
            .await
            .unwrap();
        assert!(out.contains("Started background task"));

        // Give the reader + monitor a moment to capture output and reap.
        for _ in 0..30 {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let tasks = manager().list();
            if let Some(t) = tasks.last() {
                if !t.running && t.output_tail.contains("hello-bg") {
                    return;
                }
            }
        }
        panic!("background task never captured output / exited");
    }

    #[tokio::test]
    async fn stop_unknown_is_error() {
        let ctx = ToolContext::new(std::env::temp_dir());
        let res = StopBackground.execute(json!({"id": 999999}), &ctx).await;
        assert!(res.is_err());
    }
}
