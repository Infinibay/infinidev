//! Native shell for Infinidev — now with the agent engine **embedded** (no
//! Python sidecar). The whole core runs in-process as Rust (`infinidev-engine`):
//!
//!   * `engine_send` runs a turn; the loop's [`EngineEvent`]s are emitted to the
//!     WebView as `engine://event`, which the React store consumes.
//!   * config / model discovery / file ops / search are plain commands backed
//!     by the same crates the engine uses.
//!
//! The frontend talks to this via Tauri `invoke` + `listen`, so there is no
//! local HTTP server in the desktop build.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use serde::{Deserialize, Serialize};
use tauri::menu::{AboutMetadata, Menu, MenuItem, PredefinedMenuItem, Submenu};
use tauri::{AppHandle, Emitter, Manager, Runtime, State};
use tokio::sync::{oneshot, Notify};

use infinidev_engine::{
    background_manager, list_models, search as engine_search, EngineConfig, EngineEvent,
    EngineHost, Message, Orchestrator, Role, SearchHit, TaskView, ToolContext, PROVIDERS,
};
use infinidev_knowledge::{
    fetch_and_store, DocHit, DocLibrary, DocSection, Docs, Finding, Knowledge, NoteRow, Notes,
};

const SKIP_DIRS: &[&str] = &[
    "node_modules", ".git", "__pycache__", ".venv", "venv", "target", "dist", "build",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".infinidev", ".ken", ".idea", ".vscode",
];
const MAX_FILE_BYTES: u64 = 5 * 1024 * 1024;

// ── session state ──────────────────────────────────────────────────────────
struct Session {
    history: Mutex<Vec<Message>>,
    config: Mutex<EngineConfig>,
    /// Where `config` is persisted (set during setup once the app config dir is
    /// known). `None` disables persistence (e.g. if the dir can't be resolved).
    config_path: Mutex<Option<PathBuf>>,
    /// Where the conversation history is persisted, so a session resumes on
    /// relaunch. `None` disables persistence.
    history_path: Mutex<Option<PathBuf>>,
    /// Set while a turn is running; `cancel_turn` fires it to abort the turn.
    cancel: Mutex<Option<Arc<Notify>>>,
    /// Pending `ask_user` prompts awaiting a `answer_prompt` from the webview.
    asks: Arc<AskRegistry>,
}

impl Default for Session {
    fn default() -> Self {
        Self {
            history: Mutex::new(Vec::new()),
            config: Mutex::new(EngineConfig::default()),
            config_path: Mutex::new(None),
            history_path: Mutex::new(None),
            cancel: Mutex::new(None),
            asks: Arc::new(AskRegistry::default()),
        }
    }
}

/// Registry of in-flight host questions: each gets a fresh id and a oneshot the
/// `answer_prompt` command resolves once the user responds.
#[derive(Default)]
struct AskRegistry {
    next: AtomicU64,
    pending: Mutex<HashMap<u64, oneshot::Sender<String>>>,
}

/// Payload for the `engine://ask` event the webview listens for.
#[derive(Clone, Serialize)]
struct AskPayload {
    id: u64,
    prompt: String,
    kind: String,
}

/// Write the current config to its persistence path, if one is set. Best-effort
/// — a failed write must not break the command that triggered it.
fn persist_config(state: &Session) {
    let path = state.config_path.lock().unwrap().clone();
    if let Some(path) = path {
        let snapshot = state.config.lock().unwrap().clone();
        if let Ok(text) = serde_json::to_string_pretty(&snapshot) {
            let _ = std::fs::write(path, text);
        }
    }
}

/// Persist the conversation history so the session resumes on relaunch.
/// Best-effort — a failed write must not break the turn.
fn persist_history(state: &Session) {
    let path = state.history_path.lock().unwrap().clone();
    if let Some(path) = path {
        let snapshot = state.history.lock().unwrap().clone();
        if let Ok(text) = serde_json::to_string(&snapshot) {
            let _ = std::fs::write(path, text);
        }
    }
}

// ── engine host: forward events to the WebView ──────────────────────────────
struct TauriHost {
    app: AppHandle,
    asks: Arc<AskRegistry>,
}

#[async_trait::async_trait]
impl EngineHost for TauriHost {
    fn emit(&self, event: EngineEvent) {
        let _ = self.app.emit("engine://event", event);
    }

    /// Ask the webview a question and block until `answer_prompt` resolves it.
    /// Returns `None` if the channel is dropped (e.g. the turn was cancelled).
    async fn ask_user(&self, prompt: String, kind: String) -> Option<String> {
        let id = self.asks.next.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.asks.pending.lock().unwrap().insert(id, tx);
        let _ = self.app.emit("engine://ask", AskPayload { id, prompt, kind });
        let answer = rx.await.ok();
        // Drop a stale entry if the channel closed without an answer.
        self.asks.pending.lock().unwrap().remove(&id);
        answer
    }
}

/// The initial project directory, from `$INFINIDEV_PROJECT` or the cwd.
fn default_project_dir() -> String {
    std::env::var("INFINIDEV_PROJECT").unwrap_or_else(|_| {
        std::env::current_dir()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|_| ".".to_string())
    })
}

/// Process-global current project directory. Lets "Open Folder" switch the
/// working root at runtime without threading a path through every command.
fn project_slot() -> &'static Mutex<String> {
    static SLOT: OnceLock<Mutex<String>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(default_project_dir()))
}

fn project_dir() -> String {
    project_slot().lock().unwrap().clone()
}

fn ctx() -> ToolContext {
    ToolContext::new(project_dir())
}

/// Switch the active project directory (File → Open Folder). Returns the
/// canonical path on success.
#[tauri::command]
fn set_project(path: String) -> Result<String, String> {
    let p = std::path::PathBuf::from(&path);
    if !p.is_dir() {
        return Err(format!("Not a directory: {path}"));
    }
    let abs = std::fs::canonicalize(&p)
        .map(|x| x.to_string_lossy().into_owned())
        .unwrap_or(path);
    *project_slot().lock().unwrap() = abs.clone();
    Ok(abs)
}

// ── engine ───────────────────────────────────────────────────────────────
#[tauri::command]
async fn engine_send(
    app: AppHandle,
    state: State<'_, Session>,
    text: String,
    images: Option<Vec<String>>,
) -> Result<(), String> {
    // Snapshot config + history WITHOUT holding the lock across the await.
    let cfg = state.config.lock().unwrap().clone();
    let history = state.history.lock().unwrap().clone();
    let images = images.unwrap_or_default();

    let orchestrator = Orchestrator::new(cfg, ctx()).map_err(|e| e.to_string())?;
    let host = TauriHost { app, asks: state.asks.clone() };

    // Register a cancellation signal for the duration of the turn. `cancel_turn`
    // fires it; we race it against the turn and drop the future on cancel, which
    // unwinds every in-flight await (LLM stream, tool call) cooperatively.
    let notify = Arc::new(Notify::new());
    *state.cancel.lock().unwrap() = Some(notify.clone());

    let outcome = tokio::select! {
        biased;
        _ = notify.notified() => None,
        res = orchestrator.run_turn_images(&text, &images, &history, &host) => Some(res),
    };
    *state.cancel.lock().unwrap() = None;

    match outcome {
        Some(Ok(result)) => {
            {
                let mut h = state.history.lock().unwrap();
                h.push(Message::user(text));
                h.push(Message::assistant(result));
            }
            persist_history(&state);
        }
        Some(Err(e)) => {
            host.emit(EngineEvent::Error { message: e.to_string() });
            host.emit(EngineEvent::TurnEnd { result: String::new() });
        }
        None => {
            // Cancelled: the partial turn is discarded (history unchanged) and
            // the UI is returned to idle.
            host.emit(EngineEvent::Notify {
                speaker: "system".into(),
                text: "Turn cancelled.".into(),
                kind: "system".into(),
            });
            host.emit(EngineEvent::TurnEnd { result: String::new() });
        }
    }
    Ok(())
}

/// Abort the turn in flight, if any (no-op when idle).
#[tauri::command]
fn cancel_turn(state: State<'_, Session>) {
    if let Some(n) = state.cancel.lock().unwrap().as_ref() {
        n.notify_one();
    }
}

/// Resolve a pending `ask_user` prompt with the user's answer.
#[tauri::command]
fn answer_prompt(state: State<'_, Session>, id: u64, answer: String) {
    if let Some(tx) = state.asks.pending.lock().unwrap().remove(&id) {
        let _ = tx.send(answer);
    }
}

#[tauri::command]
fn reset_session(state: State<'_, Session>) {
    state.history.lock().unwrap().clear();
    persist_history(&state);
}

/// Serializable view of the persisted history, for the webview to redraw the
/// conversation on startup (user/assistant turns only — tool detail isn't kept).
#[derive(Serialize)]
struct HistoryMsg {
    role: String,
    content: String,
}

#[tauri::command]
fn get_history(state: State<'_, Session>) -> Vec<HistoryMsg> {
    state
        .history
        .lock()
        .unwrap()
        .iter()
        .filter_map(|m| {
            let role = match m.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => "system",
                Role::Tool => "tool",
            };
            m.content
                .as_ref()
                .filter(|c| !c.trim().is_empty())
                .map(|c| HistoryMsg { role: role.into(), content: c.clone() })
        })
        .collect()
}

// ── config + models ────────────────────────────────────────────────────────
#[derive(Deserialize)]
struct ConfigPatch {
    provider: Option<String>,
    model: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    temperature: Option<f32>,
    max_iterations: Option<u32>,
    planning: Option<bool>,
    orchestrate: Option<bool>,
    review: Option<bool>,
    force_manual_tools: Option<bool>,
    confirm_commands: Option<bool>,
    confirm_writes: Option<bool>,
}

#[tauri::command]
fn set_config(state: State<'_, Session>, patch: ConfigPatch) -> EngineConfig {
    let updated = {
        let mut cfg = state.config.lock().unwrap();
        if let Some(p) = patch.provider {
            cfg.provider = p;
        }
        if let Some(m) = patch.model {
            cfg.model = m;
        }
        if patch.api_key.is_some() {
            cfg.api_key = patch.api_key;
        }
        if let Some(b) = patch.base_url {
            cfg.base_url = b;
        }
        if let Some(t) = patch.temperature {
            cfg.temperature = Some(t);
        }
        if let Some(n) = patch.max_iterations {
            cfg.max_iterations = n;
        }
        if let Some(b) = patch.planning {
            cfg.planning = b;
        }
        if let Some(b) = patch.orchestrate {
            cfg.orchestrate = b;
        }
        if let Some(b) = patch.review {
            cfg.review = b;
        }
        if let Some(b) = patch.force_manual_tools {
            cfg.force_manual_tools = b;
        }
        if let Some(b) = patch.confirm_commands {
            cfg.confirm_commands = b;
        }
        if let Some(b) = patch.confirm_writes {
            cfg.confirm_writes = b;
        }
        cfg.clone()
    }; // release the config lock before persisting (persist re-locks it)
    persist_config(&state);
    updated
}

#[tauri::command]
fn get_config(state: State<'_, Session>) -> EngineConfig {
    state.config.lock().unwrap().clone()
}

#[tauri::command]
async fn list_provider_models(
    provider: String,
    api_key: Option<String>,
    base_url: String,
) -> Result<Vec<String>, String> {
    list_models(&provider, api_key, &base_url)
        .await
        .map_err(|e| e.to_string())
}

#[derive(Serialize)]
struct ProviderDto {
    id: String,
    display_name: String,
    api_key_required: bool,
    base_url_editable: bool,
    default_base_url: String,
}

#[tauri::command]
fn providers() -> Vec<ProviderDto> {
    PROVIDERS
        .iter()
        .map(|p| ProviderDto {
            id: p.id.to_string(),
            display_name: p.display_name.to_string(),
            api_key_required: p.api_key_required,
            base_url_editable: p.base_url_editable,
            default_base_url: p.default_base_url.to_string(),
        })
        .collect()
}

// ── files + search ─────────────────────────────────────────────────────────
#[derive(Serialize)]
struct TreeEntry {
    name: String,
    path: String,
    is_dir: bool,
}

#[tauri::command]
fn fs_tree(path: String) -> Result<Vec<TreeEntry>, String> {
    let c = ctx();
    let dir = c.resolve(if path.is_empty() { "." } else { &path }).map_err(|e| e.to_string())?;
    let rd = std::fs::read_dir(&dir).map_err(|e| e.to_string())?;
    let mut entries = Vec::new();
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if SKIP_DIRS.contains(&name.as_str()) {
            continue;
        }
        let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
        let rel = e
            .path()
            .strip_prefix(&c.workspace)
            .map(|r| r.to_string_lossy().into_owned())
            .unwrap_or_else(|_| name.clone());
        entries.push(TreeEntry { name, path: rel, is_dir });
    }
    entries.sort_by(|a, b| {
        b.is_dir
            .cmp(&a.is_dir)
            .then(a.name.to_lowercase().cmp(&b.name.to_lowercase()))
    });
    Ok(entries)
}

#[derive(Serialize)]
struct FileContent {
    path: String,
    text: String,
    binary: bool,
}

#[tauri::command]
fn fs_read(path: String) -> Result<FileContent, String> {
    let c = ctx();
    let p = c.resolve(&path).map_err(|e| e.to_string())?;
    let meta = std::fs::metadata(&p).map_err(|_| format!("not found: {path}"))?;
    if meta.len() > MAX_FILE_BYTES {
        return Err("file too large".into());
    }
    let bytes = std::fs::read(&p).map_err(|e| e.to_string())?;
    let binary = bytes.iter().take(4096).any(|&b| b == 0);
    let text = if binary {
        String::new()
    } else {
        String::from_utf8_lossy(&bytes).into_owned()
    };
    Ok(FileContent { path, text, binary })
}

#[tauri::command]
fn fs_write(path: String, text: String) -> Result<(), String> {
    let c = ctx();
    let p = c.resolve(&path).map_err(|e| e.to_string())?;
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(&p, text).map_err(|e| e.to_string())
}

/// Create an empty file or a directory. Errors if the path already exists.
#[tauri::command]
fn fs_create(path: String, dir: bool) -> Result<(), String> {
    let c = ctx();
    let p = c.resolve(&path).map_err(|e| e.to_string())?;
    if p.exists() {
        return Err(format!("Already exists: {path}"));
    }
    if dir {
        std::fs::create_dir_all(&p).map_err(|e| e.to_string())
    } else {
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(&p, "").map_err(|e| e.to_string())
    }
}

/// Move/rename a file or directory. Errors if the target already exists.
#[tauri::command]
fn fs_rename(from: String, to: String) -> Result<(), String> {
    let c = ctx();
    let a = c.resolve(&from).map_err(|e| e.to_string())?;
    let b = c.resolve(&to).map_err(|e| e.to_string())?;
    if b.exists() {
        return Err(format!("Target exists: {to}"));
    }
    if let Some(parent) = b.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::rename(&a, &b).map_err(|e| e.to_string())
}

/// Delete a file, or a directory and everything under it.
#[tauri::command]
fn fs_delete(path: String) -> Result<(), String> {
    let c = ctx();
    let p = c.resolve(&path).map_err(|e| e.to_string())?;
    let meta = std::fs::metadata(&p).map_err(|e| e.to_string())?;
    if meta.is_dir() {
        std::fs::remove_dir_all(&p).map_err(|e| e.to_string())
    } else {
        std::fs::remove_file(&p).map_err(|e| e.to_string())
    }
}

#[tauri::command]
async fn project_search(query: String) -> Result<Vec<SearchHit>, String> {
    engine_search(&ctx(), &query, None, 500)
        .await
        .map_err(|e| e.to_string())
}

// ── knowledge base (findings) ────────────────────────────────────────────────
fn open_knowledge() -> Result<Knowledge, String> {
    Knowledge::open(std::path::Path::new(&project_dir())).map_err(|e| e.to_string())
}

/// Most-recent findings the agent has recorded for the active project.
#[tauri::command]
fn knowledge_list(limit: Option<i64>) -> Result<Vec<Finding>, String> {
    open_knowledge()?
        .list(limit.unwrap_or(200))
        .map_err(|e| e.to_string())
}

/// Record a new finding in the project knowledge base. Returns its id.
#[tauri::command]
fn knowledge_record(topic: String, content: String, finding_type: Option<String>) -> Result<i64, String> {
    open_knowledge()?
        .record(
            &topic,
            &content,
            finding_type.as_deref().unwrap_or("observation"),
            0.7,
        )
        .map_err(|e| e.to_string())
}

/// Findings matching `query` (falls back to the recent list when blank).
#[tauri::command]
fn knowledge_search(query: String, limit: Option<i64>) -> Result<Vec<Finding>, String> {
    let kb = open_knowledge()?;
    let limit = limit.unwrap_or(200);
    if query.trim().is_empty() {
        kb.list(limit).map_err(|e| e.to_string())
    } else {
        kb.search(&query, limit).map_err(|e| e.to_string())
    }
}

// ── library docs cache ───────────────────────────────────────────────────────
fn open_docs() -> Result<Docs, String> {
    Docs::open(std::path::Path::new(&project_dir())).map_err(|e| e.to_string())
}

/// Libraries with cached documentation (for the docs browser's left panel).
#[tauri::command]
fn docs_libraries() -> Result<Vec<DocLibrary>, String> {
    open_docs()?.libraries().map_err(|e| e.to_string())
}

/// Section headers for a library (the browser's middle panel).
#[tauri::command]
fn docs_sections(library: String) -> Result<Vec<DocSection>, String> {
    open_docs()?.sections(&library).map_err(|e| e.to_string())
}

/// Full content of one section (the browser's right panel).
#[tauri::command]
fn docs_read(library: String, section: String) -> Result<String, String> {
    Ok(open_docs()?
        .read(&library, &section)
        .map_err(|e| e.to_string())?
        .unwrap_or_default())
}

/// Search across all cached docs.
#[tauri::command]
fn docs_search(query: String) -> Result<Vec<DocHit>, String> {
    open_docs()?.search(&query, 50).map_err(|e| e.to_string())
}

/// Fetch a docs URL and cache it (the browser's "Fetch" action). Returns the
/// number of sections stored.
#[tauri::command]
async fn docs_fetch(library: String, url: String, version: Option<String>) -> Result<usize, String> {
    let ws = std::path::PathBuf::from(project_dir());
    fetch_and_store(&ws, &library, &url, version.as_deref().unwrap_or("latest"))
        .await
        .map_err(|e| e.to_string())
}

/// Remove a library's cached docs.
#[tauri::command]
fn docs_delete(library: String) -> Result<(), String> {
    open_docs()?.delete(&library).map(|_| ()).map_err(|e| e.to_string())
}

// ── background tasks ─────────────────────────────────────────────────────────
/// Live snapshot of every background task the agent started (dev servers,
/// watchers, builds), for the Background Tasks browser. Read fresh each poll.
#[tauri::command]
fn bg_list() -> Vec<TaskView> {
    background_manager().list()
}

/// Kill a background task by id. Returns false if there's no such task.
#[tauri::command]
fn bg_stop(id: u64) -> bool {
    background_manager().stop(id)
}

// ── session notes ────────────────────────────────────────────────────────────
fn open_notes() -> Result<Notes, String> {
    Notes::open(std::path::Path::new(&project_dir())).map_err(|e| e.to_string())
}

/// The agent's working-memory notes (oldest first), for the Notes browser.
#[tauri::command]
fn notes_list() -> Result<Vec<NoteRow>, String> {
    open_notes()?.list(200).map_err(|e| e.to_string())
}

/// Clear all recorded notes (the Notes browser's "Clear" action).
#[tauri::command]
fn notes_clear() -> Result<(), String> {
    open_notes()?.clear().map(|_| ()).map_err(|e| e.to_string())
}

/// Unified diff of `path` against git HEAD. For files git doesn't track yet
/// (e.g. ones the agent just created) we synthesize an all-added diff from the
/// working-tree content, so the Changes view can still show them. Returns an
/// empty string when there is nothing to show.
#[tauri::command]
fn git_diff(path: String) -> Result<String, String> {
    let dir = project_dir();
    let run = |args: &[&str]| Command::new("git").args(args).output();

    // Tracked file: ask git for the real diff against HEAD.
    let tracked = run(&["-C", &dir, "ls-files", "--error-unmatch", "--", &path])
        .map(|o| o.status.success())
        .unwrap_or(false);
    if tracked {
        let out = run(&["-C", &dir, "diff", "HEAD", "--no-color", "--", &path])
            .map_err(|e| format!("git not available: {e}"))?;
        return Ok(String::from_utf8_lossy(&out.stdout).into_owned());
    }

    // Untracked: synthesize a diff showing every line as an addition.
    let resolved = ctx().resolve(&path).map_err(|e| e.to_string())?;
    let content = std::fs::read_to_string(&resolved).map_err(|e| e.to_string())?;
    let lines: Vec<&str> = content.lines().collect();
    let mut diff = format!(
        "--- /dev/null\n+++ b/{path}\n@@ -0,0 +1,{} @@\n",
        lines.len()
    );
    for l in lines {
        diff.push('+');
        diff.push_str(l);
        diff.push('\n');
    }
    Ok(diff)
}

// ── interactive terminal ─────────────────────────────────────────────────────
#[derive(Serialize)]
struct CommandResult {
    stdout: String,
    stderr: String,
    code: i32,
}

/// Run a shell command the user typed into the terminal, in the project dir.
/// User-initiated (not the agent), so it isn't gated by `confirm_commands`.
#[tauri::command]
async fn run_command(command: String) -> Result<CommandResult, String> {
    let dir = project_dir();
    #[cfg(target_os = "windows")]
    let mut cmd = {
        let mut c = tokio::process::Command::new("cmd");
        c.args(["/C", &command]);
        c
    };
    #[cfg(not(target_os = "windows"))]
    let mut cmd = {
        let mut c = tokio::process::Command::new("sh");
        c.args(["-c", &command]);
        c
    };
    cmd.current_dir(&dir);

    let out = tokio::time::timeout(std::time::Duration::from_secs(60), cmd.output())
        .await
        .map_err(|_| "command timed out after 60s".to_string())?
        .map_err(|e| e.to_string())?;
    Ok(CommandResult {
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        code: out.status.code().unwrap_or(-1),
    })
}

#[tauri::command]
fn project_path() -> String {
    project_dir()
}

// ── external links ──────────────────────────────────────────────────────────
#[tauri::command]
async fn open_external_url(url: String) -> Result<(), String> {
    if !(url.starts_with("https://") || url.starts_with("http://")) {
        return Err("Only http/https URLs can be opened.".to_string());
    }
    tauri::async_runtime::spawn_blocking(move || open_url_with_system(&url))
        .await
        .map_err(|err| format!("Could not open link: {err}"))?
}

fn open_url_with_system(url: &str) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    let status = Command::new("cmd").args(["/C", "start", "", url]).status();
    #[cfg(target_os = "macos")]
    let status = Command::new("open").arg(url).status();
    #[cfg(all(unix, not(target_os = "macos")))]
    let status = Command::new("xdg-open").arg(url).status();

    let status = status.map_err(|e| format!("Could not invoke browser: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("Browser rejected: {status}"))
    }
}

/// "About" panel metadata, shown by the macOS app menu and the Help menu.
fn about_metadata() -> AboutMetadata<'static> {
    AboutMetadata {
        name: Some("Infinidev".into()),
        version: Some(env!("CARGO_PKG_VERSION").into()),
        comments: Some("Autonomous coding agent — local, embedded Rust engine.".into()),
        ..Default::default()
    }
}

/// Build the native application menu. Custom items emit `menu://action` with
/// their id (handled in the webview); predefined items (clipboard, window,
/// quit) are handled by the OS. On macOS the first submenu becomes the bold app
/// menu and Settings lives there; other platforms put Settings/Quit under File
/// and About under Help.
fn build_menu<R: Runtime>(app: &AppHandle<R>) -> tauri::Result<Menu<R>> {
    let open_folder = MenuItem::with_id(app, "open_folder", "Open Folder…", true, Some("CmdOrCtrl+O"))?;
    let new_session = MenuItem::with_id(app, "new_session", "New Session", true, Some("CmdOrCtrl+N"))?;
    let settings = MenuItem::with_id(app, "settings", "Settings…", true, None::<&str>)?;
    let find = MenuItem::with_id(app, "search", "Find in Project…", true, Some("CmdOrCtrl+K"))?;
    let show_activity = MenuItem::with_id(app, "show_activity", "Activity", true, None::<&str>)?;
    let show_changes = MenuItem::with_id(app, "show_changes", "Changes", true, None::<&str>)?;
    let show_files = MenuItem::with_id(app, "show_files", "Files", true, None::<&str>)?;
    let show_terminal = MenuItem::with_id(app, "show_terminal", "Terminal", true, None::<&str>)?;
    let show_knowledge = MenuItem::with_id(app, "show_knowledge", "Knowledge", true, None::<&str>)?;
    let show_notes = MenuItem::with_id(app, "show_notes", "Notes", true, None::<&str>)?;
    let show_background =
        MenuItem::with_id(app, "show_background", "Background Tasks", true, None::<&str>)?;
    let show_docs = MenuItem::with_id(app, "show_docs", "Documentation", true, None::<&str>)?;
    let show_debug = MenuItem::with_id(app, "show_debug", "Debug…", true, None::<&str>)?;
    let stop = MenuItem::with_id(app, "stop", "Stop Turn", true, Some("CmdOrCtrl+."))?;

    let edit = Submenu::with_items(
        app,
        "Edit",
        true,
        &[
            &PredefinedMenuItem::undo(app, None)?,
            &PredefinedMenuItem::redo(app, None)?,
            &PredefinedMenuItem::separator(app)?,
            &PredefinedMenuItem::cut(app, None)?,
            &PredefinedMenuItem::copy(app, None)?,
            &PredefinedMenuItem::paste(app, None)?,
            &PredefinedMenuItem::select_all(app, None)?,
            &PredefinedMenuItem::separator(app)?,
            &find,
        ],
    )?;

    let view = Submenu::with_items(
        app,
        "View",
        true,
        &[
            &show_activity,
            &show_changes,
            &show_files,
            &show_terminal,
            &PredefinedMenuItem::separator(app)?,
            &show_knowledge,
            &show_notes,
            &show_background,
            &show_docs,
            &show_debug,
        ],
    )?;

    let agent = Submenu::with_items(app, "Agent", true, &[&stop])?;

    #[cfg(target_os = "macos")]
    {
        let app_menu = Submenu::with_items(
            app,
            "Infinidev",
            true,
            &[
                &PredefinedMenuItem::about(app, Some("About Infinidev"), Some(about_metadata()))?,
                &PredefinedMenuItem::separator(app)?,
                &settings,
                &PredefinedMenuItem::separator(app)?,
                &PredefinedMenuItem::services(app, None)?,
                &PredefinedMenuItem::separator(app)?,
                &PredefinedMenuItem::hide(app, None)?,
                &PredefinedMenuItem::hide_others(app, None)?,
                &PredefinedMenuItem::show_all(app, None)?,
                &PredefinedMenuItem::separator(app)?,
                &PredefinedMenuItem::quit(app, None)?,
            ],
        )?;
        let file = Submenu::with_items(
            app,
            "File",
            true,
            &[
                &open_folder,
                &new_session,
                &PredefinedMenuItem::separator(app)?,
                &PredefinedMenuItem::close_window(app, None)?,
            ],
        )?;
        Menu::with_items(app, &[&app_menu, &file, &edit, &view, &agent])
    }
    #[cfg(not(target_os = "macos"))]
    {
        let file = Submenu::with_items(
            app,
            "File",
            true,
            &[
                &open_folder,
                &new_session,
                &PredefinedMenuItem::separator(app)?,
                &settings,
                &PredefinedMenuItem::separator(app)?,
                &PredefinedMenuItem::quit(app, None)?,
            ],
        )?;
        let help = Submenu::with_items(
            app,
            "Help",
            true,
            &[&PredefinedMenuItem::about(app, Some("About Infinidev"), Some(about_metadata()))?],
        )?;
        Menu::with_items(app, &[&file, &edit, &view, &agent, &help])
    }
}

/// Custom menu ids forwarded to the webview as `menu://action` payloads.
const MENU_ACTIONS: &[&str] = &[
    "open_folder",
    "new_session",
    "settings",
    "search",
    "show_activity",
    "show_changes",
    "show_files",
    "show_terminal",
    "show_knowledge",
    "show_notes",
    "show_background",
    "show_docs",
    "show_debug",
    "stop",
];

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(Session::default())
        .menu(|app| build_menu(app))
        .on_menu_event(|app, event| {
            let id = event.id().0.as_str();
            if MENU_ACTIONS.contains(&id) {
                let _ = app.emit("menu://action", id);
            }
        })
        .setup(|app| {
            // Resolve the config file, load any persisted settings into the
            // managed Session, and remember the path for future writes.
            if let Ok(dir) = app.path().app_config_dir() {
                let _ = std::fs::create_dir_all(&dir);
                let path = dir.join("config.json");
                let state = app.state::<Session>();
                if let Ok(text) = std::fs::read_to_string(&path) {
                    if let Ok(cfg) = serde_json::from_str::<EngineConfig>(&text) {
                        *state.config.lock().unwrap() = cfg;
                    }
                }
                *state.config_path.lock().unwrap() = Some(path);

                // Restore the prior conversation, if any, so the session resumes.
                let hpath = dir.join("history.json");
                if let Ok(text) = std::fs::read_to_string(&hpath) {
                    if let Ok(hist) = serde_json::from_str::<Vec<Message>>(&text) {
                        *state.history.lock().unwrap() = hist;
                    }
                }
                *state.history_path.lock().unwrap() = Some(hpath);
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            engine_send,
            cancel_turn,
            answer_prompt,
            reset_session,
            get_history,
            set_config,
            get_config,
            list_provider_models,
            providers,
            fs_tree,
            fs_read,
            fs_write,
            fs_create,
            fs_rename,
            fs_delete,
            project_search,
            knowledge_list,
            knowledge_search,
            knowledge_record,
            notes_list,
            notes_clear,
            bg_list,
            bg_stop,
            docs_libraries,
            docs_sections,
            docs_read,
            docs_search,
            docs_fetch,
            docs_delete,
            git_diff,
            run_command,
            project_path,
            set_project,
            open_external_url,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
