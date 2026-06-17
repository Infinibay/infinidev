//! The engine's event model — the single channel through which the loop talks
//! to a UI. Mirrors the Python `OrchestrationHooks` + `event_bus`, but as one
//! serializable enum so the Tauri layer can forward each event to the WebView
//! verbatim, and the React store can switch on `type`.

use async_trait::async_trait;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EngineEvent {
    /// Pipeline phase changed: "working" | "idle" | (future: "planning"/"review").
    Phase { phase: String },
    /// The chat-agent tier decided how to handle the turn: `kind` is
    /// "respond" (answered conversationally, turn ends) or "escalate" (handed
    /// off to the developer loop). `detail` carries the agent's understanding
    /// of the request on escalate.
    Decision { kind: String, detail: String },
    /// The planning preamble produced a plan for the task.
    Plan { steps: Vec<String> },
    /// A new loop iteration began.
    StepStart { step: u32, max: u32 },
    /// A fragment of the assistant's streamed answer.
    StreamChunk { chunk: String },
    /// A fragment of streamed reasoning / chain-of-thought.
    Reasoning { chunk: String },
    /// A complete non-streamed message (system/status/intermediate).
    Notify { speaker: String, text: String, kind: String },
    /// The model requested a tool call.
    ToolCall { id: String, name: String, args: serde_json::Value },
    /// A tool finished.
    ToolResult { id: String, name: String, ok: bool, output: String },
    /// A file was created/edited by a tool (drives the Changes view).
    FileChange { path: String, action: String },
    /// Token accounting + computed cost for the last LLM call.
    Usage {
        prompt_tokens: u64,
        completion_tokens: u64,
        total_tokens: u64,
        cost_usd: Option<f64>,
    },
    /// The review tier's verdict on the developer's changes: `status` is
    /// "approve" or "changes"; `notes` lists the issues found (empty on approve).
    Review { status: String, notes: Vec<String> },
    /// The turn finished; `result` is the final answer.
    TurnEnd { result: String },
    /// A recoverable error during the turn.
    Error { message: String },
}

/// The host a turn runs against: receives events and answers questions.
/// Implemented by the Tauri layer (emits to the WebView) and by tests.
#[async_trait]
pub trait EngineHost: Send + Sync {
    fn emit(&self, event: EngineEvent);

    /// Block until the user answers, or `None` if non-interactive.
    async fn ask_user(&self, _prompt: String, _kind: String) -> Option<String> {
        None
    }
}
