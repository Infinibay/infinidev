use serde::{Deserialize, Serialize};

/// Engine configuration for a turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub provider: String,
    /// Bare model id sent to the provider API (no LiteLLM prefix), e.g. "gemma:7b".
    pub model: String,
    pub api_key: Option<String>,
    pub base_url: String,
    pub temperature: Option<f32>,
    pub max_iterations: u32,
    /// Run a planning preamble before executing (plan-execute-summarize).
    pub planning: bool,
    /// Route turns through the read-only chat-agent tier first, which decides
    /// respond-vs-escalate before any developer work. When false, every turn
    /// goes straight to the developer loop.
    #[serde(default = "default_true")]
    pub orchestrate: bool,
    /// After the developer loop changes files, run a critic review pass (with
    /// up to one fix iteration). Ignored when no files changed.
    #[serde(default = "default_true")]
    pub review: bool,
    /// Force prompt-based ("manual") tool calling even when the model advertises
    /// native function-calling. The engine otherwise auto-selects manual mode
    /// for models whose capability matrix reports no function-calling support.
    /// Useful for flaky local models that ignore the native tool slot.
    #[serde(default)]
    pub force_manual_tools: bool,
    /// Ask the host to confirm before running a shell command (`execute_command`).
    /// The host answers via `EngineHost::ask_user`; a non-affirmative answer (or
    /// no host) denies the command. Off by default so headless runs are unblocked.
    #[serde(default)]
    pub confirm_commands: bool,
    /// Ask the host to confirm before a file-writing tool (create/edit/delete)
    /// runs. Same answer contract as [`Self::confirm_commands`].
    #[serde(default)]
    pub confirm_writes: bool,
}

fn default_true() -> bool {
    true
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            provider: "ollama".to_string(),
            model: "qwen2.5-coder:7b".to_string(),
            api_key: None,
            base_url: String::new(),
            temperature: Some(0.2),
            max_iterations: 25,
            planning: true,
            orchestrate: true,
            review: true,
            force_manual_tools: false,
            confirm_commands: false,
            confirm_writes: false,
        }
    }
}

impl EngineConfig {
    /// Convenience for a local Ollama model.
    pub fn ollama(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            ..Default::default()
        }
    }
}
