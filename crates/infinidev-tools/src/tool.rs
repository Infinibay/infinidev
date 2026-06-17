use async_trait::async_trait;

use crate::context::ToolContext;
use crate::error::Result;

/// An executable agent tool. Each tool advertises a JSON-Schema for its
/// parameters (consumed by the LLM tool-calling layer) and runs against a
/// [`ToolContext`].
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    /// JSON Schema for the tool's parameters.
    fn parameters(&self) -> serde_json::Value;

    /// Read-only tools (no side effects) — the security boundary for the
    /// chat/analyst tiers, mirroring the Python `is_read_only` attribute.
    fn is_read_only(&self) -> bool {
        false
    }

    /// Run the tool. `args` is the JSON object the model produced.
    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<String>;

    /// The LLM-facing tool schema (`infinidev_llm::Tool`).
    fn schema(&self) -> infinidev_llm::Tool {
        infinidev_llm::Tool::function(self.name(), self.description(), self.parameters())
    }
}
