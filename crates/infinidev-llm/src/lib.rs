//! Infinidev LLM layer — provider registry, multi-provider chat clients,
//! capability matrix and pricing. The first crate of the Rust agent core.
//!
//! ```no_run
//! # async fn demo() -> infinidev_llm::Result<()> {
//! use infinidev_llm::{build_client, ChatRequest, Message};
//! let client = build_client("ollama", None, "http://localhost:11434")?;
//! let req = ChatRequest::new("gemma:7b", vec![Message::user("Hello!")]);
//! let resp = client.chat(&req).await?;
//! println!("{:?}", resp.content);
//! # Ok(()) }
//! ```

pub mod anthropic;
pub mod capabilities;
pub mod client;
pub mod discovery;
pub mod error;
pub mod gemini;
pub mod openai;
pub mod pricing;
pub mod provider;
pub mod types;

pub use capabilities::{for_model as capabilities_for, Capabilities};
pub use discovery::list_models;
pub use client::LlmClient;
pub use error::{LlmError, Result};
pub use anthropic::AnthropicClient;
pub use gemini::GeminiClient;
pub use openai::OpenAiCompatClient;
pub use pricing::{cost, pricing_for, Cost, Pricing};
pub use provider::{Provider, Wire, PROVIDERS};
pub use types::{
    ChatRequest, ChatResponse, FunctionCall, FunctionDef, Message, Role, StreamChunk, Tool,
    ToolCall, ToolCallDelta, Usage,
};

/// Build a chat client for a provider id, an optional API key, and a
/// (possibly user-overridden) base URL. Returns a boxed [`LlmClient`].
pub fn build_client(
    provider_id: &str,
    api_key: Option<String>,
    base_url: &str,
) -> Result<Box<dyn LlmClient>> {
    let provider = provider::get(provider_id)
        .ok_or_else(|| LlmError::Config(format!("unknown provider: {provider_id}")))?;
    let base = provider.chat_base_url(base_url);
    match provider.wire {
        Wire::OpenAiCompat => Ok(Box::new(OpenAiCompatClient::new(
            base,
            api_key,
            default_headers(),
        ))),
        Wire::Anthropic => Ok(Box::new(anthropic::AnthropicClient::new(
            base,
            api_key,
            default_headers(),
        ))),
        Wire::Gemini => Ok(Box::new(gemini::GeminiClient::new(
            base,
            api_key,
            default_headers(),
        ))),
    }
}

/// Client-identifying headers, mirroring the Python `get_litellm_params`.
fn default_headers() -> Vec<(String, String)> {
    let v = env!("CARGO_PKG_VERSION");
    vec![
        ("User-Agent".to_string(), format!("infinidev/{v}")),
        ("X-Client-Name".to_string(), "infinidev".to_string()),
        ("X-Client-Version".to_string(), v.to_string()),
    ]
}

/// Apply provider-specific request quirks (ported from `get_litellm_params`):
///  - Qwen3 on local OpenAI-compatible backends: disable the think pass so
///    tool calls emit to the structured slot.
///  - MiniMax: split reasoning out of `content` into `reasoning_content`.
pub fn apply_quirks(provider_id: &str, model: &str, req: &mut ChatRequest) {
    let m = model.to_lowercase();
    if matches!(provider_id, "llama_cpp" | "vllm" | "openai_compatible") && m.contains("qwen3") {
        req.extra.insert(
            "chat_template_kwargs".to_string(),
            serde_json::json!({ "enable_thinking": false }),
        );
    }
    if provider_id == "minimax" {
        req.extra
            .insert("reasoning_split".to_string(), serde_json::Value::Bool(true));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_all_providers() {
        assert_eq!(PROVIDERS.len(), 14);
        assert!(provider::get("ollama").is_some());
        assert!(provider::get("anthropic").is_some());
        assert!(provider::get("nope").is_none());
        assert_eq!(provider::get("anthropic").unwrap().wire, Wire::Anthropic);
        assert_eq!(provider::get("openai").unwrap().wire, Wire::OpenAiCompat);
    }

    #[test]
    fn strips_prefixes() {
        assert_eq!(provider::bare_model("ollama_chat/gemma:7b"), "gemma:7b");
        assert_eq!(provider::bare_model("custom_openai/qwen3-max"), "qwen3-max");
        assert_eq!(provider::bare_model("anthropic/claude-opus-4-0"), "claude-opus-4-0");
        assert_eq!(provider::bare_model("gpt-4o"), "gpt-4o");
        assert_eq!(provider::normalize_model("ollama/x"), "ollama_chat/x");
    }

    #[test]
    fn ollama_base_gets_v1() {
        let p = provider::get("ollama").unwrap();
        assert_eq!(p.chat_base_url("http://localhost:11434"), "http://localhost:11434/v1");
        assert_eq!(p.chat_base_url(""), "http://localhost:11434/v1");
        let o = provider::get("openai").unwrap();
        assert_eq!(o.chat_base_url(""), "https://api.openai.com/v1");
    }

    #[test]
    fn pricing_picks_longest_match() {
        assert_eq!(pricing_for("gpt-4o-mini").unwrap().input, 0.15);
        assert_eq!(pricing_for("gpt-4o").unwrap().input, 2.50);
        assert_eq!(pricing_for("openai/gpt-4o-mini").unwrap().output, 0.60);
        assert!(pricing_for("gemma:7b").is_none()); // local → no price
    }

    #[test]
    fn cost_subtracts_cached_tokens() {
        let usage = Usage {
            prompt_tokens: 1_000_000,
            completion_tokens: 1_000_000,
            total_tokens: 2_000_000,
            cache_read_tokens: Some(0),
            cache_creation_tokens: None,
        };
        let c = cost("minimax-m2.7", &usage).unwrap();
        // 0.30 input + 1.20 output
        assert!((c.total_usd - 1.50).abs() < 1e-9);
    }

    #[test]
    fn capabilities_reflect_provider_quirks() {
        assert!(capabilities_for("qwen", "qwen3-max").needs_schema_sanitization);
        assert!(!capabilities_for("ollama", "gemma:7b").tool_choice_required);
        assert!(capabilities_for("minimax", "MiniMax-M2.7").thinking_sections);
        assert!(capabilities_for("openai", "gpt-4o").vision);
    }

    #[test]
    fn request_serialization_omits_empties() {
        let req = ChatRequest::new("m", vec![Message::user("hi")]);
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"messages\""));
        assert!(!json.contains("\"stream\"")); // false → skipped
        assert!(!json.contains("\"tools\""));
        assert!(!json.contains("\"temperature\""));
    }

    #[test]
    fn quirks_apply() {
        let mut req = ChatRequest::new("MiniMax-M2.7", vec![Message::user("hi")]);
        apply_quirks("minimax", "MiniMax-M2.7", &mut req);
        assert_eq!(req.extra.get("reasoning_split"), Some(&serde_json::Value::Bool(true)));

        let mut req2 = ChatRequest::new("qwen3-coder", vec![Message::user("hi")]);
        apply_quirks("vllm", "qwen3-coder", &mut req2);
        assert!(req2.extra.contains_key("chat_template_kwargs"));
    }

    #[test]
    fn builds_clients() {
        assert!(build_client("ollama", None, "").is_ok());
        assert!(build_client("anthropic", Some("k".into()), "").is_ok());
        assert!(build_client("gemini", Some("k".into()), "").is_ok());
        assert!(matches!(build_client("nope", None, ""), Err(LlmError::Config(_))));
    }
}
