//! Anthropic Messages API client (`POST {base}/v1/messages`).
//!
//! Translates the OpenAI-shaped [`ChatRequest`] to Anthropic's dialect:
//! `system` is a top-level field (not a message), assistant tool calls become
//! `tool_use` content blocks, and tool results become `tool_result` blocks in
//! a user message. Streaming is synthesized from a single response (the engine
//! only needs the delta sequence), which keeps this robust without parsing
//! Anthropic's bespoke SSE event stream.

use async_trait::async_trait;
use futures::stream::BoxStream;
use serde::Deserialize;
use serde_json::json;

use crate::client::LlmClient;
use crate::error::{LlmError, Result};
use crate::types::{
    ChatRequest, ChatResponse, FunctionCall, Role, StreamChunk, ToolCall, ToolCallDelta, Usage,
};

const ANTHROPIC_VERSION: &str = "2023-06-01";

pub struct AnthropicClient {
    http: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    headers: Vec<(String, String)>,
}

impl AnthropicClient {
    pub fn new(base_url: impl Into<String>, api_key: Option<String>, headers: Vec<(String, String)>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key,
            headers,
        }
    }
}

/// Build the Anthropic Messages request body from a provider-agnostic request.
pub(crate) fn to_anthropic_body(req: &ChatRequest) -> serde_json::Value {
    let mut system = String::new();
    let mut messages: Vec<serde_json::Value> = Vec::new();

    for m in &req.messages {
        match m.role {
            Role::System => {
                if !system.is_empty() {
                    system.push('\n');
                }
                system.push_str(m.content.as_deref().unwrap_or(""));
            }
            Role::User => {
                if m.images.is_empty() {
                    messages.push(json!({ "role": "user", "content": m.content.clone().unwrap_or_default() }));
                } else {
                    // Multimodal: text block (if any) + base64 image blocks.
                    let mut blocks: Vec<serde_json::Value> = Vec::new();
                    if let Some(c) = &m.content {
                        if !c.is_empty() {
                            blocks.push(json!({ "type": "text", "text": c }));
                        }
                    }
                    for img in &m.images {
                        if let Some((mime, data)) = crate::types::split_data_url(img) {
                            blocks.push(json!({
                                "type": "image",
                                "source": { "type": "base64", "media_type": mime, "data": data },
                            }));
                        }
                    }
                    messages.push(json!({ "role": "user", "content": blocks }));
                }
            }
            Role::Assistant => {
                if let Some(tcs) = &m.tool_calls {
                    let mut blocks: Vec<serde_json::Value> = Vec::new();
                    if let Some(c) = &m.content {
                        if !c.is_empty() {
                            blocks.push(json!({ "type": "text", "text": c }));
                        }
                    }
                    for tc in tcs {
                        let input: serde_json::Value =
                            serde_json::from_str(&tc.function.arguments).unwrap_or_else(|_| json!({}));
                        blocks.push(json!({
                            "type": "tool_use",
                            "id": tc.id.clone().unwrap_or_default(),
                            "name": tc.function.name,
                            "input": input,
                        }));
                    }
                    messages.push(json!({ "role": "assistant", "content": blocks }));
                } else {
                    messages.push(json!({ "role": "assistant", "content": m.content.clone().unwrap_or_default() }));
                }
            }
            Role::Tool => {
                messages.push(json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id.clone().unwrap_or_default(),
                        "content": m.content.clone().unwrap_or_default(),
                    }],
                }));
            }
        }
    }

    let mut body = json!({
        "model": req.model,
        "max_tokens": req.max_tokens.unwrap_or(4096),
        "messages": messages,
    });
    if !system.is_empty() {
        body["system"] = json!(system);
    }
    if let Some(t) = req.temperature {
        body["temperature"] = json!(t);
    }
    if let Some(tools) = &req.tools {
        let arr: Vec<serde_json::Value> = tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.function.name,
                    "description": t.function.description,
                    "input_schema": t.function.parameters,
                })
            })
            .collect();
        body["tools"] = json!(arr);
    }
    body
}

#[async_trait]
impl LlmClient for AnthropicClient {
    fn wire_name(&self) -> &'static str {
        "anthropic"
    }

    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse> {
        let body = to_anthropic_body(req);
        let url = format!("{}/v1/messages", self.base_url);
        let mut rb = self.http.post(url).header("anthropic-version", ANTHROPIC_VERSION).json(&body);
        if let Some(key) = &self.api_key {
            rb = rb.header("x-api-key", key);
        }
        for (h, v) in &self.headers {
            rb = rb.header(h.as_str(), v.as_str());
        }
        let resp = rb.send().await?;
        let status = resp.status();
        if !status.is_success() {
            let message: String = resp.text().await.unwrap_or_default().chars().take(600).collect();
            return Err(LlmError::Api { status: status.as_u16(), message });
        }
        let wire: WireResp = resp.json().await?;

        let mut content = String::new();
        let mut tool_calls = Vec::new();
        for block in wire.content {
            match block.get("type").and_then(|t| t.as_str()) {
                Some("text") => content.push_str(block.get("text").and_then(|t| t.as_str()).unwrap_or("")),
                Some("tool_use") => {
                    tool_calls.push(ToolCall {
                        id: block.get("id").and_then(|v| v.as_str()).map(String::from),
                        kind: "function".to_string(),
                        function: FunctionCall {
                            name: block.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                            arguments: block.get("input").map(|v| v.to_string()).unwrap_or_else(|| "{}".to_string()),
                        },
                    });
                }
                _ => {}
            }
        }
        let usage = Usage {
            prompt_tokens: wire.usage.input_tokens,
            completion_tokens: wire.usage.output_tokens,
            total_tokens: wire.usage.input_tokens + wire.usage.output_tokens,
            cache_read_tokens: wire.usage.cache_read_input_tokens,
            cache_creation_tokens: wire.usage.cache_creation_input_tokens,
        };
        Ok(ChatResponse {
            model: req.model.clone(),
            content: if content.is_empty() { None } else { Some(content) },
            reasoning: None,
            tool_calls,
            finish_reason: wire.stop_reason,
            usage,
        })
    }

    async fn chat_stream(&self, req: &ChatRequest) -> Result<BoxStream<'static, Result<StreamChunk>>> {
        // Synthesize the delta sequence from one full response.
        let resp = self.chat(req).await?;
        let mut chunks: Vec<Result<StreamChunk>> = Vec::new();
        if let Some(c) = resp.content {
            chunks.push(Ok(StreamChunk { content: Some(c), ..Default::default() }));
        }
        for (i, tc) in resp.tool_calls.into_iter().enumerate() {
            chunks.push(Ok(StreamChunk {
                tool_call: Some(ToolCallDelta {
                    index: i as u32,
                    id: tc.id,
                    name: Some(tc.function.name),
                    arguments_fragment: Some(tc.function.arguments),
                }),
                ..Default::default()
            }));
        }
        chunks.push(Ok(StreamChunk {
            usage: Some(resp.usage),
            finish_reason: resp.finish_reason,
            ..Default::default()
        }));
        Ok(Box::pin(futures::stream::iter(chunks)))
    }
}

#[derive(Deserialize)]
struct WireResp {
    #[serde(default)]
    content: Vec<serde_json::Value>,
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    usage: WireUsage,
}

#[derive(Deserialize, Default)]
struct WireUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    cache_read_input_tokens: Option<u64>,
    #[serde(default)]
    cache_creation_input_tokens: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Message, Tool};

    #[test]
    fn maps_system_tools_and_tool_results() {
        let req = ChatRequest::new(
            "claude-sonnet-4-6",
            vec![
                Message::system("be terse"),
                Message::user("hi"),
                Message::tool("call_1", "result text"),
            ],
        )
        .with_tools(vec![Tool::function("read_file", "read a file", serde_json::json!({"type": "object"}))]);

        let body = to_anthropic_body(&req);
        assert_eq!(body["system"], "be terse");
        assert_eq!(body["max_tokens"], 4096);
        // tools use input_schema, not parameters
        assert_eq!(body["tools"][0]["name"], "read_file");
        assert!(body["tools"][0]["input_schema"].is_object());
        // the system message is NOT in messages; user + tool_result are
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][1]["content"][0]["type"], "tool_result");
        assert_eq!(body["messages"][1]["content"][0]["tool_use_id"], "call_1");
    }
}
