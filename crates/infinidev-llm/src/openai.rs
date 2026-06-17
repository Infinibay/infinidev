//! OpenAI-compatible chat client (`POST {base}/chat/completions`).
//!
//! Covers the majority of providers: OpenAI, Ollama (via `/v1`), vLLM,
//! llama.cpp, Mistral, Z.AI, Kimi, MiniMax, OpenRouter, Qwen (DashScope
//! compat-mode), and any generic OpenAI-compatible endpoint.

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use serde::Deserialize;

use crate::client::LlmClient;
use crate::error::{LlmError, Result};
use crate::types::{ChatRequest, ChatResponse, StreamChunk, ToolCall, ToolCallDelta, Usage};

pub struct OpenAiCompatClient {
    http: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    headers: Vec<(String, String)>,
}

impl OpenAiCompatClient {
    pub fn new(
        base_url: impl Into<String>,
        api_key: Option<String>,
        headers: Vec<(String, String)>,
    ) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key,
            headers,
        }
    }

    fn post(&self, body: &ChatRequest) -> reqwest::RequestBuilder {
        let url = format!("{}/chat/completions", self.base_url);
        let mut rb = self.http.post(url).json(body);
        if let Some(key) = &self.api_key {
            rb = rb.bearer_auth(key);
        }
        for (k, v) in &self.headers {
            rb = rb.header(k.as_str(), v.as_str());
        }
        rb
    }
}

#[async_trait]
impl LlmClient for OpenAiCompatClient {
    fn wire_name(&self) -> &'static str {
        "openai-compat"
    }

    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse> {
        let mut body = req.clone();
        body.stream = false;
        let resp = self.post(&body).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let message = resp.text().await.unwrap_or_default();
            return Err(LlmError::Api { status: status.as_u16(), message: truncate(&message, 600) });
        }
        let wire: WireResp = resp.json().await?;
        let model = if wire.model.is_empty() { req.model.clone() } else { wire.model };
        let usage = wire.usage.map(map_usage).unwrap_or_default();
        let choice = wire.choices.into_iter().next().unwrap_or_default();
        let msg = choice.message;
        Ok(ChatResponse {
            model,
            content: msg.content,
            reasoning: msg.reasoning_content.or(msg.reasoning),
            tool_calls: msg.tool_calls.unwrap_or_default(),
            finish_reason: choice.finish_reason,
            usage,
        })
    }

    async fn chat_stream(
        &self,
        req: &ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk>>> {
        let mut body = req.clone();
        body.stream = true;
        // Ask OpenAI-style servers to include usage in the final chunk.
        body.extra
            .entry("stream_options".to_string())
            .or_insert_with(|| serde_json::json!({ "include_usage": true }));

        let resp = self.post(&body).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let message = resp.text().await.unwrap_or_default();
            return Err(LlmError::Api { status: status.as_u16(), message: truncate(&message, 600) });
        }

        let byte_stream = resp.bytes_stream();
        let s = async_stream::try_stream! {
            futures::pin_mut!(byte_stream);
            let mut buf = String::new();
            while let Some(bytes) = byte_stream.next().await {
                let bytes = bytes?;
                buf.push_str(&String::from_utf8_lossy(&bytes));
                // SSE events are separated by a blank line.
                while let Some(idx) = buf.find("\n\n") {
                    let event = buf[..idx].to_string();
                    buf.drain(..idx + 2);
                    for line in event.lines() {
                        let line = line.trim_start();
                        let Some(data) = line.strip_prefix("data:") else { continue };
                        let data = data.trim();
                        if data == "[DONE]" {
                            return;
                        }
                        if data.is_empty() {
                            continue;
                        }
                        if let Ok(wire) = serde_json::from_str::<WireStreamResp>(data) {
                            if let Some(chunk) = map_stream(wire) {
                                yield chunk;
                            }
                        }
                    }
                }
            }
        };
        Ok(Box::pin(s))
    }
}

fn truncate(s: &str, max: usize) -> String {
    let flat: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.len() <= max {
        flat
    } else {
        format!("{}…", &flat[..max])
    }
}

fn map_usage(w: WireUsage) -> Usage {
    let total = if w.total_tokens > 0 {
        w.total_tokens
    } else {
        w.prompt_tokens + w.completion_tokens
    };
    Usage {
        prompt_tokens: w.prompt_tokens,
        completion_tokens: w.completion_tokens,
        total_tokens: total,
        cache_read_tokens: w.prompt_tokens_details.and_then(|d| d.cached_tokens),
        cache_creation_tokens: None,
    }
}

fn map_stream(wire: WireStreamResp) -> Option<StreamChunk> {
    let usage = wire.usage.map(map_usage);
    let mut chunk = StreamChunk { usage, ..Default::default() };
    if let Some(choice) = wire.choices.into_iter().next() {
        chunk.finish_reason = choice.finish_reason;
        chunk.content = choice.delta.content;
        chunk.reasoning = choice.delta.reasoning_content.or(choice.delta.reasoning);
        if let Some(tc) = choice.delta.tool_calls.and_then(|v| v.into_iter().next()) {
            let (name, args) = match tc.function {
                Some(f) => (f.name, f.arguments),
                None => (None, None),
            };
            chunk.tool_call = Some(ToolCallDelta {
                index: tc.index,
                id: tc.id,
                name,
                arguments_fragment: args,
            });
        }
    }
    let empty = chunk.content.is_none()
        && chunk.reasoning.is_none()
        && chunk.tool_call.is_none()
        && chunk.finish_reason.is_none()
        && chunk.usage.is_none();
    if empty {
        None
    } else {
        Some(chunk)
    }
}

// ── wire structs (private) ────────────────────────────────────────────────
#[derive(Deserialize)]
struct WireResp {
    #[serde(default)]
    model: String,
    #[serde(default)]
    choices: Vec<WireChoice>,
    #[serde(default)]
    usage: Option<WireUsage>,
}

#[derive(Deserialize, Default)]
struct WireChoice {
    #[serde(default)]
    message: WireMsg,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize, Default)]
struct WireMsg {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Deserialize)]
struct WireUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    total_tokens: u64,
    #[serde(default)]
    prompt_tokens_details: Option<PromptDetails>,
}

#[derive(Deserialize, Default)]
struct PromptDetails {
    #[serde(default)]
    cached_tokens: Option<u64>,
}

#[derive(Deserialize)]
struct WireStreamResp {
    #[serde(default)]
    choices: Vec<WireStreamChoice>,
    #[serde(default)]
    usage: Option<WireUsage>,
}

#[derive(Deserialize)]
struct WireStreamChoice {
    #[serde(default)]
    delta: WireDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize, Default)]
struct WireDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<WireToolCallDelta>>,
}

#[derive(Deserialize)]
struct WireToolCallDelta {
    #[serde(default)]
    index: u32,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<WireFnDelta>,
}

#[derive(Deserialize, Default)]
struct WireFnDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}
