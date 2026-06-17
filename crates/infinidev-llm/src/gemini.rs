//! Google Gemini client (`generateContent` / `streamGenerateContent`).
//!
//! Gemini's dialect differs from OpenAI's in several load-bearing ways, all
//! handled here:
//!   - The model id lives in the URL path, not the body.
//!   - There is no `system` / `assistant` / `tool` role: system text goes in a
//!     top-level `systemInstruction`, the model speaks as `model`, and tool
//!     results come back as a `user` turn carrying `functionResponse` parts.
//!   - Tool calls/results correlate by **function name**, not call id (Gemini
//!     has no call ids), so we read the name the engine stamped onto the
//!     tool-result message via [`Message::tool_named`].
//!   - `functionResponse.response` must be a JSON object, so string tool output
//!     is wrapped as `{ "result": … }`.
//!   - Consecutive same-role turns are merged — Gemini rejects two `user`
//!     contents in a row, which our one-message-per-tool-result loop produces.
//!   - Tool parameter schemas are sanitized to Gemini's OpenAPI subset.

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::client::LlmClient;
use crate::error::{LlmError, Result};
use crate::provider;
use crate::types::{
    ChatRequest, ChatResponse, FunctionCall, Role, StreamChunk, ToolCall, ToolCallDelta, Usage,
};

pub struct GeminiClient {
    http: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    headers: Vec<(String, String)>,
}

impl GeminiClient {
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

    fn url(&self, model: &str, method: &str, query: &str) -> String {
        let m = provider::bare_model(model);
        format!("{}/v1beta/models/{m}:{method}{query}", self.base_url)
    }

    fn auth(&self, mut rb: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(key) = &self.api_key {
            rb = rb.header("x-goog-api-key", key);
        }
        for (h, v) in &self.headers {
            rb = rb.header(h.as_str(), v.as_str());
        }
        rb
    }
}

/// Build the Gemini `generateContent` body from a provider-agnostic request.
pub(crate) fn to_gemini_body(req: &ChatRequest) -> Value {
    let mut system = String::new();
    let mut contents: Vec<Value> = Vec::new();

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
                    contents.push(json!({
                        "role": "user",
                        "parts": [{ "text": m.content.clone().unwrap_or_default() }],
                    }));
                } else {
                    let mut parts: Vec<Value> = Vec::new();
                    if let Some(c) = &m.content {
                        if !c.is_empty() {
                            parts.push(json!({ "text": c }));
                        }
                    }
                    for img in &m.images {
                        if let Some((mime, data)) = crate::types::split_data_url(img) {
                            parts.push(json!({ "inlineData": { "mimeType": mime, "data": data } }));
                        }
                    }
                    contents.push(json!({ "role": "user", "parts": parts }));
                }
            }
            Role::Assistant => {
                let mut parts: Vec<Value> = Vec::new();
                if let Some(c) = &m.content {
                    if !c.is_empty() {
                        parts.push(json!({ "text": c }));
                    }
                }
                if let Some(tcs) = &m.tool_calls {
                    for tc in tcs {
                        let args: Value = serde_json::from_str(&tc.function.arguments)
                            .unwrap_or_else(|_| json!({}));
                        parts.push(json!({
                            "functionCall": { "name": tc.function.name, "args": args },
                        }));
                    }
                }
                if parts.is_empty() {
                    parts.push(json!({ "text": "" }));
                }
                contents.push(json!({ "role": "model", "parts": parts }));
            }
            Role::Tool => {
                // Gemini correlates by function name (no call ids).
                let name = m.name.clone().unwrap_or_default();
                let response = json!({ "result": m.content.clone().unwrap_or_default() });
                contents.push(json!({
                    "role": "user",
                    "parts": [{ "functionResponse": { "name": name, "response": response } }],
                }));
            }
        }
    }

    let contents = merge_consecutive(contents);

    let mut body = json!({ "contents": contents });
    if !system.is_empty() {
        body["systemInstruction"] = json!({ "parts": [{ "text": system }] });
    }

    let mut gen_cfg = serde_json::Map::new();
    if let Some(t) = req.temperature {
        gen_cfg.insert("temperature".into(), json!(t));
    }
    gen_cfg.insert("maxOutputTokens".into(), json!(req.max_tokens.unwrap_or(4096)));
    body["generationConfig"] = Value::Object(gen_cfg);

    if let Some(tools) = &req.tools {
        if !tools.is_empty() {
            let decls: Vec<Value> = tools
                .iter()
                .map(|t| {
                    json!({
                        "name": t.function.name,
                        "description": t.function.description.clone().unwrap_or_default(),
                        "parameters": sanitize_schema(&t.function.parameters),
                    })
                })
                .collect();
            body["tools"] = json!([{ "functionDeclarations": decls }]);
        }
    }
    body
}

/// Fold consecutive same-role `contents` into one entry (concatenating their
/// `parts`). Gemini rejects two turns of the same role in a row, which our
/// per-tool-call result messages would otherwise produce.
fn merge_consecutive(contents: Vec<Value>) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::with_capacity(contents.len());
    for c in contents {
        if let Some(last) = out.last_mut() {
            if last.get("role") == c.get("role") {
                if let (Some(a), Some(b)) = (
                    last.get_mut("parts").and_then(|p| p.as_array_mut()),
                    c.get("parts").and_then(|p| p.as_array()),
                ) {
                    a.extend(b.iter().cloned());
                    continue;
                }
            }
        }
        out.push(c);
    }
    out
}

/// Strip JSON-Schema keywords Gemini's parameter schema (an OpenAPI 3.0 subset)
/// does not accept. Recurses through objects and arrays.
fn sanitize_schema(v: &Value) -> Value {
    match v {
        Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for (k, val) in map {
                if matches!(
                    k.as_str(),
                    "additionalProperties"
                        | "$schema"
                        | "$id"
                        | "$ref"
                        | "$defs"
                        | "definitions"
                        | "title"
                        | "default"
                        | "examples"
                ) {
                    continue;
                }
                out.insert(k.clone(), sanitize_schema(val));
            }
            Value::Object(out)
        }
        Value::Array(arr) => Value::Array(arr.iter().map(sanitize_schema).collect()),
        other => other.clone(),
    }
}

#[async_trait]
impl LlmClient for GeminiClient {
    fn wire_name(&self) -> &'static str {
        "gemini"
    }

    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse> {
        let body = to_gemini_body(req);
        let url = self.url(&req.model, "generateContent", "");
        let rb = self.auth(self.http.post(url).json(&body));
        let resp = rb.send().await?;
        let status = resp.status();
        if !status.is_success() {
            let message: String = resp.text().await.unwrap_or_default().chars().take(600).collect();
            return Err(LlmError::Api { status: status.as_u16(), message });
        }
        let wire: WireResp = resp.json().await?;

        let mut content = String::new();
        let mut tool_calls = Vec::new();
        if let Some(cand) = wire.candidates.into_iter().next() {
            for (i, part) in cand.content.parts.iter().enumerate() {
                if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                    content.push_str(text);
                } else if let Some(fc) = part.get("functionCall") {
                    tool_calls.push(function_call(fc, i));
                }
            }
        }
        let usage = wire.usage_metadata.map(map_usage).unwrap_or_default();
        Ok(ChatResponse {
            model: req.model.clone(),
            content: if content.is_empty() { None } else { Some(content) },
            reasoning: None,
            tool_calls,
            finish_reason: Some("stop".to_string()),
            usage,
        })
    }

    async fn chat_stream(&self, req: &ChatRequest) -> Result<BoxStream<'static, Result<StreamChunk>>> {
        let body = to_gemini_body(req);
        let url = self.url(&req.model, "streamGenerateContent", "?alt=sse");
        let rb = self.auth(self.http.post(url).json(&body));
        let resp = rb.send().await?;
        let status = resp.status();
        if !status.is_success() {
            let message: String = resp.text().await.unwrap_or_default().chars().take(600).collect();
            return Err(LlmError::Api { status: status.as_u16(), message });
        }

        let byte_stream = resp.bytes_stream();
        let s = async_stream::try_stream! {
            futures::pin_mut!(byte_stream);
            let mut buf = String::new();
            let mut fn_index: u32 = 0;
            while let Some(bytes) = byte_stream.next().await {
                let bytes = bytes?;
                buf.push_str(&String::from_utf8_lossy(&bytes));
                while let Some(idx) = buf.find("\n\n") {
                    let event = buf[..idx].to_string();
                    buf.drain(..idx + 2);
                    for line in event.lines() {
                        let line = line.trim_start();
                        let Some(data) = line.strip_prefix("data:") else { continue };
                        let data = data.trim();
                        if data.is_empty() {
                            continue;
                        }
                        let Ok(wire) = serde_json::from_str::<WireResp>(data) else { continue };
                        if let Some(u) = wire.usage_metadata {
                            yield StreamChunk { usage: Some(map_usage(u)), ..Default::default() };
                        }
                        if let Some(cand) = wire.candidates.into_iter().next() {
                            for part in &cand.content.parts {
                                if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                                    if !text.is_empty() {
                                        yield StreamChunk { content: Some(text.to_string()), ..Default::default() };
                                    }
                                } else if let Some(fc) = part.get("functionCall") {
                                    let tc = function_call(fc, fn_index as usize);
                                    yield StreamChunk {
                                        tool_call: Some(ToolCallDelta {
                                            index: fn_index,
                                            id: tc.id,
                                            name: Some(tc.function.name),
                                            arguments_fragment: Some(tc.function.arguments),
                                        }),
                                        ..Default::default()
                                    };
                                    fn_index += 1;
                                }
                            }
                        }
                    }
                }
            }
            yield StreamChunk { finish_reason: Some("stop".to_string()), ..Default::default() };
        };
        Ok(Box::pin(s))
    }
}

/// Map a Gemini `functionCall` part into a normalized [`ToolCall`].
fn function_call(fc: &Value, index: usize) -> ToolCall {
    let name = fc.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let arguments = fc
        .get("args")
        .map(|v| v.to_string())
        .unwrap_or_else(|| "{}".to_string());
    ToolCall {
        id: Some(format!("call_{index}")),
        kind: "function".to_string(),
        function: FunctionCall { name, arguments },
    }
}

fn map_usage(w: WireUsage) -> Usage {
    let total = if w.total_token_count > 0 {
        w.total_token_count
    } else {
        w.prompt_token_count + w.candidates_token_count
    };
    Usage {
        prompt_tokens: w.prompt_token_count,
        completion_tokens: w.candidates_token_count,
        total_tokens: total,
        cache_read_tokens: w.cached_content_token_count,
        cache_creation_tokens: None,
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct WireResp {
    #[serde(default)]
    candidates: Vec<WireCandidate>,
    #[serde(default)]
    usage_metadata: Option<WireUsage>,
}

#[derive(Deserialize)]
struct WireCandidate {
    #[serde(default)]
    content: WireContent,
}

#[derive(Deserialize, Default)]
struct WireContent {
    #[serde(default)]
    parts: Vec<Value>,
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct WireUsage {
    #[serde(default)]
    prompt_token_count: u64,
    #[serde(default)]
    candidates_token_count: u64,
    #[serde(default)]
    total_token_count: u64,
    #[serde(default)]
    cached_content_token_count: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Message, Tool};

    #[test]
    fn maps_system_tools_and_function_responses() {
        // Realistic flow: user → model(functionCall) → user(functionResponse).
        let mut call = Message::assistant("");
        call.content = None;
        call.tool_calls = Some(vec![ToolCall {
            id: Some("call_0".into()),
            kind: "function".into(),
            function: FunctionCall { name: "read_file".into(), arguments: "{}".into() },
        }]);
        let req = ChatRequest::new(
            "gemini/gemini-2.5-flash",
            vec![
                Message::system("be terse"),
                Message::user("hi"),
                call,
                Message::tool_named("call_0", "read_file", "result text"),
            ],
        )
        .with_tools(vec![Tool::function(
            "read_file",
            "read a file",
            serde_json::json!({ "type": "object", "additionalProperties": false }),
        )]);

        let body = to_gemini_body(&req);
        // system → systemInstruction, not a content turn
        assert_eq!(body["systemInstruction"]["parts"][0]["text"], "be terse");
        // first content is the user turn
        assert_eq!(body["contents"][0]["role"], "user");
        assert_eq!(body["contents"][0]["parts"][0]["text"], "hi");
        // model turn carries the functionCall
        assert_eq!(body["contents"][1]["role"], "model");
        // tool result → user turn with functionResponse carrying the name
        assert_eq!(body["contents"][2]["role"], "user");
        assert_eq!(
            body["contents"][2]["parts"][0]["functionResponse"]["name"],
            "read_file"
        );
        assert_eq!(
            body["contents"][2]["parts"][0]["functionResponse"]["response"]["result"],
            "result text"
        );
        // tools → functionDeclarations, schema sanitized
        assert_eq!(
            body["tools"][0]["functionDeclarations"][0]["name"],
            "read_file"
        );
        assert!(body["tools"][0]["functionDeclarations"][0]["parameters"]
            .get("additionalProperties")
            .is_none());
    }

    #[test]
    fn merges_consecutive_user_turns() {
        let req = ChatRequest::new(
            "gemini-2.5-flash",
            vec![
                Message::tool_named("c0", "a", "ra"),
                Message::tool_named("c1", "b", "rb"),
            ],
        );
        let body = to_gemini_body(&req);
        // two tool results collapse into one user turn with two parts
        assert_eq!(body["contents"].as_array().unwrap().len(), 1);
        assert_eq!(body["contents"][0]["role"], "user");
        assert_eq!(body["contents"][0]["parts"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn maps_assistant_tool_calls() {
        let mut tc_msg = Message::assistant("");
        tc_msg.content = None;
        tc_msg.tool_calls = Some(vec![ToolCall {
            id: Some("call_0".into()),
            kind: "function".into(),
            function: FunctionCall { name: "ls".into(), arguments: "{\"path\":\".\"}".into() },
        }]);
        let req = ChatRequest::new("gemini-2.5-flash", vec![Message::user("hi"), tc_msg]);
        let body = to_gemini_body(&req);
        assert_eq!(body["contents"][1]["role"], "model");
        assert_eq!(body["contents"][1]["parts"][0]["functionCall"]["name"], "ls");
        assert_eq!(body["contents"][1]["parts"][0]["functionCall"]["args"]["path"], ".");
    }
}
