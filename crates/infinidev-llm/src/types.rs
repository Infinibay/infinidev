//! Provider-agnostic chat types. The wire shape here follows the OpenAI
//! chat-completions schema (the lingua franca most providers speak); the
//! per-provider clients translate to/from their own dialect where it differs
//! (e.g. Anthropic Messages, Gemini generateContent).

use serde::{Deserialize, Serialize};

fn is_false(b: &bool) -> bool {
    !*b
}

/// Split a `data:<mime>;base64,<data>` URL into `(mime, base64)`. Used by the
/// Anthropic/Gemini clients, which carry images as raw base64 + media type
/// rather than the OpenAI `image_url` data-URL form.
pub fn split_data_url(url: &str) -> Option<(String, String)> {
    let rest = url.strip_prefix("data:")?;
    let (meta, data) = rest.split_once(',')?;
    let mime = meta.split(';').next().unwrap_or("image/png").to_string();
    Some((mime, data.to_string()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// A single chat message.
///
/// Serde is hand-written (not derived) so multimodal messages round-trip
/// through the OpenAI wire shape: when [`images`](Self::images) is non-empty,
/// `content` is emitted as an array of `text`/`image_url` parts; otherwise it
/// is a plain string exactly as before. The Anthropic/Gemini clients build
/// their own bodies and read [`images`](Self::images) directly.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: Option<String>,
    pub name: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub tool_call_id: Option<String>,
    /// Reasoning / chain-of-thought lifted out of `content` by providers that
    /// emit it separately (DeepSeek, MiniMax, Qwen with reasoning_split).
    pub reasoning_content: Option<String>,
    /// Attached images as `data:` URLs (base64). Empty for ordinary messages.
    pub images: Vec<String>,
}

impl Serialize for Message {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = ser.serialize_map(None)?;
        map.serialize_entry("role", &self.role)?;
        if self.images.is_empty() {
            if let Some(c) = &self.content {
                map.serialize_entry("content", c)?;
            }
        } else {
            // OpenAI multimodal: content becomes a parts array.
            let mut parts: Vec<serde_json::Value> = Vec::new();
            if let Some(c) = &self.content {
                if !c.is_empty() {
                    parts.push(serde_json::json!({ "type": "text", "text": c }));
                }
            }
            for img in &self.images {
                parts.push(serde_json::json!({ "type": "image_url", "image_url": { "url": img } }));
            }
            map.serialize_entry("content", &parts)?;
        }
        if let Some(n) = &self.name {
            map.serialize_entry("name", n)?;
        }
        if let Some(tc) = &self.tool_calls {
            map.serialize_entry("tool_calls", tc)?;
        }
        if let Some(id) = &self.tool_call_id {
            map.serialize_entry("tool_call_id", id)?;
        }
        if let Some(r) = &self.reasoning_content {
            map.serialize_entry("reasoning_content", r)?;
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for Message {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct ImageUrl {
            url: String,
        }
        #[derive(Deserialize)]
        struct Part {
            #[serde(rename = "type")]
            kind: String,
            #[serde(default)]
            text: Option<String>,
            #[serde(default)]
            image_url: Option<ImageUrl>,
        }
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum ContentField {
            Text(String),
            Parts(Vec<Part>),
        }
        #[derive(Deserialize)]
        struct Shadow {
            role: Role,
            #[serde(default)]
            content: Option<ContentField>,
            #[serde(default)]
            name: Option<String>,
            #[serde(default)]
            tool_calls: Option<Vec<ToolCall>>,
            #[serde(default)]
            tool_call_id: Option<String>,
            #[serde(default)]
            reasoning_content: Option<String>,
        }
        let s = Shadow::deserialize(de)?;
        let (content, images) = match s.content {
            None => (None, Vec::new()),
            Some(ContentField::Text(t)) => (Some(t), Vec::new()),
            Some(ContentField::Parts(parts)) => {
                let mut text = String::new();
                let mut imgs = Vec::new();
                for p in parts {
                    if p.kind == "image_url" {
                        if let Some(iu) = p.image_url {
                            imgs.push(iu.url);
                        }
                    } else if let Some(t) = p.text {
                        if !text.is_empty() {
                            text.push('\n');
                        }
                        text.push_str(&t);
                    }
                }
                (if text.is_empty() { None } else { Some(text) }, imgs)
            }
        };
        Ok(Message {
            role: s.role,
            content,
            name: s.name,
            tool_calls: s.tool_calls,
            tool_call_id: s.tool_call_id,
            reasoning_content: s.reasoning_content,
            images,
        })
    }
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self::text(Role::System, content)
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self::text(Role::User, content)
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::text(Role::Assistant, content)
    }
    /// A user message with attached images (`data:` URLs).
    pub fn user_with_images(content: impl Into<String>, images: Vec<String>) -> Self {
        Self {
            role: Role::User,
            content: Some(content.into()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
            images,
        }
    }
    /// A tool-result message answering a prior tool call.
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: Some(content.into()),
            name: None,
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            reasoning_content: None,
            images: Vec::new(),
        }
    }
    /// A tool-result message that also carries the tool's function name.
    /// OpenAI correlates results to calls by `tool_call_id`; Gemini correlates
    /// by function name (it has no call ids), so the engine records both.
    pub fn tool_named(
        tool_call_id: impl Into<String>,
        name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            role: Role::Tool,
            content: Some(content.into()),
            name: Some(name.into()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            reasoning_content: None,
            images: Vec::new(),
        }
    }
    fn text(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: Some(content.into()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
            images: Vec::new(),
        }
    }
}

/// A tool call emitted by the model (final, non-streaming form).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", default = "function_kind")]
    pub kind: String,
    pub function: FunctionCall,
}

fn function_kind() -> String {
    "function".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    /// JSON-encoded arguments string (OpenAI sends this as a string, not an object).
    pub arguments: String,
}

/// A tool the model may call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionDef,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema for the parameters.
    pub parameters: serde_json::Value,
}

impl Tool {
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            kind: "function".to_string(),
            function: FunctionDef {
                name: name.into(),
                description: Some(description.into()),
                parameters,
            },
        }
    }
}

/// Token accounting returned by the provider.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
    /// Cached prompt tokens (cheaper). Providers nest this differently; the
    /// clients normalize it here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_tokens: Option<u64>,
}

/// A chat request. `extra` is flattened into the request body to carry
/// provider-specific knobs (e.g. `enable_thinking`, `reasoning_split`).
#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "is_false")]
    pub stream: bool,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl ChatRequest {
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            messages,
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            stream: false,
            extra: serde_json::Map::new(),
        }
    }
    pub fn with_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }
    pub fn with_max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }
    pub fn streaming(mut self, on: bool) -> Self {
        self.stream = on;
        self
    }
    /// Set a provider-specific top-level body parameter.
    pub fn with_extra(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra.insert(key.into(), value);
        self
    }
}

/// A normalized (provider-agnostic) chat response.
#[derive(Debug, Clone, Default)]
pub struct ChatResponse {
    pub model: String,
    pub content: Option<String>,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: Option<String>,
    pub usage: Usage,
}

/// An incremental streaming delta.
#[derive(Debug, Clone, Default)]
pub struct StreamChunk {
    pub content: Option<String>,
    pub reasoning: Option<String>,
    pub tool_call: Option<ToolCallDelta>,
    pub finish_reason: Option<String>,
    pub usage: Option<Usage>,
}

/// A partial tool call arriving over a stream; fragments accumulate by `index`.
#[derive(Debug, Clone, Default)]
pub struct ToolCallDelta {
    pub index: u32,
    pub id: Option<String>,
    pub name: Option<String>,
    pub arguments_fragment: Option<String>,
}

#[cfg(test)]
mod message_serde_tests {
    use super::*;

    #[test]
    fn text_only_message_is_unchanged_wire_shape() {
        let m = Message::user("hello");
        let v = serde_json::to_value(&m).unwrap();
        assert_eq!(v, serde_json::json!({ "role": "user", "content": "hello" }));
        // No stray "images" key on ordinary messages.
        assert!(v.get("images").is_none());
    }

    #[test]
    fn image_message_emits_openai_parts() {
        let url = "data:image/png;base64,AAAA".to_string();
        let m = Message::user_with_images("look", vec![url.clone()]);
        let v = serde_json::to_value(&m).unwrap();
        let parts = v["content"].as_array().unwrap();
        assert_eq!(parts[0], serde_json::json!({ "type": "text", "text": "look" }));
        assert_eq!(parts[1], serde_json::json!({ "type": "image_url", "image_url": { "url": url } }));
    }

    #[test]
    fn multimodal_round_trips() {
        let m = Message::user_with_images("hi", vec!["data:image/jpeg;base64,ZZ".into()]);
        let s = serde_json::to_string(&m).unwrap();
        let back: Message = serde_json::from_str(&s).unwrap();
        assert_eq!(back.content.as_deref(), Some("hi"));
        assert_eq!(back.images, vec!["data:image/jpeg;base64,ZZ".to_string()]);
    }

    #[test]
    fn split_data_url_parses_mime_and_data() {
        let (mime, data) = split_data_url("data:image/png;base64,SGVsbG8=").unwrap();
        assert_eq!(mime, "image/png");
        assert_eq!(data, "SGVsbG8=");
        assert!(split_data_url("https://x/y.png").is_none());
    }
}
