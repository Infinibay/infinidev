use std::collections::BTreeMap;

use futures::StreamExt;
use infinidev_llm::{
    apply_quirks, build_client, capabilities_for, cost, ChatRequest, FunctionCall, LlmClient,
    Message, Role, Tool as LlmTool, ToolCall, Usage,
};
use infinidev_tools::{tools_for, Tool, ToolContext, ToolError};

use crate::config::EngineConfig;
use crate::error::Result;
use crate::event::{EngineEvent, EngineHost};
use crate::prompt::{MANUAL_TOOLS_PREAMBLE, PLAN_PROMPT, SYSTEM_PROMPT};

const TOOL_OUTPUT_LIMIT: usize = 8000;
/// Plan-execute-**summarize**: keep the last N messages verbatim and trim
/// older tool outputs so context stays small for local models.
const RECENT_FULL_MESSAGES: usize = 6;
const OLD_TOOL_OUTPUT_CAP: usize = 240;

/// The agentic tool-calling loop: stream a completion, execute any tool calls,
/// feed results back, repeat until the model produces a final answer.
pub struct Engine {
    client: Box<dyn LlmClient>,
    tools: Vec<Box<dyn Tool>>,
    ctx: ToolContext,
    cfg: EngineConfig,
    /// Prompt-based tool calling instead of the native tool slot — see
    /// [`EngineConfig::force_manual_tools`] and the capability matrix.
    manual: bool,
}

/// Decide whether to use manual (prompt-based) tool calling: forced by config,
/// or auto-selected when the model's capability matrix reports no native
/// function-calling.
fn use_manual_tools(cfg: &EngineConfig) -> bool {
    cfg.force_manual_tools || !capabilities_for(&cfg.provider, &cfg.model).function_calling
}

impl Engine {
    /// Build an engine with a live client for the configured provider.
    pub fn new(cfg: EngineConfig, ctx: ToolContext) -> Result<Self> {
        let client = build_client(&cfg.provider, cfg.api_key.clone(), &cfg.base_url)?;
        let mut tools = tools_for(false);
        tools.extend(infinidev_knowledge::tools());
        tools.extend(infinidev_codeintel::tools());
        tools.extend(infinidev_tools::background_tools());
        let manual = use_manual_tools(&cfg);
        Ok(Self { client, tools, ctx, cfg, manual })
    }

    /// Build an engine with an injected client + tool set (tests, custom hosts).
    pub fn with_client(
        client: Box<dyn LlmClient>,
        tools: Vec<Box<dyn Tool>>,
        ctx: ToolContext,
        cfg: EngineConfig,
    ) -> Self {
        let manual = use_manual_tools(&cfg);
        Self { client, tools, ctx, cfg, manual }
    }

    fn tool_schemas(&self) -> Vec<LlmTool> {
        self.tools.iter().map(|t| t.schema()).collect()
    }

    fn find_tool(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.iter().find(|t| t.name() == name).map(|b| b.as_ref())
    }

    /// Execute one tool call, emitting `ToolCall`/`ToolResult`/`FileChange`
    /// events. Returns (id, name, ok, shown_output). Shared by the native and
    /// manual tool-calling paths.
    async fn execute_tool(&self, tc: &ToolCall, host: &dyn EngineHost) -> (String, String, bool, String) {
        let id = tc.id.clone().unwrap_or_default();
        let name = tc.function.name.clone();
        let args: serde_json::Value =
            serde_json::from_str(&tc.function.arguments).unwrap_or_else(|_| serde_json::json!({}));
        host.emit(EngineEvent::ToolCall { id: id.clone(), name: name.clone(), args: args.clone() });

        // Permission gate: confirm risky tools with the host before running
        // them. A non-affirmative answer (or no interactive host) denies it, and
        // the model sees the denial as the tool result so it can adapt.
        let is_write = matches!(
            name.as_str(),
            "create_file" | "write_file" | "replace_lines" | "multi_edit" | "delete_file"
        );
        let gate = if self.cfg.confirm_commands && name == "execute_command" {
            let cmd = args.get("command").and_then(|c| c.as_str()).unwrap_or_default();
            Some(format!("Allow Infinidev to run this shell command?\n\n{cmd}"))
        } else if self.cfg.confirm_writes && is_write {
            let path = args.get("path").and_then(|p| p.as_str()).unwrap_or("(unknown)");
            Some(format!("Allow Infinidev to modify a file?\n\n{name}: {path}"))
        } else {
            None
        };
        if let Some(prompt) = gate {
            let answer = host.ask_user(prompt, "permission".into()).await;
            let allowed = matches!(
                answer.as_deref().map(str::trim),
                Some("allow") | Some("yes") | Some("y")
            );
            if !allowed {
                let output = "Denied by the user.".to_string();
                let shown = truncate(&output, TOOL_OUTPUT_LIMIT);
                host.emit(EngineEvent::ToolResult {
                    id: id.clone(),
                    name: name.clone(),
                    ok: false,
                    output: shown.clone(),
                });
                return (id, name, false, shown);
            }
        }

        let result = match self.find_tool(&name) {
            Some(tool) => tool.execute(args.clone(), &self.ctx).await,
            None => Err(ToolError::NotFound(format!("unknown tool: {name}"))),
        };
        let (ok, output) = match result {
            Ok(o) => (true, o),
            Err(e) => (false, format!("ERROR: {e}")),
        };
        let shown = truncate(&output, TOOL_OUTPUT_LIMIT);
        host.emit(EngineEvent::ToolResult { id: id.clone(), name: name.clone(), ok, output: shown.clone() });

        if ok {
            if let Some(path) = args.get("path").and_then(|p| p.as_str()) {
                let action = match name.as_str() {
                    "create_file" => "create",
                    "replace_lines" | "multi_edit" => "edit",
                    _ => "",
                };
                if !action.is_empty() {
                    host.emit(EngineEvent::FileChange { path: path.to_string(), action: action.into() });
                }
            }
        }
        (id, name, ok, shown)
    }

    /// Run one user turn to completion. `history` carries prior turns (the
    /// caller owns the conversation). Returns the final assistant answer.
    pub async fn run_turn(
        &self,
        user_input: &str,
        history: &[Message],
        host: &dyn EngineHost,
    ) -> Result<String> {
        self.run_turn_images(user_input, &[], history, host).await
    }

    /// Like [`run_turn`](Self::run_turn) but the user message carries attached
    /// images (`data:` URLs) for vision-capable models.
    pub async fn run_turn_images(
        &self,
        user_input: &str,
        images: &[String],
        history: &[Message],
        host: &dyn EngineHost,
    ) -> Result<String> {
        host.emit(EngineEvent::Phase { phase: "working".into() });

        let schemas = self.tool_schemas();
        let mut messages: Vec<Message> = Vec::with_capacity(history.len() + 3);
        messages.push(Message::system(SYSTEM_PROMPT));
        if self.manual {
            messages.push(Message::system(manual_tools_prompt(&schemas)));
        }
        // Working memory: re-inject any notes the agent recorded (this or a
        // prior turn) so `record_note` actually steers behaviour, mirroring the
        // Python loop's session-notes context block.
        if let Some(notes) = self.load_notes() {
            messages.push(Message::system(format!(
                "Your notes so far (from `record_note`):\n{notes}"
            )));
        }
        messages.extend_from_slice(history);
        messages.push(if images.is_empty() {
            Message::user(user_input)
        } else {
            Message::user_with_images(user_input, images.to_vec())
        });

        // Plan-execute: a short planning preamble the UI surfaces, then the
        // loop works through it. Soft-fails — a bad/absent plan just means we
        // execute directly.
        if self.cfg.planning {
            if let Some(steps) = self.make_plan(user_input).await {
                host.emit(EngineEvent::Plan { steps: steps.clone() });
                let plan_text = steps
                    .iter()
                    .enumerate()
                    .map(|(i, s)| format!("{}. {}", i + 1, s))
                    .collect::<Vec<_>>()
                    .join("\n");
                messages.push(Message::system(format!(
                    "Plan for this task:\n{plan_text}\n\nWork through it using tools, then give a concise final answer."
                )));
            }
        }

        let mut final_text = String::new();

        for step in 1..=self.cfg.max_iterations {
            host.emit(EngineEvent::StepStart { step, max: self.cfg.max_iterations });

            let sent = compact_context(&messages, RECENT_FULL_MESSAGES, OLD_TOOL_OUTPUT_CAP);
            let mut req = ChatRequest::new(&self.cfg.model, sent);
            if !self.manual {
                // Native mode: advertise tools via the structured slot.
                req = req.with_tools(schemas.clone());
            }
            if let Some(t) = self.cfg.temperature {
                req = req.with_temperature(t);
            }
            apply_quirks(&self.cfg.provider, &self.cfg.model, &mut req);

            let (content, native_calls, usage) =
                stream_completion(self.client.as_ref(), &req, host).await?;

            if let Some(u) = &usage {
                let cost_usd = cost(&self.cfg.model, u).map(|c| c.total_usd);
                host.emit(EngineEvent::Usage {
                    prompt_tokens: u.prompt_tokens,
                    completion_tokens: u.completion_tokens,
                    total_tokens: u.total_tokens,
                    cost_usd,
                });
            }

            // In manual mode the tool calls live in the model's text, not the
            // native slot — parse them out of `content`.
            let tool_calls = if self.manual {
                parse_manual_tool_calls(&content)
            } else {
                native_calls
            };

            if tool_calls.is_empty() {
                final_text = content;
                break;
            }

            // Record the assistant's tool-calling turn. Native: the structured
            // tool_calls. Manual: the raw text (which holds the call blocks),
            // with no native tool_calls field.
            if self.manual {
                messages.push(Message::assistant(content));
            } else {
                messages.push(Message {
                    role: Role::Assistant,
                    content: if content.is_empty() { None } else { Some(content) },
                    name: None,
                    tool_calls: Some(tool_calls.clone()),
                    tool_call_id: None,
                    reasoning_content: None,
                    images: Vec::new(),
                });
            }

            if self.manual {
                // Feed results back as a single user message — manual-mode
                // backends don't expect role=tool turns.
                let mut results = String::new();
                for tc in &tool_calls {
                    let (_, name, ok, shown) = self.execute_tool(tc, host).await;
                    results.push_str(&format!(
                        "[tool {name} — {}]\n{shown}\n\n",
                        if ok { "ok" } else { "error" }
                    ));
                }
                messages.push(Message::user(format!(
                    "Tool results:\n\n{results}Continue with more `tool_call` blocks, or give your final answer if done."
                )));
            } else {
                for tc in &tool_calls {
                    let (id, name, _ok, shown) = self.execute_tool(tc, host).await;
                    messages.push(Message::tool_named(id, name, shown));
                }
            }
        }

        if final_text.is_empty() {
            final_text = "Reached the step limit without a final answer.".into();
        }
        host.emit(EngineEvent::Phase { phase: "idle".into() });
        host.emit(EngineEvent::TurnEnd { result: final_text.clone() });
        Ok(final_text)
    }

    /// Load the agent's recorded notes as a numbered block, or `None` if there
    /// are none. Best-effort — a DB error just means no notes are injected. A
    /// quick local-SQLite read; kept synchronous to avoid a tokio runtime dep.
    fn load_notes(&self) -> Option<String> {
        let rows = infinidev_knowledge::Notes::open(&self.ctx.workspace)
            .and_then(|n| n.list(50))
            .ok()?;
        if rows.is_empty() {
            return None;
        }
        let text = rows
            .iter()
            .enumerate()
            .map(|(i, r)| format!("{}. {}", i + 1, r.note))
            .collect::<Vec<_>>()
            .join("\n");
        Some(text)
    }

    /// Planning preamble: ask the model for a short step list (non-streaming).
    /// Returns `None` for trivial requests or on any failure.
    async fn make_plan(&self, user_input: &str) -> Option<Vec<String>> {
        let messages = vec![Message::system(PLAN_PROMPT), Message::user(user_input)];
        let mut req = ChatRequest::new(&self.cfg.model, messages);
        if let Some(t) = self.cfg.temperature {
            req = req.with_temperature(t);
        }
        apply_quirks(&self.cfg.provider, &self.cfg.model, &mut req);
        let resp = self.client.chat(&req).await.ok()?;
        let steps = parse_plan(&resp.content?);
        (!steps.is_empty()).then_some(steps)
    }

}

/// Stream one completion from `client`, emitting content/reasoning deltas to
/// `host` and accumulating streamed tool-call fragments by index. Returns
/// (content, tool_calls, usage). Shared by the developer loop and the
/// orchestrator's read-only chat-agent tier.
pub(crate) async fn stream_completion(
    client: &dyn LlmClient,
    req: &ChatRequest,
    host: &dyn EngineHost,
) -> Result<(String, Vec<ToolCall>, Option<Usage>)> {
    let mut stream = client.chat_stream(req).await?;
    let mut content = String::new();
    let mut accs: BTreeMap<u32, Acc> = BTreeMap::new();
    let mut usage = None;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if let Some(c) = chunk.content {
            content.push_str(&c);
            host.emit(EngineEvent::StreamChunk { chunk: c });
        }
        if let Some(r) = chunk.reasoning {
            host.emit(EngineEvent::Reasoning { chunk: r });
        }
        if let Some(td) = chunk.tool_call {
            let a = accs.entry(td.index).or_default();
            if let Some(id) = td.id {
                a.id = id;
            }
            if let Some(n) = td.name {
                a.name = n;
            }
            if let Some(f) = td.arguments_fragment {
                a.args.push_str(&f);
            }
        }
        if let Some(u) = chunk.usage {
            usage = Some(u);
        }
    }

    let tool_calls = accs
        .into_iter()
        .filter(|(_, a)| !a.name.is_empty())
        .map(|(idx, a)| ToolCall {
            id: Some(if a.id.is_empty() { format!("call_{idx}") } else { a.id }),
            kind: "function".to_string(),
            function: FunctionCall {
                name: a.name,
                arguments: if a.args.trim().is_empty() { "{}".to_string() } else { a.args },
            },
        })
        .collect();

    Ok((content, tool_calls, usage))
}

#[derive(Default)]
struct Acc {
    id: String,
    name: String,
    args: String,
}

pub(crate) fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(max).collect();
        t.push_str("\n…(truncated)");
        t
    }
}

/// Render the manual tool-calling system prompt: the protocol preamble followed
/// by each tool's name, description, and parameter schema.
fn manual_tools_prompt(schemas: &[LlmTool]) -> String {
    let mut s = String::from(MANUAL_TOOLS_PREAMBLE);
    for t in schemas {
        s.push_str(&format!(
            "\n\n### {}\n{}\nParameters (JSON Schema): {}",
            t.function.name,
            t.function.description.as_deref().unwrap_or(""),
            t.function.parameters,
        ));
    }
    s
}

/// Parse prompt-based tool calls out of a model's reply. Recognizes fenced
/// blocks (```tool_call / ```tool / ```json, or an untagged fence) whose body
/// is a JSON object `{"name": ..., "arguments": {...}}`. Falls back to treating
/// the whole reply as one bare JSON call. Returns calls with synthesized ids.
fn parse_manual_tool_calls(content: &str) -> Vec<ToolCall> {
    let mut out: Vec<ToolCall> = Vec::new();
    // Fenced segments are the odd-indexed pieces of a split on the ``` fence.
    let parts: Vec<&str> = content.split("```").collect();
    for (i, seg) in parts.iter().enumerate() {
        if i % 2 == 0 {
            continue; // text outside fences
        }
        let seg = seg.trim_start_matches(['\n', '\r']);
        let (info, body) = match seg.split_once('\n') {
            Some((first, rest)) => (first.trim(), rest),
            None => ("", seg),
        };
        let toolish = info.is_empty()
            || info.starts_with("tool_call")
            || info.starts_with("tool")
            || info.starts_with("json");
        if !toolish {
            continue;
        }
        if let Some(tc) = parse_one_call(body) {
            out.push(tc);
        }
    }
    // No fenced calls — maybe the whole reply is a bare JSON call object.
    if out.is_empty() {
        if let Some(tc) = parse_one_call(content) {
            out.push(tc);
        }
    }
    for (idx, tc) in out.iter_mut().enumerate() {
        tc.id = Some(format!("manual_{idx}"));
    }
    out
}

/// Parse a single `{"name", "arguments"}` JSON object into a [`ToolCall`].
fn parse_one_call(s: &str) -> Option<ToolCall> {
    let v: serde_json::Value = serde_json::from_str(s.trim()).ok()?;
    let name = v.get("name")?.as_str()?.trim().to_string();
    if name.is_empty() {
        return None;
    }
    let args = v.get("arguments").or_else(|| v.get("args")).cloned();
    let arguments = match args {
        Some(serde_json::Value::String(s)) => s,
        Some(other) => other.to_string(),
        None => "{}".to_string(),
    };
    Some(ToolCall {
        id: None,
        kind: "function".to_string(),
        function: FunctionCall { name, arguments },
    })
}

/// Trim older tool outputs to keep the context window small (keeps the last
/// `recent_full` messages verbatim). The full outputs were already shown to
/// the user via events; the model only needs the recent ones in detail.
fn compact_context(messages: &[Message], recent_full: usize, cap: usize) -> Vec<Message> {
    let cutoff = messages.len().saturating_sub(recent_full);
    messages
        .iter()
        .enumerate()
        .map(|(i, m)| {
            if i < cutoff && matches!(m.role, Role::Tool) {
                if let Some(content) = &m.content {
                    if content.chars().count() > cap {
                        let head: String = content.chars().take(cap).collect();
                        let mut trimmed = m.clone();
                        trimmed.content = Some(format!("{head}\n…(older tool output trimmed)"));
                        return trimmed;
                    }
                }
            }
            m.clone()
        })
        .collect()
}

/// Parse a model-produced plan (numbered/bulleted list) into step strings.
fn parse_plan(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if t.eq_ignore_ascii_case("none") {
            return Vec::new();
        }
        let s = t
            .trim_start_matches(|c: char| c.is_ascii_digit())
            .trim_start_matches([')', '.', '-', '*', ':', ' '])
            .trim();
        if s.chars().count() > 2 {
            out.push(s.to_string());
        }
        if out.len() >= 8 {
            break;
        }
    }
    out
}

#[cfg(test)]
mod plan_tests {
    use super::{compact_context, parse_plan};
    use infinidev_llm::Message;

    #[test]
    fn compacts_old_tool_output() {
        let big = "x".repeat(1000);
        let msgs = vec![
            Message::system("s"),
            Message::user("u"),
            Message::tool("c1", big.clone()), // old → trimmed
            Message::assistant("a"),
            Message::user("u2"),
            Message::tool("c2", big.clone()),
            Message::assistant("a2"),
            Message::user("u3"),
            Message::tool("c3", big.clone()), // within recent → kept
        ];
        let out = compact_context(&msgs, 3, 50);
        assert!(out[2].content.as_ref().unwrap().chars().count() < 200);
        assert!(out[8].content.as_ref().unwrap().chars().count() > 200);
    }

    #[test]
    fn parses_lists() {
        assert_eq!(
            parse_plan("1. Read the config\n2) Edit main.rs\n- run tests"),
            vec!["Read the config", "Edit main.rs", "run tests"]
        );
    }

    #[test]
    fn none_is_empty() {
        assert!(parse_plan("NONE").is_empty());
        assert!(parse_plan("none").is_empty());
    }
}

#[cfg(test)]
mod manual_tests {
    use super::{parse_manual_tool_calls, use_manual_tools};
    use crate::config::EngineConfig;

    #[test]
    fn parses_fenced_tool_call() {
        let content = "I'll read it.\n```tool_call\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.rs\"}}\n```";
        let calls = parse_manual_tool_calls(content);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read_file");
        assert!(calls[0].function.arguments.contains("a.rs"));
        assert_eq!(calls[0].id.as_deref(), Some("manual_0"));
    }

    #[test]
    fn parses_multiple_calls() {
        let content = "```tool_call\n{\"name\":\"a\",\"arguments\":{}}\n```\n```tool_call\n{\"name\":\"b\",\"arguments\":{}}\n```";
        let calls = parse_manual_tool_calls(content);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "a");
        assert_eq!(calls[1].function.name, "b");
        assert_eq!(calls[1].id.as_deref(), Some("manual_1"));
    }

    #[test]
    fn plain_text_has_no_calls() {
        assert!(parse_manual_tool_calls("Just a normal answer, no tools.").is_empty());
    }

    #[test]
    fn bare_json_object_is_a_call() {
        let calls = parse_manual_tool_calls("{\"name\":\"list_directory\",\"arguments\":{\"path\":\".\"}}");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "list_directory");
    }

    #[test]
    fn manual_mode_is_config_driven() {
        // The capability matrix marks every current provider function-calling
        // capable, so manual mode is opt-in via the config flag …
        assert!(!use_manual_tools(&EngineConfig::ollama("gemma:7b")));
        let mut cfg = EngineConfig::ollama("gemma:7b");
        cfg.force_manual_tools = true;
        assert!(use_manual_tools(&cfg));
        // … and would also auto-engage for any model the matrix reports as
        // lacking native function-calling (none today — future-proofing).
    }
}
